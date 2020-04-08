import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection._utils import BoxCoder

from cvdatasetutils.engine import train_one_epoch
import cvdatasetutils.utils as utils
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from cvdatasetutils.ad20kfrcnn import AD20kFasterRCNN
from time import gmtime, strftime
from cvdatasetutils.imageutils import show_objects_in_image
from cvdatasetutils.dnnutils import get_transform, clean_from_error
from mltrainingtools.cmdlogging import section_logger
import pandas as pd
from apex import amp
import time
from mltrainingtools.metaparameters import generate_metaparameters
from torch.utils.tensorboard import SummaryWriter

from cvdatasetutils.basicevaluation import evaluate_map, evaluate_masks


ANALYSIS_FILE = 'MaskRCNNAnalysis.csv'
FINETUNE_FILE = 'MaskRCNNFinetune_{}.csv'


class AnchorGeneratorHalfWrapper(AnchorGenerator):
    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device="cpu"):
        return super().generate_anchors(scales, aspect_ratios, dtype, device)

    def set_cell_anchors(self, dtype, device):
        super().set_cell_anchors(dtype, device)

    def num_anchors_per_location(self):
        return super().num_anchors_per_location()

    def grid_anchors(self, grid_sizes, strides):
        result_list = super().grid_anchors(grid_sizes, strides)
        return [t.half() for t in result_list]

    def cached_grid_anchors(self, grid_sizes, strides):
        result_list = super().cached_grid_anchors(grid_sizes, strides)
        return [t.half() for t in result_list]

    def forward(self, image_list, feature_maps):
        result_list = super().forward(image_list, feature_maps)
        return [t.half() for t in result_list]

    def __init__(self, anchor_sizes, aspect_ratios):
        super(AnchorGeneratorHalfWrapper, self).__init__(sizes=anchor_sizes, aspect_ratios=aspect_ratios)


def create_anchor_wrapper():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGeneratorHalfWrapper(
        anchor_sizes, aspect_ratios
    )

    return rpn_anchor_generator


def decode_wrapper(fn):
    def wrapper(self, rel_codes, boxes):
        result = fn(self, rel_codes, boxes)
        return result.half()

    return wrapper


def get_model_instance_segmentation(num_classes, device, mask_hidden_layer, half_precision=False):
    if half_precision:
        anchor_generator = create_anchor_wrapper()
        BoxCoder.decode = decode_wrapper(BoxCoder.decode)

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, rpn_anchor_generator=anchor_generator)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, mask_hidden_layer, num_classes)

    model = model.to(device)

    return model


def save_model(output_path, model, name):
    torch.save(model.state_dict(), os.path.join(output_path, name + '.pt'))


def generate_datasets(dataset_base, batch_size, half_precision, new_size=(600, 600), downsampling=0,
                      test_batch_size=2):
    dataset = AD20kFasterRCNN(
        os.path.join(dataset_base, 'ADE20K_CLEAN/ade20ktrain.csv'),
        os.path.join(dataset_base, 'ADE20K_2016_07_26/images/training'),
        transforms=get_transform(train=True),
        half_precision=half_precision,
        new_size=new_size,
        perc_normalization=downsampling
    )

    dataset_test = AD20kFasterRCNN(
        os.path.join(dataset_base, 'ADE20K_CLEAN/ade20ktest.csv'),
        os.path.join(dataset_base, 'ADE20K_2016_07_26/images/validation'),
        transforms=get_transform(train=False),
        is_test=True,
        labels=dataset.labels,
        half_precision=half_precision,
        new_size=new_size,
        perc_normalization=0)

    num_classes = len(dataset.labels)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test, dataset, dataset_test, num_classes


def clean_tensor(t):
    return float(t.detach().cpu().numpy())


def write_evaluation_results(model_id, alpha, batch_accumulator, batch_size, mixed,
                             results_test, results_train, epoch, time, mask_hidden, map_train, mask_train,
                             map_test, mask_test, downsampling, momentum, decay, evaluation_path):


    results = {
        'test': results_test,
        'train': results_train
    }

    data = []

    for dataset in ['train', 'test']:
        for batch_n, row in enumerate(results[dataset]):
            raw_values = [
                epoch, dataset, model_id, alpha, batch_accumulator, batch_size, mixed, time, mask_hidden,
                map_train, mask_train, map_test, mask_test, downsampling, momentum, decay
            ]
            raw_values += results[dataset][batch_n]

            data.append(raw_values)

    raw_df = pd.DataFrame(data, columns=[
        "epoch", "dataset", "model_id", "alpha", "batch_accumulator", "batch_size", "mixed", "time", "mask_hidden",
        "map_train", "mask_train", "map_test", "mask_test", "downsampling", "momentum", "decay",
        "batch_n", "loss", 'loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg',
        "gpu_total", "gpu_used", "gpu_free", "ram_current", "ram_peak"
    ])

    if os.path.exists(evaluation_path):
        raw_df.to_csv(evaluation_path, mode='a', header=None)
    else:
        raw_df.to_csv(evaluation_path, mode='a')


def execute_experiment(dataset_base, batch_size=1, alpha=0.003, num_epochs=20, mask_hidden=256,
                       half_precision=False, batch_accumulator=5, max_examples_eval=10, downsampling=0,
                       momentum=0.9, decay=0.0005, start_epoch=0,
                       model=None, data_loader=None, data_loader_test=None, model_id_finetune=None):

    log = section_logger()
    sublog = section_logger(1)

    log('Starting experiment with: Alpha = [{}], Hidden = [{}], Accumulator = [{}], '.format(alpha,
                                                                                             mask_hidden,
                                                                                             batch_accumulator) +
        'Mixed = [{}], Down = [{}], Momentum=[{}], Decay=[{}]'.format(half_precision, downsampling, momentum, decay))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    log('Generating datasets')
    if data_loader is None or data_loader_test is None:
        dataset_results = generate_datasets(dataset_base, batch_size=batch_size,
                                            half_precision=half_precision, downsampling=downsampling)

        data_loader, data_loader_test, dataset, dataset_test, num_classes = dataset_results

    log('Generating segmentation model')
    if model == None:
        model = get_model_instance_segmentation(num_classes, device, mask_hidden_layer=mask_hidden,
                                                half_precision=half_precision)

    log('Create the optimizer')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=alpha, momentum=momentum, weight_decay=decay)

    log('Create the learning rate scheduler')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if half_precision:
        log('Optimizing with AMP')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if model_id_finetune is None:
        model_id = strftime("%Y%m%d%H%M", gmtime())
    else:
        model_id = model_id_finetune

    writer = SummaryWriter('runs/MaskRCNN_{}'.format(model_id))

    for epoch in range(start_epoch, start_epoch + num_epochs):
        sublog('Training epoch [{}]'.format(epoch))
        start = time.time()
        results_train = train_one_epoch(model, optimizer, data_loader, lr_scheduler, writer, device, epoch,
                                        print_freq=100, batch_accumulator=batch_accumulator,
                                        half_precision=half_precision)
        end = time.time()

        sublog('Evaluating epoch loss [{}]'.format(epoch))
        results_test = train_one_epoch(model, optimizer, data_loader_test, lr_scheduler, writer, device, epoch,
                                       print_freq=100, batch_accumulator=batch_accumulator, test=True)

        sublog('Evaluating epoch map [{}]'.format(epoch))
        map_train, mask_train = compute_map(data_loader, device, model, max_examples_eval)
        map_test, mask_test = compute_map(data_loader_test, device, model, max_examples_eval)

        sublog('Train performance: mAP={} maskAP={}'.format(map_train, mask_train))
        sublog('Test performance: mAP={} maskAP={}'.format(map_test, mask_test))

        sublog('Writing the results')
        if model_id_finetune is not None:
            evaluation_path = os.path.join('./models', FINETUNE_FILE.format(model_id_finetune))
        else:
            evaluation_path = os.path.join('./models', ANALYSIS_FILE)

        write_evaluation_results(model_id, alpha, batch_accumulator, batch_size, half_precision,
                                 results_test, results_train, epoch, round(end - start), mask_hidden,
                                 map_train, mask_train, map_test, mask_test, downsampling, momentum, decay,
                                 evaluation_path=evaluation_path)

        save_model('./models', model, "MaskRCNN_" + model_id)


def load_frcnn(input_path, num_classes, device, mask_hidden_layers, eval=True, half_precision=True):
    model_weights = torch.load(input_path)

    if mask_hidden_layers is None:
        mask_hidden_layers = model_weights['roi_heads.mask_predictor.conv5_mask.weight'].shape[1]

    model = get_model_instance_segmentation(num_classes, device, mask_hidden_layer=mask_hidden_layers,
                                            half_precision=half_precision)
    model.load_state_dict(model_weights)

    if eval:
        model.eval()

    model = model.to(device)
    return model


def test(input_path, dataset_base, output_path, n, half_precision=False, threshold=0.3,
         segmentation_alpha=0.3, mask_threshold=0.1, test=True):

    dataset, dataset_test, model, device, data_loader, data_loader_test = load_model_and_dataset(dataset_base,
                                                                                                 None,
                                                                                                 input_path,
                                                                                                 half_precision=half_precision)
    if half_precision:
        model = amp.initialize(model, opt_level='O1')

    num_examples = 0

    if test:
        dataset_origin = data_loader_test
    else:
        dataset_origin = data_loader

    for id, objects in enumerate(dataset_origin):
        try:
            images = [image.to(device) for image in objects[0]]
            predictions = model(images)

            for image_id, ground_truth in enumerate(objects[1]):
                _, width, height = images[image_id].shape

                clean_image = images[image_id].permute(1, 2, 0).detach().cpu().numpy()

                show_objects_in_image(output_path, clean_image, ground_truth, "{}".format(id), "FasterRCNN",
                                      dataset.labels, predictions=predictions[image_id],
                                      prediction_classes=dataset.labels, threshold=threshold,
                                      segmentation_alpha=segmentation_alpha, mask_threshold=mask_threshold)

                if num_examples < n:
                    num_examples += 1
                else:
                    return
        except:
            print('Exception trying to print test image')


METAPARAMETER_DEF = {
    'hidden':
        {
            'base': 700,
            'range': 400,
            'default': 300,
            'type': 'integer'
        },
    'accumulator':
        {
            'base': 5,
            'range': 20,
            'default': 15,
            'type': 'integer'
        },
    'alpha':
        {
            'base': 1,
            'range': 1,
            'default': 1e-4,
            'type': 'smallfloat'
        },
    'downsampling':
        {
            'base': 0.3,
            'range': 1,
            'default': 1e-4,
            'type': 'smallfloat'
        },
    'momentum':
        {
            'base': 1e-3,
            'range': 0.1,
            'default': 2e-2,
            'type': 'smallfloat'
        },
    'decay':
        {
            'base': 3,
            'range': 2,
            'default':5e-4,
            'type': 'smallfloat'
        }

}


def metaparameter_experiments(metaparameter_number, dataset_base):
    for mixed in [False]:
        metaparameters = generate_metaparameters(metaparameter_number, METAPARAMETER_DEF, static=False)

        for meta_id in range(len(metaparameters['alpha'])):
            accumulator = int(metaparameters['accumulator'][meta_id])
            alpha = metaparameters['alpha'][meta_id]
            hidden = metaparameters['hidden'][meta_id]
            downsampling = metaparameters['downsampling'][meta_id]
            momentum= metaparameters['momentum'][meta_id]
            decay = metaparameters['decay'][meta_id]

            execute_experiment(dataset_base, batch_size=4, alpha=alpha, num_epochs=4, mask_hidden=hidden,
                               half_precision=mixed, batch_accumulator=accumulator, downsampling=downsampling,
                               momentum=momentum, decay=decay)


def load_model_and_dataset(dataset_base, mask_hidden_layers, model_path, batch_size=2, half_precision=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_loader, data_loader_test, dataset, dataset_test, num_classes = generate_datasets(dataset_base,
                                                                                          batch_size=batch_size,
                                                                                          half_precision=half_precision)

    model = load_frcnn(model_path, num_classes, device, mask_hidden_layers, half_precision=half_precision)

    return dataset, dataset_test, model, device, data_loader, data_loader_test


def evaluate(input_path, dataset_base, n, half_precision=False):
    log = section_logger()

    log('Loading datasets')
    dataset, dataset_test, model, device, data_loader, data_loader_test = load_model_and_dataset(dataset_base,
                                                                                                 None,
                                                                                                 input_path,
                                                                                                 half_precision=half_precision)
    if half_precision:
        model = amp.initialize(model, opt_level='O1')

    global_map, global_mask = compute_map(data_loader_test, device, model, n)

    print('Train mAP={} and the MaskAP={}'.format(global_map, global_mask))

    global_map, global_mask = compute_map(data_loader, device, model, n)
    print('Test mAP={} and the MaskAP={}'.format(global_map, global_mask))


def compute_map(data_loader, device, model, max_examples, max_oom_errors=5, error_delay=5):
    num_examples = 0
    global_map = []
    global_mask = []

    model.eval()

    oom_error = 0

    for id, objects in enumerate(data_loader):
        try:
            images = [image.to(device) for image in objects[0]]
            predictions = model(images)

            for image_id, ground_truth in enumerate(objects[1]):
                _, width, height = images[image_id].shape

                local_map, _ = evaluate_map(predictions[image_id], ground_truth, width, height)
                local_mask_value, _ = evaluate_masks(predictions[image_id], ground_truth, width, height)

                global_map.append(local_map)
                global_mask.append(local_mask_value)
        except RuntimeError as e:
            if 'out of memory' in str(e) and oom_error < max_oom_errors:
                print('| WARNING: ran out of memory, retrying batch on 5s')
                clean_from_error(error_delay, model, oom_error)
                oom_error += 1
            elif 'illegal' in str(e) and oom_error < max_oom_errors:
                print('Illegal memory access found, retrying anyway, and logging')
                oom_error += 1
            else:
                print('| Couldnt recover from OOM error or similar', e)
                raise e

        if num_examples > max_examples:
            break
        else:
            num_examples += len(images)

    global_map = sum(global_map) / num_examples
    global_mask = sum(global_mask) / num_examples
    return global_map, global_mask


def create_finetune_analysis(model_folder, model_id, analysis_df):
    analysis_file_path = os.path.join(model_folder, FINETUNE_FILE.format(model_id))

    if os.path.exists(analysis_file_path):
       return

    model_df = analysis_df[analysis_df.model_id.astype(str) == model_id]
    model_df = model_df.drop(columns=model_df.columns[0])
    model_df.to_csv(analysis_file_path, mode='a')


def extract_metaparameters(model_folder, model_file):
    model_id = model_file.split('.')[0].split('_')[1]
    path_analysis = os.path.join(model_folder, 'MaskRCNNAnalysis.csv')
    analysis_df = pd.read_csv(path_analysis)
    parameter_data = analysis_df[analysis_df.model_id.astype(str) == model_id].iloc[0]
    last_epoch = analysis_df[analysis_df.model_id.astype(str) == model_id].epoch.max()

    metaparameters = {
        'hidden': parameter_data.mask_hidden,
        'accumulator': parameter_data.batch_accumulator,
        'alpha': parameter_data.alpha,
        'downsampling': 0.3 if 'downsampling' not in parameter_data else parameter_data.downsampling,
        'momentum': 0.95 if 'momentum' not in parameter_data else parameter_data.momentum,
        'decay': 0.0005 if 'decay' not in parameter_data else parameter_data.decay,
        'last_epoch': last_epoch,
        'model_id': model_id
    }

    create_finetune_analysis(model_folder, model_id, analysis_df)

    return metaparameters


def finetune(model_folder, model_file, dataset_base, half_precision, num_epochs, fixed_metaparameters=None):
    model_path = os.path.join(model_folder, model_file)

    loaded_data = load_model_and_dataset(dataset_base, None, model_path, half_precision=half_precision)
    dataset, dataset_test, model, device, data_loader, data_loader_test = loaded_data

    metaparameters = extract_metaparameters(model_folder, model_file)

    if fixed_metaparameters is not None:
        for key, value in fixed_metaparameters.items():
            metaparameters[key] = value

    execute_experiment(dataset_base, batch_size=3, alpha=metaparameters['alpha'], num_epochs=num_epochs,
                       mask_hidden=metaparameters['hidden'], half_precision=half_precision,
                       batch_accumulator=metaparameters['accumulator'], downsampling=metaparameters['downsampling'],
                       momentum=metaparameters['momentum'], decay=metaparameters['decay'],
                       model=model, data_loader=data_loader, data_loader_test=data_loader_test,
                       start_epoch=metaparameters['last_epoch'] + 1,
                       model_id_finetune=metaparameters['model_id'])


def module_main():
    option = 1
    #dataset_base = "/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K"
    dataset_base = "/home/daniel/Documentos/Doctorado/Datasets/ADE20K"

    if option == 1:
        metaparameter_experiments(20, dataset_base)
    elif option == 2:
        evaluate('./models/MaskRCNN_202004012149.pt',
                 dataset_base,
                 5,
                 half_precision=False)
    elif option == 3:
        finetune(
            './models',
            'MaskRCNN_202004062137.pt',
            dataset_base,
            half_precision=False,
            num_epochs=10,
            fixed_metaparameters={
                'alpha': 0.005
            }
        )
    else:
        test('./models/MaskRCNN_202004062137.pt',
             dataset_base,
             '../images', 15,
             half_precision=False, test=True,
             segmentation_alpha=0.3, threshold=0.4, mask_threshold=0.1)


if __name__== "__main__":
    module_main()
