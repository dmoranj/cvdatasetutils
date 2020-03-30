import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection._utils import BoxCoder

from cvdatasetutils.engine import train_one_epoch, evaluate
import cvdatasetutils.utils as utils
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from cvdatasetutils.ad20kfrcnn import AD20kFasterRCNN
from time import gmtime, strftime
from cvdatasetutils.imageutils import show_objects_in_image
from cvdatasetutils.dnnutils import get_transform
from mltrainingtools.cmdlogging import section_logger
import pandas as pd
from apex import amp
import time
from mltrainingtools.metaparameters import generate_metaparameters
from torch.utils.tensorboard import SummaryWriter


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

    model.to(device)

    if False:
        model.half()
        model.backbone.half()
        model.roi_heads.box_predictor.half()
        model.roi_heads.mask_predictor.half()

        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torchvision.ops.misc.FrozenBatchNorm2d):
                layer.float()
            else:
                layer.half()

    return model


def save_model(output_path, model, name):
    torch.save(model.state_dict(), os.path.join(output_path, name + '.pt'))


def generate_datasets(dataset_base, batch_size, half_precision, new_size=(600, 600)):
    dataset = AD20kFasterRCNN(
        os.path.join(dataset_base, 'ADE20K_CLEAN/ade20ktrain.csv'),
        os.path.join(dataset_base, 'ADE20K_2016_07_26/images/training'),
        transforms=get_transform(train=True),
        half_precision=half_precision,
        new_size=new_size
    )

    dataset_test = AD20kFasterRCNN(
        os.path.join(dataset_base, 'ADE20K_CLEAN/ade20ktest.csv'),
        os.path.join(dataset_base, 'ADE20K_2016_07_26/images/validation'),
        transforms=get_transform(train=False),
        is_test=True,
        max_size=10,
        labels=dataset.labels,
        half_precision=half_precision,
        new_size=new_size)

    num_classes = len(dataset.labels)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test, dataset, dataset_test, num_classes


def clean_tensor(t):
    return float(t.detach().cpu().numpy())


def write_evaluation_results(model_id, alpha, batch_accumulator, batch_size, mixed,
                             results_test, results_train, epoch, time):
    evaluation_path = './models/MaskRCNNAnalysis.csv'

    results = {
        'test': results_test,
        'train': results_train
    }

    data = []

    for dataset in ['train', 'test']:
        for batch_n, row in enumerate(results[dataset]):
            raw_values = [
                epoch, dataset, model_id, alpha, batch_accumulator, batch_size, mixed, time,
            ]
            raw_values += results[dataset][batch_n]

            data.append(raw_values)

    raw_df = pd.DataFrame(data, columns=[
        "epoch", "dataset", "model_id", "alpha", "batch_accumulator", "batch_size", "mixed", "time",
        "batch_n", "loss", 'loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg',
        "gpu_total", "gpu_used", "gpu_free", "ram_current", "ram_peak"
    ])

    if os.path.exists(evaluation_path):
        raw_df.to_csv(evaluation_path, mode='a', header=None)
    else:
        raw_df.to_csv(evaluation_path, mode='a')


def execute_experiment(dataset_base, batch_size=1, alpha=0.003, num_epochs=20, mask_hidden=256,
                       half_precision=False, batch_accumulator=5):
    log = section_logger()
    sublog = section_logger(1)

    log('Starting experiment with: Alpha = [{}], Hidden = [{}], Accumulator = [{}], Mixed = [{}]'.format(
        alpha, mask_hidden, batch_accumulator, half_precision))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    log('Generating datasets')
    data_loader, data_loader_test, dataset, dataset_test, num_classes = generate_datasets(dataset_base, batch_size=batch_size,
                                                                   half_precision=half_precision)

    log('Generating segmentation model')
    model = get_model_instance_segmentation(num_classes, device, mask_hidden_layer=mask_hidden,
                                            half_precision=half_precision)

    log('Create the optimizer')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=alpha, momentum=0.9, weight_decay=0.0005)

    log('Create the learning rate scheduler')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if half_precision:
        log('Optimizing with AMP')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    model_id = strftime("%Y%m%d%H%M", gmtime())

    writer = SummaryWriter('runs/MaskRCNN')

    for epoch in range(num_epochs):
        sublog('Training epoch [{}]'.format(epoch))
        start = time.time()
        results_train = train_one_epoch(model, optimizer, data_loader, lr_scheduler, writer, device, epoch,
                                        print_freq=50, batch_accumulator=batch_accumulator,
                                        half_precision=half_precision)
        end = time.time()

        sublog('Evaluating epoch [{}]'.format(epoch))
        results_test = train_one_epoch(model, optimizer, data_loader_test, lr_scheduler, writer, device, epoch,
                                       print_freq=50, batch_accumulator=batch_accumulator, test=True)

        sublog('Writing the results')
        write_evaluation_results(model_id, alpha, batch_accumulator, batch_size, half_precision,
                                 results_test, results_train, epoch, round(end - start))

        save_model('./models', model, "MaskRCNN_" + model_id)


def load_frcnn(input_path, num_classes, device, mask_hidden_layers, eval=True):
    model = get_model_instance_segmentation(num_classes, device, mask_hidden_layer=mask_hidden_layers)
    model.load_state_dict(torch.load(input_path))

    if eval:
        model.eval()

    model.to(device)
    return model


def test(input_path, dataset_base, output_path, n, mask_hidden_layers=256):
    dataset, dataset_test, model, device, data_loader, data_loader_test = load_model_and_dataset(dataset_base, mask_hidden_layers, input_path)

    num_examples = 0

    for id, objects in enumerate(dataset_test):
        image = objects[0].to(device)
        predictions = model([image])
        ground_truth = objects[1]

        clean_image = image.permute(1, 2, 0).detach().cpu().numpy()

        show_objects_in_image(output_path, clean_image, ground_truth, "{}".format(id), "FasterRCNN",
                              dataset.labels, predictions=predictions[0],
                              prediction_classes=dataset.labels)

        if num_examples < n:
            num_examples += 1
        else:
            break


METAPARAMETER_DEF = {
    'hidden':
        {
            'base': 200,
            'range': 800,
            'default': 300,
            'type': 'integer'
        },
    'accumulator':
        {
            'base': 5,
            'range': 10,
            'default': 50,
            'type': 'integer'
        },
    'alpha':
        {
            'base': 1.2,
            'range': 2,
            'default': 1e-4,
            'type': 'smallfloat'
        }
}


def metaparameter_experiments(metaparameter_number, dataset_base):
    for mixed in [True, False]:
        metaparameters = generate_metaparameters(metaparameter_number, METAPARAMETER_DEF, static=False)

        for meta_id in range(len(metaparameters['alpha'])):
            accumulator = int(metaparameters['accumulator'][meta_id])
            alpha = metaparameters['alpha'][meta_id]
            hidden = metaparameters['hidden'][meta_id]

            execute_experiment(dataset_base, batch_size=4, alpha=alpha,
                               num_epochs=3, mask_hidden=hidden, half_precision=mixed, batch_accumulator=accumulator)


def load_model_and_dataset(dataset_base, mask_hidden_layers, model_path, batch_size=2, half_precision=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_loader, data_loader_test, dataset, dataset_test, num_classes = generate_datasets(dataset_base,
                                                                                          batch_size=batch_size,
                                                                                          half_precision=half_precision)

    model = load_frcnn(model_path, num_classes, device, mask_hidden_layers)

    return dataset, dataset_test, model, device, data_loader, data_loader_test


if __name__== "__main__":
    option = 1
    #torch.backends.cudnn.benchmark = False
    dataset_base = "/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K"

    if option == 1:
        metaparameter_experiments(6, dataset_base)
    elif option == 2:
        print("Pending...")
    else:
        test('./models/MaskRCNN_202003171107.pt',
             os.path.join(dataset_base, 'ADE20K_CLEAN'),
             '../images', 5, mask_hidden_layers=994)
