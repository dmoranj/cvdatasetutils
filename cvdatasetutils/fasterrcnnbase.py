from cvdatasetutils.engine import train_one_epoch, evaluate
import cvdatasetutils.utils as utils
import torch
import cvdatasetutils.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from cvdatasetutils.rvgfrcnn import RestrictedVgFasterRCNN
import os
from time import gmtime, strftime


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def save_model(output_path, model, name):
    torch.save(model.state_dict(), os.path.join(output_path, name + '.pt'))


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person

    # use our dataset and defined transformations
    dataset = RestrictedVgFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/restrictedGenome',
                                     '/home/dani/Documentos/Proyectos/Doctorado/Datasets/vgtests/images',
                                     get_transform(train=True))

    dataset_test = RestrictedVgFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/restrictedGenome',
                                          '/home/dani/Documentos/Proyectos/Doctorado/Datasets/vgtests/images',
                                          get_transform(train=False))

    num_classes = len(dataset.labels)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-10000])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10000:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    model_id = strftime("%Y%m%d%H%M", gmtime())

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        save_model('.', model, "FasterRCNN_" + model_id)


if __name__== "__main__":
    main()