import torchvision
import torch.nn as nn
import json
import cvdatasetutils.transforms as T


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_pretrained(num_classes, backbone, finetune=False, remove_linear=True, finetune_skipping=None):
    if backbone == "RESNET101":
        print("Using RESNET 101 as the Backbone")
        feature_extractor = torchvision.models.resnet101(pretrained=True)

        if finetune_skipping is None:
            finetune_skipping = 290

        freeze_layers(feature_extractor, finetune, finetune_skipping)

        num_ftrs = feature_extractor.fc.in_features
        feature_extractor.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 224
    else:
        print("Using VGG16 as the Backbone")
        feature_extractor = torchvision.models.vgg16(pretrained=True)

        if finetune_skipping is None:
            finetune_skipping = 25

        freeze_layers(feature_extractor, finetune, finetune_skipping)

        if remove_linear:
            feature_extractor.classifier = nn.Identity()
            feature_extractor.avgpool = nn.Sequential(*[
                nn.Conv2d(512, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ])
        else:
            num_ftrs = feature_extractor.classifier[6].in_features
            feature_extractor.classifier[6] = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    return input_size, feature_extractor


def freeze_layers(feature_extractor, finetune, finetune_skipping):
    current = 0
    for param in feature_extractor.parameters():
        if finetune and finetune_skipping >= current:
            param.requires_grad = False
        else:
            param.requires_grad = finetune

        current += 1


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)