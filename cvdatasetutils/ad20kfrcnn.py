import os

import cvdatasetutils.ad20k as ad20k
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torch
from PIL import Image
import cvdatasetutils.transforms as T
from cvdatasetutils.imageutils import show_objects_in_image
import spacy
import pandas as pd
import numpy as np
import random


IMAGE_EXTENSION = ".jpg"


def create_object_masks(segmentation, new_size):
    if new_size is not None:
        segmentation = F.resize(segmentation, new_size, interpolation=Image.NEAREST)

    mask = np.array(segmentation)
    _, _, B = np.swapaxes(mask, 0, 2)

    obj_ids = np.unique(B)
    obj_ids = obj_ids[1:]

    mask_array = np.array([B == obj_id for obj_id in obj_ids]).astype(np.uint8)
    return torch.tensor(np.swapaxes(mask_array, 1, 2))


def filter_categories(ds, filtered_categories):
    return ds[~ds['name'].isin(filtered_categories)]


def downsample(target, normalization_weights, labels):
    keep_indexes = [index for index, value in enumerate(target['labels'])
                    if value == 0 or random.random() < normalization_weights.loc[labels[value]]]

    if len(keep_indexes) > 0:
        for key in ['labels', 'boxes', 'area', 'iscrowd', 'masks']:
            target[key] = target[key][keep_indexes]

    return target


class AD20kFasterRCNN(Dataset):
    def __init__(self, dataset_folder, images_folder, transforms, is_test=False, new_size=None,
                 labels=None, return_segmentation=False, max_size=None, half_precision=False,
                 filtered_categories=['wall', 'sky', 'road', 'floor', 'ceiling', 'earth', 'grass', 'sidewalk', 'road'],
                 perc_normalization=0.3):
        
        self.ds = ad20k.load_or_dataset(dataset_folder, is_test, image_folder=images_folder)
        self.ds = filter_categories(self.ds, filtered_categories)
        
        self.ds['image_name'] = self.ds['image_id']
        self.ds.drop(columns='image_id')
        self.new_size = new_size
        self.image_folder = images_folder

        if labels is None:
            self.labels = ['unknown'] + self.ds[self.ds.name.ne('unknown')].sort_values(by='name').name.unique().tolist()
        else:
            self.labels = labels

        self.ds['image_id'] = self.ds.groupby('image_name').ngroup()

        if max_size is None:
            self.images = self.ds.image_id.unique()
        else:
            self.images = np.random.choice(self.ds.image_id.unique(), max_size)

        self.transforms = transforms
        self.return_segmentation = return_segmentation
        self.half_precision = half_precision

        if perc_normalization is not None:
            label_count = self.ds.name.value_counts()
            max_label = label_count.max()
            self.normalization_weights = (1 - perc_normalization) + perc_normalization*(1 - label_count/max_label)
        else:
            self.normalization_weights = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        objects = self.ds.loc[self.ds.image_id == image_id]
        image_path = os.path.join(self.image_folder, objects.iloc[0]['image_name'])
        segmentation_path = image_path.replace('.jpg', '_seg.png')

        image = Image.open(image_path).convert("RGB")
        previous_size = image.size

        if self.new_size is not None:
            image = F.resize(image, self.new_size)

        segmentation = Image.open(segmentation_path)

        target = self.extract_target(objects, segmentation, image_id, previous_size, self.new_size)

        if self.normalization_weights is not None:
            target = downsample(target, self.normalization_weights, self.labels)

        with torch.no_grad():
            img, target = self.transforms(image, target)

        if self.half_precision:
            img = img.half()
            new_target = {key: value.half() for key, value in target.items() if key != 'labels' and key != 'area'}
            new_target["area"] = target["area"]
            new_target["labels"] = target["labels"]
            target = new_target

        if self.return_segmentation:
            segmentation = segmentation.convert("RGB")
            seg = self.transforms(segmentation, target)
            return img, target, seg
        else:
            return img, target

    def extract_target(self, objects, segmentation, image_id, previous_size, new_size):
        with torch.no_grad():
            target = {
                'boxes': [],
                'labels': [],
                'image_id': torch.tensor([int(image_id)], dtype=torch.int64),
                'area': [],
                'iscrowd': []
            }

            for index, row in objects.iterrows():
                x = row['x']
                y = row['y']
                w = row['w']
                h = row['h']

                target['boxes'].append([x, y, x + w, y + h])
                target['labels'].append(row['name'])
                target['area'].append(w * h)
                target['iscrowd'].append(False)

            target['masks'] = create_object_masks(segmentation, new_size)
            target['boxes'] = torch.FloatTensor(target['boxes'])
            target['area'] = torch.tensor(target['area'])

            if new_size is not None:
                w_factor = new_size[0]/previous_size[0]
                h_factor = new_size[1]/previous_size[1]
                target['boxes'][:, [0, 2]] = target['boxes'][:, [0, 2]] * w_factor
                target['boxes'][:, [1, 3]] = target['boxes'][:, [1, 3]] * h_factor
                target['area'] = (target['area'] * w_factor * h_factor).round()

            target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int8)
            target['labels'] = [self.labels.index(id) if id in self.labels else 0 for id in target['labels']]
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

        return target


def test_ds(n, path):
    ds = AD20kFasterRCNN(os.path.join(path, 'ade20ktrain.csv'),
                         '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                         transforms=T.Compose([]), return_segmentation=True,
                         new_size=(600, 600))

    num_examples = 0

    for img_id, objects in enumerate(ds):
        image = objects[0]
        ground_truth = objects[1]

        show_objects_in_image('../images/', image, ground_truth, "{}".format(img_id), "ADE20K", ds.labels)

        if num_examples < n:
            num_examples += 1
        else:
            break


def nlp_test():
    ds = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
                         '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                         transforms=T.Compose([]))

    labels = ds.labels[1:]

    nlp = spacy.load('en_core_web_md')
    tokens = nlp(" ".join(labels))
    similarities = [(l1, l2, l1.similarity(l2)) for l1 in tokens for l2 in tokens]

    similarities_df = pd.DataFrame(similarities, columns=['label1', 'label2', 'similarity'])
    similarities_df.to_csv('../analytics/ad20k_similarities.csv')


if __name__== "__main__":
    test_ds(10, '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_CLEAN')
