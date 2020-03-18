import os

import cvdatasetutils.ad20k as ad20k
from torch.utils.data import Dataset
import torch
from PIL import Image
import cvdatasetutils.transforms as T
from cvdatasetutils.imageutils import show_objects_in_image
import spacy
import pandas as pd
import numpy as np


IMAGE_EXTENSION = ".jpg"


def create_object_masks(segmentation):
    mask = np.array(segmentation)
    _, _, B = np.swapaxes(mask, 0, 2)

    obj_ids = np.unique(B)
    obj_ids = obj_ids[1:]

    mask_array = np.array([B == obj_id for obj_id in obj_ids]).astype(np.uint8)
    return torch.tensor(np.swapaxes(mask_array, 1, 2))


class AD20kFasterRCNN(Dataset):
    def __init__(self, dataset_folder, images_folder, transforms, is_test=False,
                 labels=None, return_segmentation=False, max_size=None, half_precision=False):
        self.ds = ad20k.load_or_dataset(dataset_folder, is_test, image_folder=images_folder)
        self.ds['image_name'] = self.ds['image_id']
        self.ds.drop(columns='image_id')

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        objects = self.ds.loc[self.ds.image_id == image_id]

        target = {
            'boxes': [],
            'labels': [],
            'image_id': torch.tensor([int(image_id)], dtype=torch.int64),
            'area': [],
            'iscrowd': []
        }

        image_path = os.path.join(self.image_folder, objects.iloc[0]['image_name'])
        segmentation_path = image_path.replace('.jpg', '_seg.png')

        image = Image.open(image_path).convert("RGB")
        segmentation = Image.open(segmentation_path)

        for index, row in objects.iterrows():
            x = row['x']
            y = row['y']
            w = row['w']
            h = row['h']

            target['boxes'].append([x, y, x + w, y + h])
            target['labels'].append(row['name'])
            target['area'].append(w * h)
            target['iscrowd'].append(False)

        target['masks'] = create_object_masks(segmentation)

        target['boxes'] = torch.FloatTensor(target['boxes'])
        target['area'] = torch.tensor(target['area'])
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int8)

        target['labels'] = [self.labels.index(id) if id in self.labels else 0 for id in target['labels']]
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

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


def test_ds(n, path):
    ds = AD20kFasterRCNN(os.path.join(path, 'ade20ktrain.csv'),
                         '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                         transforms=T.Compose([]), return_segmentation=True)

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
    test_ds(5, '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_CLEAN')
