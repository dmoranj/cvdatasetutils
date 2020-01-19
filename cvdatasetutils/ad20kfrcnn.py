import os

import cvdatasetutils.ad20k as ad20k
from torch.utils.data import Dataset
import torch
from PIL import Image
import cvdatasetutils.transforms as T
from cvdatasetutils.imageutils import show_objects_in_image
import spacy
import pandas as pd


IMAGE_EXTENSION = ".jpg"


class AD20kFasterRCNN(Dataset):
    def __init__(self, dataset_folder, images_folder, transforms, is_test=False, labels=None):
        self.ds = ad20k.load_or_dataset(dataset_folder, is_test)
        self.ds['image_name'] = self.ds['image_id']
        self.ds.drop(columns='image_id')
        self.image_folder = images_folder

        if labels is None:
            self.labels = self.ds.sort_values(by='name').name.unique().tolist()
        else:
            self.labels = labels

        self.ds['image_id'] = self.ds.groupby('image_name').ngroup()
        self.images = self.ds.image_id.unique()
        self.transforms = transforms

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

        image = Image.open(image_path).convert("RGB")

        for index, row in objects.iterrows():
            x = row['x']
            y = row['y']
            w = row['w']
            h = row['h']

            target['boxes'].append([x, y, x + w, y + h])
            target['labels'].append(row['name'])
            target['area'].append(w * h)
            target['iscrowd'].append(False)

        target['boxes'] = torch.FloatTensor(target['boxes'])
        target['area'] = torch.tensor(target['area'])
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int8)

        target['labels'] = [self.labels.index(id) if id in self.labels else 0 for id in target['labels']]
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

        img, target = self.transforms(image, target)

        return img, target


def test_ds(n):
    ds = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
                         '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                         transforms=T.Compose([]))

    num_examples = 0

    for id, objects in enumerate(ds):
        image = objects[0]
        ground_truth = objects[1]

        show_objects_in_image('../images/', image, ground_truth, "{}".format(id), "ADE20K", ds.labels)

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
    test_ds(5)
