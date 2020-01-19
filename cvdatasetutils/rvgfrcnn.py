import os

import cvdatasetutils.restrictedgenome as rvg
from torch.utils.data import Dataset
import torch
from PIL import Image
from cvdatasetutils.imageutils import show_objects_in_image
import cvdatasetutils.transforms as T


IMAGE_EXTENSION = ".jpg"


class RestrictedVgFasterRCNN(Dataset):
    def __init__(self, dataset_folder, images_folder, transforms):
        self.ds = rvg.load_dataframe(dataset_folder)
        self.ds.image_id = self.ds.image_id.apply(int).apply(str)
        self.images = self.ds.image_id.unique()
        self.image_folder = images_folder
        self.labels = self.ds.sort_values(by='name').name.unique().tolist()
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
            'iscrowd':[]
        }

        image_path = os.path.join(self.image_folder, image_id + IMAGE_EXTENSION)

        image = Image.open(image_path).convert("RGB")
        height = image.height
        width = image.width

        for index, row in objects.iterrows():
            x = row['x'] * width
            y = row['y'] * height
            w = row['w'] * width
            h = row['h'] * height

            target['boxes'].append([x, y, x + w, y + h])
            target['labels'].append(row['name'])
            target['area'].append(w * h)
            target['iscrowd'].append(False)

        target['boxes'] = torch.FloatTensor(target['boxes'])
        target['area'] = torch.tensor(target['area'])
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int8)

        target['labels'] = [self.labels.index(id) for id in target['labels']]
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

        img, target = self.transforms(image, target)

        return img, target


def test_ds(n):
    ds = RestrictedVgFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/restrictedGenome',
                                '/home/dani/Documentos/Proyectos/Doctorado/Datasets/vgtests/images',
                                transforms=T.Compose([]))

    num_examples = 0

    for id, objects in enumerate(ds):
        show_objects_in_image('../images/', objects, "{}".format(id), "RestrictedVG", ds.labels)

        if num_examples < n:
            num_examples += 1
        else:
            break


if __name__== "__main__":
    test_ds(50)

