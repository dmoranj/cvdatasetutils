from torch.utils.data import Dataset
from cvdatasetutils.imageutils import load_image
import cvdatasetutils.config as conf
import os
import xml.etree.ElementTree as ET
from multiset import Multiset

VOC_CLASSES = 20

class PascalVOCOR(Dataset):
    def __init__(self, dataset_folder):
        annotations, classes = load_VOC(dataset_folder)
        self.voc = annotations
        self.classes = classes
        self.images_folder = os.path.join(dataset_folder, conf.VOC_IMAGES)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        annotations = self.voc[idx]

        image, image_np = load_image(os.path.join(self.images_folder, annotations['filename']))

        return image_np, annotations

    def get_raw(self):
        return self.voc

    def get_class_list(self):
        return self.classes


def parse_annotation(dataset_folder, annotation):
    with open(os.path.join(dataset_folder, conf.VOC_ANNOTATIONS, annotation), "r") as a:
        annotation_data = a.read()

        root = ET.fromstring(annotation_data)

        size = root.find('size')
        img = {
            "filename": root.find('filename').text,
            "objects": [],
            "height": int(size.find('height').text),
            "width": int(size.find('width').text)
        }

        for object in root.findall('object'):
            bndbox = object.find('bndbox')
            xmax = float(bndbox.find('xmax').text)
            xmin = float(bndbox.find('xmin').text)
            ymax = float(bndbox.find('ymax').text)
            ymin = float(bndbox.find('ymin').text)

            h = ymax - ymin
            w = xmax - xmin

            obj = {
                'class': object.find('name').text,
                'bx': (xmin + w/2) / img['width'],
                'by': (ymin + h/2) / img['height'],
                'h': h/img['height'],
                'w': w/img['width']
            }

            img['objects'].append(obj)

        return img


def load_VOC(dataset_folder):
    annotation_files = os.listdir(os.path.join(dataset_folder, conf.VOC_ANNOTATIONS))

    images = []
    classes = Multiset()

    for annotation in annotation_files:
        img = parse_annotation(dataset_folder, annotation)

        classes = classes.combine([o['class'] for o in img['objects']])

        images.append(img)

    return images, list(classes.distinct_elements())


