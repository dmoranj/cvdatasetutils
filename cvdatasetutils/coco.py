from cvdatasetutils.dnnutils import load_json
import os
import pandas as pd


ANNOTATION_FOLDER = 'annotations'
INSTANCES_FILE = 'instances_train2017.json'


def parse_labels(instances):
    return { category['id']: category['name'] for category in instances['categories']}


def parse_images(instances):
    def to_array(image):
        return [image['file_name'], image['height'], image['width'], image['id']]

    list_of_lists = map(to_array, instances['images'])

    return pd.DataFrame(list(list_of_lists), columns=['file_name', 'height', 'width', 'image_id'])


class COCOSet:
    def __init__(self, base_path):
        self.base_path = base_path
        instances = self.load_dataset()
        self.labels = parse_labels(instances)
        self.images = parse_images(instances)
        self.annotations = self.parse_annotations(instances)

    def load_dataset(self):
        return load_json(os.path.join(self.base_path, ANNOTATION_FOLDER, INSTANCES_FILE))

    def get_annotations(self):
        return self.annotations

    def get_images(self):
        return self.images

    def parse_annotations(self, instances):
        def to_array(annotation):
            return[annotation['area'], annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][2],
                   annotation['bbox'][3], annotation['category_id'], self.labels[annotation['category_id']],
                   annotation['image_id']]

        list_of_lists = map(to_array, instances['annotations'])

        return pd.DataFrame(list(list_of_lists), columns=['area', 'x', 'y', 'w', 'h', 'class', 'name', 'image_id'])


