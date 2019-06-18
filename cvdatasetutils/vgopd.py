import cvdatasetutils.visualgenome as vg
import cvdatasetutils.config as conf
import multiset as ms
from mltrainingtools.cmdlogging import section_logger
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import numpy as np

lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)


class VGOPD(Dataset):
    """
    Visual Genome Object Probability Distributions dataset
    """

    def __init__(self, dataset_folder, images_folder, test=False):
        """

        :param dataset_folder:      folder that contains all the CSV files for the VGOPD
        :param images_folder:       folder that contains the original Visual Genome images
        """
        self.test = test
        self.images, self.pds = self.load_data(dataset_folder)
        self.images_folder = images_folder

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])

    def load_data(self, dataset_folder):
        if self.test:
            filename = conf.VGOPD_TEST_FILE
        else:
            filename = conf.VGOPD_TRAIN_FILE

        X_path = os.path.join(dataset_folder, 'X_' + filename)
        y_path = os.path.join(dataset_folder, 'y_' + filename)

        return pd.read_csv(X_path), pd.read_csv(y_path)

    def __len__(self):
        return self.pds.shape[0]

    def __getitem__(self, idx):
        safe_idx = idx % self.pds.shape[0]

        try:
            img_name = os.path.join(self.images_folder, str(self.images.iloc[safe_idx, 1]) + "." + conf.VG_IMAGE_EXTENSION)
        except:
            safe_idx = 2
            img_name = os.path.join(self.images_folder, str(self.images.iloc[safe_idx, 1]) + "." + conf.VG_IMAGE_EXTENSION)

        image = io.imread(img_name)

        if len(image.shape) != 3:
            image = np.stack((image,)*3, axis=-1)

        image = self.transforms(image)

        pds = (self.pds.iloc[safe_idx, :].as_matrix().astype('float32'))[1:]

        return image, pds

    def get_labels(self):
        return self.pds.columns


def simplify(name):
    return lemmatizer(name, "NOUN")[0]


def check_image_exists(id):
    img_path = os.path.join(os.path.join(vg.VG_BASE, conf.VG_IMAGES), str(id) + "." + conf.VG_IMAGE_EXTENSION)
    return os.path.isfile(img_path)


def extract_distributions(data, max_imgs=conf.MAX_LOADED_IMAGES):
    log = section_logger(1)

    log('Extracting distributions ')

    global_objects = ms.Multiset()
    images = []

    for i, image in enumerate(data['objects']):
        if i > max_imgs:
            break;

        if not check_image_exists(image['image_id']):
            print('Skipping unexistent image: {}.jpg'.format(image['image_id']))

        if i % 10000 == 0:
            log("Processing image {}".format(i))

        image_objs = {}
        image_objs['id'] = image['image_id']

        objs = ms.Multiset()
        objs_pd = {}
        total = 0

        for j, obj in enumerate(image['objects']):
            if len(obj['names']) > 0:
                obj_name = obj['names'][0]
                obj_name = simplify(obj_name)
                total += 1

                global_objects.add(obj_name)
                objs.add(obj_name)

        for key in objs.distinct_elements():
            objs_pd[key] = objs.get(key, 0) / total

        image_objs['pds'] = objs_pd

        images.append(image_objs)

    return global_objects, images


def convert_to_dataframe(data_pd, global_objects):
    # Create dataframe of objects
    obj_data = {
        'id': [],
        'name': [],
        'number': []
    }

    for i, obj in enumerate(global_objects.distinct_elements()):
        obj_data['id'].append(i)
        obj_data['name'].append(obj)
        obj_data['number'].append(global_objects.get(obj, 0))

    obj_df = pd.DataFrame(data=obj_data)

    # Create dataframe of image to object pds
    image_data = {
        'image_id': [],
        'name': [],
        'p': []
    }

    for j, img in enumerate(data_pd):
        for p, img_obj in enumerate(img['pds'].keys()):
            image_data['image_id'].append(img['id'])
            image_data['name'].append(img_obj)
            image_data['p'].append(img['pds'].get(img_obj))

    image_df = pd.DataFrame(data=image_data)

    return obj_df, image_df


def save_raw_data(output_path, obj_df, image_df):
    obj_df.to_csv(os.path.join(output_path, conf.GLOBAL_OBJECTS_FILE))
    image_df.to_csv(os.path.join(output_path, conf.IMAGE_OBJ_PD_FILE))


def filter_top_objects(image_df, global_objects, object_number):
    selected_objects = global_objects.sort_values(by=['number'], ascending=False).head(object_number)
    filtered = pd.merge(image_df, selected_objects, on='name', how='right')
    sums = filtered.groupby(['image_id']).sum()

    filtered = pd.merge(filtered, sums, on='image_id', how='left')
    filtered['pd'] = filtered['p_x'] / filtered['p_y']

    filtered = filtered[['image_id', 'name', 'pd']]
    filtered = filtered.pivot(index='image_id', columns='name', values='pd').fillna(0).reset_index()

    return filtered

def split_distributions(data_df, perc):
    X = data_df.loc[:, 'image_id']
    Y = data_df.loc[:, data_df.columns != 'image_id']

    return train_test_split(X, Y, test_size=perc)


def save_distributions(output_path, splits):
    X_train, X_test, y_train, y_test = splits

    X_test.to_csv(os.path.join(output_path, "X_" + conf.VGOPD_TEST_FILE))
    X_train.to_csv(os.path.join(output_path, "X_" + conf.VGOPD_TRAIN_FILE))

    y_test.to_csv(os.path.join(output_path, "y_" + conf.VGOPD_TEST_FILE))
    y_train.to_csv(os.path.join(output_path, "y_" + conf.VGOPD_TRAIN_FILE))


def generate_vgopd_from_vg(output_path, input_path, top_objects, perc):
    section = section_logger()

    section('Loading Visual Genome')
    data = vg.load_visual_genome(input_path)

    section('Creating distributions')
    global_objects, data_pd = extract_distributions(data)

    section('Converting to DataFrame')
    obj_df, image_df = convert_to_dataframe(data_pd, global_objects)
    save_raw_data(output_path, obj_df, image_df)

    section('Filtering objects')
    data_df = filter_top_objects(image_df, obj_df, top_objects)
    splits = split_distributions(data_df, perc)

    section('Saving final distribution')
    save_distributions(output_path, splits)


#generate_vgopd_from_vg(conf.DATA_FOLDER, conf.VG_BASE, conf.TOP_OBJECTS, conf.SPLIT_DISTRIBUTION)