import cvdatasetutils.visualgenome as vg
import cvdatasetutils.config as conf
import multiset as ms
from mltrainingtools.cmdlogging import section_logger
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import spacy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage import io
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
nlp = spacy.load('en_core_web_sm')


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
            img_id = self.images.iloc[safe_idx, 1]
            img_name = self.create_img_name(img_id)
        except:
            safe_idx = 2
            img_id = self.images.iloc[safe_idx, 1]
            img_name = self.create_img_name(img_id)

        image = io.imread(img_name
)
        if len(image.shape) != 3:
            image = np.stack((image,)*3, axis=-1)

        image = self.transforms(image)

        pds = (self.pds.loc[self.pds['image_id'] == img_id].iloc[:, 2:].values.astype('float32'))

        return image, pds

    def create_img_name(self, img_id):
        if conf.VG_IMAGE_EXTENSION in str(img_id):
            image_name = str(img_id)
        else:
            image_name = str(img_id) + "." + conf.VG_IMAGE_EXTENSION

        return os.path.join(self.images_folder, image_name)

    def get_labels(self):
        return self.pds.columns[2:]


def simplify(name):
    filtered = filter(lambda token: token.pos_ == "NOUN", nlp(name))
    lemmatized = map(lambda token: token.lemma_, filtered)
    return " ".join(lemmatized)


def check_image_exists(id):
    img_path = os.path.join(os.path.join(vg.VG_BASE, conf.VG_IMAGES), str(id) + "." + conf.VG_IMAGE_EXTENSION)
    return os.path.isfile(img_path)


def extract_distributions(data, max_imgs=conf.MAX_LOADED_IMAGES):
    log = section_logger(1)

    log('Extracting distributions ')

    global_objects = ms.Multiset()
    occurrences = ms.Multiset()
    images = []

    for i, image in enumerate(data['objects']):
        if i > max_imgs:
            break;

        if not check_image_exists(image['image_id']):
            print('Skipping unexistent image: {}.jpg'.format(image['image_id']))

        if i % 25 == 0:
            log("Processing image {}".format(i))

        image_objs = {}
        image_objs['id'] = image['image_id']

        objs = ms.Multiset()
        objs_pd = {}
        total = 0
        appeared = set()

        for j, obj in enumerate(image['objects']):
            if len(obj['names']) > 0:
                obj_name = obj['names'][0]
                obj_name = simplify(obj_name)

                if len(obj_name.strip()) == 0:
                    continue

                total += 1
                appeared.add(obj_name)

                global_objects.add(obj_name)
                objs.add(obj_name)

        for obj in appeared:
            occurrences.add(obj)

        for key in objs.distinct_elements():
            objs_pd[key] = objs.get(key, 0) / total

        image_objs['pds'] = objs_pd

        images.append(image_objs)

    return global_objects, occurrences, images


def convert_to_dataframe(data_pd, global_objects, occurrences):
    # Create dataframe of objects
    obj_data = {
        'id': [],
        'name': [],
        'number': [],
        'occurrences': []
    }

    for i, obj in enumerate(global_objects.distinct_elements()):
        obj_data['id'].append(i)
        obj_data['name'].append(obj)
        obj_data['number'].append(global_objects.get(obj, 0))
        obj_data['occurrences'].append(occurrences.get(obj, 0))

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
    os.makedirs(output_path, exist_ok=True)
    obj_df.to_csv(os.path.join(output_path, conf.GLOBAL_OBJECTS_FILE))
    image_df.to_csv(os.path.join(output_path, conf.IMAGE_OBJ_PD_FILE))


def filter_top_objects(image_df, global_objects, object_number):
    selected_objects = global_objects.sort_values(by=['number'], ascending=False).head(object_number)
    filtered = pd.merge(image_df, selected_objects, on='name', how='right')
    sums = filtered.groupby(['image_id']).sum()

    filtered = pd.merge(filtered, sums[['p']], on='image_id', how='left')
    filtered['pd'] = filtered['p_x'] / filtered['p_y']

    filtered = filtered[['image_id', 'name', 'pd']]
    pivoted = filtered.pivot(index='image_id', columns='name', values='pd').fillna(0).reset_index()

    return pivoted

def split_distributions(data_df, perc):
    X = data_df.loc[:, ['image_id']]

    return train_test_split(X, data_df, test_size=perc)


def save_distributions(output_path, splits):
    X_train, X_test, y_train, y_test = splits

    X_test.to_csv(os.path.join(output_path, "X_" + conf.VGOPD_TEST_FILE))
    X_train.to_csv(os.path.join(output_path, "X_" + conf.VGOPD_TRAIN_FILE))

    y_test.to_csv(os.path.join(output_path, "y_" + conf.VGOPD_TEST_FILE))
    y_train.to_csv(os.path.join(output_path, "y_" + conf.VGOPD_TRAIN_FILE))


def generate_vgopd_from_vg(output_path, input_path, top_objects, perc):
    vg.set_base(input_path)
    section = section_logger()

    section('Loading Visual Genome')
    data = vg.load_visual_genome(os.path.join(input_path, conf.VG_DATA))

    section('Creating distributions')
    global_objects, occurrences, data_pd = extract_distributions(data)

    section('Converting to DataFrame')
    obj_df, image_df = convert_to_dataframe(data_pd, global_objects, occurrences)
    save_raw_data(output_path, obj_df, image_df)

    section('Filtering objects')
    data_df = filter_top_objects(image_df, obj_df, top_objects)
    splits = split_distributions(data_df, perc)

    section('Saving final distribution')
    save_distributions(output_path, splits)


def show_example(inputs, outputs, labels, ranking=5, colormap=None):
    top_5 = np.argpartition(outputs[0, :], -ranking)[-ranking:].tolist()
    top_5_names = ", ".join([labels[p] for p in top_5])

    fig = plt.figure(frameon=False)
    image = np.array(inputs)
    image = image.transpose(1, 2, 0)
    plt.imshow(image, cmap=colormap)
    plt.text(-5, -8, top_5_names, horizontalalignment='left', verticalalignment='center', color='green')
    plt.tight_layout()
    plt.imshow(image)
    plt.show()
    cv2.waitKey(0)


def evaluate_vgopd_datasets(input_path, with_dataloader=False):
    batch_size=5
    vgopd_test = VGOPD(dataset_folder=input_path, images_folder=os.path.join(vg.VG_BASE, conf.VG_IMAGES))
    vgopd_gen = DataLoader(vgopd_test, batch_size=batch_size, shuffle=True, num_workers=7)

    if with_dataloader:
        for inputs, labels in vgopd_gen:
            for j in range(batch_size):
                show_example(inputs[j, :, :, :], labels[j, :], vgopd_test.get_labels())
    else:
        for i in range(batch_size):
            image, pd = vgopd_test[i]
            show_example(image, pd, vgopd_test.get_labels())


if __name__== "__main__":
    generate_vgopd_from_vg('/home/dani/Documentos/Proyectos/Doctorado/Datasets/VGOPD/100C',
                           '/home/dani/Documentos/Proyectos/Doctorado/Datasets/vgtests/', 100, 0.10)
