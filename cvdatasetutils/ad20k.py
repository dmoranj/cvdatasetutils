from mltrainingtools.cmdlogging import section_logger
import os
from glob import glob
import pandas as pd
from skimage import io
import scipy.io
from cvdatasetutils.vgopd import save_raw_data, filter_top_objects, split_distributions, save_distributions, convert_to_dataframe
import numpy as np
import spacy

nlp = spacy.load("en_core_web_md")


TRAINING_PATH = 'images/training'
TEST_PATH = 'images/validation'
TEST_FILE_NAME = 'ade20ktest.csv'
TRAIN_FILE_NAME = 'ade20ktrain.csv'
GLOBAL_FILE_NAME = 'ade20kglobal.csv'
GLOBAL_DATA_FILE = 'index_ade20k.mat'
MIN_PIXELS = 5

def load_attributes(image_id, category, attribute_file_path):
    object_rows = []

    with open(attribute_file_path, 'r') as f:
        contents = f.read()
        lines = contents.split('\n')

        for line in lines:
            columns = [column.strip() for column in line.split('#')]
            columns

            if len(columns) != 6 or columns[1] != '0':
                continue

            clean_name = extract_single_label(columns[3])

            object_rows.append([category, image_id, columns[0], clean_name, 0, 0, 0, 0, 0, 0, 0])

    objects_df = pd.DataFrame(object_rows, columns=['category', 'imageId', 'objId', 'label',
                                                    'x', 'y', 'h', 'w', 'labelId',
                                                    'imgHeight', 'imgWidth'])

    return objects_df


def extract_single_label(labels):
    if ',' in labels:
        clean_name = labels[:labels.index(',')]
    else:
        clean_name = labels

    clean_name = " ".join([token.lemma_ for token in nlp(str(clean_name))])

    return clean_name


def decode_mask_label(R, G):
    return int((R/10)*256 + G)


def generate_box_information(category_folder, category, image_id, labels):
    boxes, height, width, segmentation = extract_boxes(category_folder, image_id)
    box_items = list(boxes.items())
    box_items.sort(key=lambda box: box[0])

    rows = []

    for index, box in box_items:
        row = {}
        row['image_id'] = image_id
        row['segmentation'] = segmentation
        row['category'] = category
        row['instance_id'] = box['instance_id']
        row['x'] = box['min_w']
        row['y'] = box['min_h']
        row['w'] = box['max_w'] - box['min_w']
        row['h'] = box['max_h'] - box['min_h']
        row['labelId'] = box['label'] - 1
        row['label'] = extract_single_label(labels[row['labelId']])
        row['imgHeight'] = height
        row['imgWidth'] = width

        rows.append(row)

    return pd.DataFrame(rows)


def extract_boxes(category_folder, image_id):
    object_mask_filename = image_id + '_seg.png'
    object_mask_path = os.path.join(category_folder, object_mask_filename)
    image = io.imread(object_mask_path)
    height, width, _ = image.shape
    boxes = {}
    for i in range(height):
        for j in range(width):
            label = decode_mask_label(image[i, j, 0], image[i, j, 1])
            instance_id = image[i, j, 2]

            if instance_id == 0:
                continue

            if not instance_id in boxes.keys():
                boxes[instance_id] = {
                    'min_h': i,
                    'max_h': i,
                    'min_w': j,
                    'max_w': j,
                    'instance_id': instance_id,
                    'label': label
                }
            else:
                if i < boxes[instance_id]['min_h']:
                    boxes[instance_id]['min_h'] = i
                if i > boxes[instance_id]['max_h']:
                    boxes[instance_id]['max_h'] = i
                if j < boxes[instance_id]['min_w']:
                    boxes[instance_id]['min_w'] = j
                if j > boxes[instance_id]['max_w']:
                    boxes[instance_id]['max_w'] = j

    return boxes, height, width, object_mask_filename


def load_df_from(input_path, labels, max_images=3e1):
    index_folders = os.listdir(input_path)
    section = section_logger(1)
    num_images = 0
    result_df = None

    for folder_id, index in enumerate(index_folders):
        index_folder = os.path.join(input_path, index)

        category_folders = os.listdir(index_folder)

        for category_id, category in enumerate(category_folders):
            category_folder = os.path.join(index_folder, category)
            category_dfs = []

            section('Processing category {} [{}/{}] of folder [{}/{}]'.format(category, category_id,
                                                                              len(category_folders),
                                                                              folder_id, len(index_folders)))

            for attribute_file_path in glob(os.path.join(category_folder, '*.txt')):
                attribute_file_name = attribute_file_path[attribute_file_path.rindex('/')+1:]
                image_id = attribute_file_name[:attribute_file_name.index('.')]
                image_id = image_id[:image_id.rindex('_')]
                box_df = generate_box_information(category_folder, category, image_id, labels)

                category_dfs.append(box_df)
                num_images += 1

            if result_df is not None:
                category_dfs.append(result_df)

            result_df = pd.concat(category_dfs)

            if num_images > max_images:
                return result_df

    return result_df


def save_df(output_path, training_df, test_df, global_info):
    os.makedirs(output_path, exist_ok=True)

    test_df_path = os.path.join(output_path, TEST_FILE_NAME)
    train_df_path = os.path.join(output_path, TRAIN_FILE_NAME)
    global_info_path = os.path.join(output_path, GLOBAL_FILE_NAME)

    training_df.to_csv(train_df_path)
    test_df.to_csv(test_df_path)
    global_info.to_csv(global_info_path)


def load_mat_file(dataset_path):
    mat = scipy.io.loadmat(os.path.join(dataset_path, GLOBAL_DATA_FILE))
    data = {
        'labels': get_mat_element(mat, 6),
        'text_2': get_mat_element(mat, 10),
        'text_3': get_mat_element(mat, 11),
        'text_4': get_mat_element(mat, 12),
        'description': get_mat_element(mat, 13),
        'text_5': get_mat_element(mat, 14)
    }

    data_instances = {
        'image': get_mat_element(mat, 0),
        'relative_path': get_mat_element(mat, 1),
        'category': get_mat_element(mat, 8)
    }

    return pd.DataFrame(data), pd.DataFrame(data_instances)


def get_mat_element(mat, n):
    if isinstance(mat['index'][0][0][n][0][0], np.uint8):
        return [element[0] if len(element) > 0 else '' for element in mat['index'][0][0][n]]
    elif isinstance(mat['index'][0][0][n][0][0], np.float32):
        return [element if len(element) > 0 else '' for element in mat['index'][0][0][n][0]]
    else:
        return [element[0] if len(element) > 0 else '' for element in mat['index'][0][0][n][0]]


def balance_training_dataset(training_df, max_categories):
    most_frequent_ids = training_df\
                            .groupby("labelId")\
                            .count()['instance_id']\
                            .sort_values(ascending=False)\
                            .iloc[:max_categories]\
                            .index.to_list()

    training_df.loc[~training_df['labelId'].isin(most_frequent_ids), 'labelId'] = 0
    training_df.loc[training_df['labelId'] == 0, 'label'] = 'unknown'

    return training_df.drop('labelId', axis=1)


def generate_or_dataset_from_ad20k(dataset_path, output_path, max_categories, max_images=1e9):
    section = section_logger()

    section('Loading Global Mat file')
    labels_info, instances_info = load_mat_file(dataset_path)
    labels = extract_labels_from_global_info(labels_info)

    section('Loading Training data')
    training_df = load_df_from(os.path.join(dataset_path, TRAINING_PATH), labels, max_images=max_images)

    section('Balance Training dataset')
    training_df = balance_training_dataset(training_df, max_categories=max_categories)

    section('Loading Test data')
    test_df = load_df_from(os.path.join(dataset_path, TEST_PATH), labels, max_images=max_images)

    section('Saving data to CSV')
    save_df(output_path, training_df, test_df, labels_info)


def extract_labels_from_global_info(labels_info):
    return labels_info.iloc[:, 0]


def extract_objects(data):
    obj_df = data[['name', 'image_id']].groupby('name').count()
    obj_df['ocurrences'] = data[['name', 'image_id']].groupby('name').image_id.nunique()
    obj_df.reset_index(level=0, inplace=True)
    obj_df.reset_index(level=0, inplace=True)
    obj_df.columns = ['id', 'name', 'number', 'occurrences']
    return obj_df


def extract_global_objs(frequencies):
    raw_data = frequencies[frequencies.columns.difference(['sum'])]
    raw_data.reset_index(level=0, inplace=True)
    img_objs = raw_data.melt(id_vars=['image_id'])
    img_objs = img_objs[img_objs.value != 0]
    img_objs.columns = ['image_id', 'name', 'p']
    return img_objs


def extract_distributions(raw_data):
    log = section_logger(1)

    log('Extracting distributions ')

    data = raw_data.loc[:, ['image_id', 'name', 'class']]

    frequencies = data.pivot_table(
        index='image_id',
        columns='name',
        values='class',
        aggfunc='count',
        fill_value=0.0)

    frequencies['sum'] = frequencies.sum(1)
    frequencies[frequencies.columns.difference(['image_id', 'sum'])] = \
        frequencies[frequencies.columns.difference(['image_id', 'sum'])].div(frequencies["sum"], axis=0)

    img_objs = extract_global_objs(frequencies)
    obj_df = extract_objects(raw_data)

    return img_objs, obj_df


def compute_file_column(input_path, raw_data, is_test, image_folder=None):
    if is_test:
        img_subfolder = 'validation'
    else:
        img_subfolder = 'training'

    base_url = input_path[:input_path.rindex('/')]

    if image_folder is None:
        image_folder = os.path.join(base_url, 'images/' + img_subfolder)
    else:
        image_folder = image_folder

    tentative_values = raw_data['category'].str.slice(stop=1) + '/' + raw_data['category'].astype(str) + '/' + raw_data['filename']

    return [file if os.path.isfile(os.path.join(image_folder, file)) else os.path.join('outliers', file[2:]) for file in tentative_values]


def load_or_dataset(input_path, is_test=False, image_folder=None):
    raw_data = pd.read_csv(input_path)
    raw_data = raw_data.loc[(raw_data.h > MIN_PIXELS) & (raw_data.w > MIN_PIXELS)]
    raw_data['image_name'] = raw_data['image_id']
    pattern = r'.*_(\d+).jpg'
    raw_data['filename'] = raw_data['image_id'] + '.jpg'
    raw_data['image_name'] = raw_data.loc[:, 'filename'].str.extract(pattern)

    raw_data['image_id'] = compute_file_column(input_path, raw_data, is_test, image_folder)
    raw_data['image_seg'] = raw_data['image_id'].replace('.jpg$', '_seg.png', regex=True)

    raw_data = raw_data.reset_index()
    raw_data['name'] = raw_data['label']
    raw_data['area'] = raw_data['w'] * raw_data['h']
    results_df = raw_data.loc[:, ['area', 'instance_id', 'x', 'y', 'w', 'h', 'name', 'image_id', 'image_seg']]
    return results_df


def generate_opd_from_ad20k(input_path, output_path, top_objects, perc):
    section = section_logger()

    section('Loading AD20k Dataset')
    ad20k_definitions = load_or_dataset(input_path)

    section('Creating distributions')
    image_df, obj_df = extract_distributions(ad20k_definitions)

    section('Saving Raw DataFrame')
    save_raw_data(output_path, obj_df, image_df)

    section('Filtering objects')
    data_df = filter_top_objects(image_df, obj_df, top_objects)
    splits = split_distributions(data_df, perc)

    section('Saving final distribution')
    save_distributions(output_path, splits)


if __name__ == '__main__':
    #generate_opd_from_ad20k('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
    #                        '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20KOPD/1000C', 1000, 0.10)
    generate_or_dataset_from_ad20k('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26',
                                   '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_CLEAN',
                                   300)
