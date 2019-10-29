import cvdatasetutils.visualgenome as vg
import cvdatasetutils.config as conf
from mltrainingtools.cmdlogging import section_logger
import pandas as pd
import os


OBJECT_FILE_NAME = "objects.csv"


"""
Target structure for each image:
img = {
    "filename": root.find('filename').text,
    "objects": [],
    "height": int(size.find('height').text),
    "width": int(size.find('width').text)
}

Target structure for each object:

{
    'class': object.find('name').text,
    'bx': (xmin + w/2) / img['width'],
    'by': (ymin + h/2) / img['height'],
    'h': h/img['height'],
    'w': w/img['width']
}

Formato de la restricción:

* Minimum number of objects per image
* Minimum number of instances per object
* Number of objects
* NLP Processing steps
* Use of name or synset (and desambiguation policy)

"""


def get_objects_df(data, image_df):
    objects_df = pd.concat([extract_objects_df(idx, image) for idx, image in enumerate(data['objects'])])

    # Left join of objects with images
    objects_df = pd.merge(objects_df, image_df, how='left', on='image_id')

    # Normalize widths and heights
    objects_df['w'] = objects_df['w']/objects_df['width']
    objects_df['h'] = objects_df['h']/objects_df['height']
    objects_df['x'] = objects_df['x']/objects_df['width']
    objects_df['y'] = objects_df['y']/objects_df['height']

    objects_df['area'] = objects_df['w']*objects_df['h']
    return objects_df


def extract_objects_df(idx, image, log_each=25000):
    log = section_logger(1)

    obj_data = {
        'image_id': [],
        'obj_id': [],
        'label': [],
        'x': [],
        'y': [],
        'w': [],
        'h': []
    }

    if idx % log_each == 0:
        log('Logging image [{}]'.format(idx))

    for obj in image['objects']:
        if len(obj['synsets']) != 1:
            continue

        obj_data['image_id'].append(image['image_id'])
        obj_data['obj_id'].append(obj['object_id'])
        obj_data['x'].append(obj['x'])
        obj_data['y'].append(obj['y'])
        obj_data['w'].append(obj['w'])
        obj_data['h'].append(obj['h'])
        obj_data['label'].append(obj['synsets'][0])

    image_df = pd.DataFrame(data=obj_data)
    return image_df


def filter_objects(objects_df, restrictions):
    object_count = objects_df[['label', 'obj_id']].groupby(by=['label']).count()
    popular_objects = object_count[object_count.obj_id > 200]
    with_enough_instances = pd.merge(objects_df, popular_objects, left_on='label', right_index=True)
    big_objects = with_enough_instances[with_enough_instances.area > 1e-4]
    instances_per_image = big_objects[['image_id', 'obj_id_x']].groupby(by=['image_id']).count()
    dense_images = instances_per_image[instances_per_image.obj_id_x > 10]
    result_images = pd.merge(big_objects, dense_images, left_on='image_id', right_index=True)

    return result_images[['image_id', 'label', 'x', 'y', 'w', 'h']].rename(columns={'label': 'name'})


def create_images_df(data):
    image_data = {
        'image_id': [],
        'width': [],
        'height': []
    }

    for image in data['images']:
        image_data['image_id'].append(image['image_id'])
        image_data['width'].append(image['width'])
        image_data['height'].append(image['height'])

    image_df = pd.DataFrame(data=image_data)
    return image_df


def save_results(output_path, filtered_df):
    filepath = os.path.join(output_path, OBJECT_FILE_NAME)
    os.makedirs(output_path, exist_ok=True)

    filtered_df.to_csv(filepath)


def create_restricted_genome(input_path, output_path, restrictions):
    vg.set_base(input_path)
    section = section_logger()

    section('Loading Visual Genome')
    data = vg.load_visual_genome(os.path.join(input_path, conf.VG_DATA))

    section('Creating image dataframes')
    image_df = create_images_df(data)

    section('Creating object dataframes')
    objects_df = get_objects_df(data, image_df)

    section('Filter objects')
    filtered_df = filter_objects(objects_df, restrictions)

    section('Save results')
    save_results(output_path, filtered_df)


def load_dataframe(input_path):
    filepath = os.path.join(input_path, OBJECT_FILE_NAME)

    return pd.read_csv(filepath)


if __name__== "__main__":
    create_restricted_genome('/home/dani/Documentos/Proyectos/Doctorado/Datasets/vgtests',
                             '/home/dani/Documentos/Proyectos/Doctorado/Datasets/restrictedGenome', None)
