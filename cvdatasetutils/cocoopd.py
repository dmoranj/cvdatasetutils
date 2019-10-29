from mltrainingtools.cmdlogging import section_logger
from cvdatasetutils.coco import COCOSet
from cvdatasetutils.vgopd import save_raw_data, filter_top_objects, split_distributions, save_distributions, convert_to_dataframe


def extract_distributions(data):
    log = section_logger(1)

    log('Extracting distributions ')

    raw_data = data.annotations.join(data.images.set_index("image_id"), on="image_id")[['file_name', 'name', 'class']]
    raw_data.columns = ['image_id', 'name', 'class']

    frequencies = raw_data.pivot_table(
        index='image_id',
        columns='name',
        values='class',
        aggfunc='count',
        fill_value=0.0)

    frequencies['sum'] = frequencies.sum(1)
    frequencies[frequencies.columns.difference(['image_id', 'sum'])] = \
        frequencies[frequencies.columns.difference(['image_id', 'sum'])].div(frequencies["sum"], axis=0)

    img_objs = extract_global_objs(frequencies)
    obj_df = extract_objects(data.annotations)

    return img_objs, obj_df


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


def generate_vgopd_from_coco(output_path, input_path, top_objects, perc):
    section = section_logger()

    section('Loading COCO Dataset')
    coco_definitions = COCOSet(input_path)

    section('Creating distributions')
    image_df, obj_df = extract_distributions(coco_definitions)

    section('Saving Raw DataFrame')
    save_raw_data(output_path, obj_df, image_df)

    section('Filtering objects')
    data_df = filter_top_objects(image_df, obj_df, top_objects)
    splits = split_distributions(data_df, perc)

    section('Saving final distribution')
    save_distributions(output_path, splits)


if __name__== "__main__":
    generate_vgopd_from_coco('/home/dani/Documentos/Proyectos/Doctorado/Datasets/COCOOPD/1000C',
                             '/home/dani/Documentos/Proyectos/Doctorado/Datasets/COCO', 1000, 0.10)












