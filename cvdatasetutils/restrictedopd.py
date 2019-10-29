from mltrainingtools.cmdlogging import section_logger
from cvdatasetutils.coco import COCOSet
from cvdatasetutils.cocoopd import extract_global_objs
from cvdatasetutils.restrictedgenome import load_dataframe
from cvdatasetutils.vgopd import save_raw_data, filter_top_objects, split_distributions, save_distributions, convert_to_dataframe


def extract_objects(data):
    obj_df = data[['name', 'image_id']].groupby('name').count()
    obj_df['ocurrences'] = data[['name', 'image_id']].groupby('name').image_id.nunique()
    obj_df.reset_index(level=0, inplace=True)
    obj_df.reset_index(level=0, inplace=True)
    obj_df.columns = ['id', 'name', 'number', 'occurrences']
    obj_df = obj_df.astype({'number': 'int32'})

    return obj_df


def extract_distributions(data):
    log = section_logger(1)

    log('Extracting distributions ')
    raw_data = data[['image_id', 'name']].reset_index()
    raw_data.columns = ['class', 'image_id', 'name']
    raw_data = raw_data.astype({'image_id': 'int32'})

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
    obj_df = extract_objects(data)

    return img_objs, obj_df


def generate_vgopd_from_restrictedvg(output_path, input_path, top_objects, perc):
    section = section_logger()

    section('Loading RestrictedVG Dataset')
    restricted_definitions = load_dataframe(input_path)

    section('Creating distributions')
    image_df, obj_df = extract_distributions(restricted_definitions)

    section('Saving Raw DataFrame')
    save_raw_data(output_path, obj_df, image_df)

    section('Filtering objects')
    data_df = filter_top_objects(image_df, obj_df, top_objects)
    splits = split_distributions(data_df, perc)

    section('Saving final distribution')
    save_distributions(output_path, splits)


if __name__== "__main__":
    generate_vgopd_from_restrictedvg('/home/dani/Documentos/Proyectos/Doctorado/Datasets/RestrictedOPD/1000C',
                                     '/home/dani/Documentos/Proyectos/Doctorado/Datasets/restrictedGenome', 1000, 0.10)
