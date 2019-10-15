from mltrainingtools.cmdlogging import section_logger
from cvdatasetutils.vgopd import save_raw_data, filter_top_objects, split_distributions, save_distributions, convert_to_dataframe
from cvdatasetutils.pascalvoc import PascalVOCOR
import pandas as pd
from cvdatasetutils.cocoopd import extract_global_objs, extract_objects


def extract_distributions(voc_definitions):

    data = pd.DataFrame(columns=['image_id', 'name'])

    for image in voc_definitions.voc:
        for obj in image['objects']:
            data = data.append({
                'image_id': image['filename'],
                'name': obj['class'],
                'class': voc_definitions.classes.index(obj['class'])
            }, ignore_index=True)

    frequencies = data[['image_id', 'name', 'class']].pivot_table(
        index='image_id',
        columns='name',
        values='class',
        aggfunc='count',
        fill_value=0.0)

    frequencies['sum'] = frequencies.sum(1)
    frequencies[frequencies.columns.difference(['image_id', 'sum'])] = \
        frequencies[frequencies.columns.difference(['image_id', 'sum'])].div(frequencies["sum"], axis=0)

    img_obj = extract_global_objs(frequencies)
    obj_df = extract_objects(data)

    return img_obj, obj_df


def generate_vgopd_from_coco(output_path, input_path, top_objects, perc):
    section = section_logger()

    section('Loading VOC Dataset')
    voc_definitions = PascalVOCOR(input_path)

    section('Creating distributions')
    image_df, obj_df = extract_distributions(voc_definitions)

    section('Saving Raw DataFrame')
    save_raw_data(output_path, obj_df, image_df)

    section('Filtering objects')
    data_df = filter_top_objects(image_df, obj_df, top_objects)
    splits = split_distributions(data_df, perc)

    section('Saving final distribution')
    save_distributions(output_path, splits)


if __name__== "__main__":
    generate_vgopd_from_coco('/home/dani/Documentos/Proyectos/Doctorado/Datasets/VOCOPD/1000C',
                             '/home/dani/Documentos/Proyectos/Doctorado/Datasets/VOC2012/VOCdevkit/VOC2012', 1000, 0.10)




