from cvdatasetutils.pascalvoc import load_VOC
from cvdatasetutils.visualgenome import load_visual_genome, extract_object_dataframe
from mltrainingtools.cmdlogging import section_logger
import pandas as pd
import os
from cvdatasetutils.coco import COCOSet


def extract_voc_object_data(img_url):
    def extractor(obj):
        return [
            '',
            '',
            obj['class'],
            img_url,
            obj['bx'],
            obj['by'],
            obj['h'],
            obj['w']
        ]

    return extractor

def extract_voc_object_dataframe(voc, limit=10, report=2e5):
    section = section_logger(1)
    data = []
    counter = 0

    for img in voc[0]:
        if counter > limit:
            break
        else:
            counter += 1

        if counter % report == 0:
            section("Loaded objects in {} images".format(counter))

        rows = map(extract_voc_object_data(img['filename']), img['objects'])
        data.extend(rows)

    odf = pd.DataFrame(data, columns=['object_id', 'synsets', 'names', 'img', 'x', 'y', 'h', 'w'])
    return odf


def extract_coco_object_dataframe(coco_definitions):
    annotations = coco_definitions.get_annotations()
    images = coco_definitions.get_images()

    joined_df = annotations.join(images.set_index('image_id'), on='image_id')
    joined_df['x'] = joined_df.x / joined_df.width
    joined_df['y'] = joined_df.y / joined_df.height
    joined_df['w'] = joined_df.w / joined_df.width
    joined_df['h'] = joined_df.h / joined_df.height

    return joined_df[['x', 'y', 'w', 'h', 'name', 'image_id']]


def generate_analysis(coco_path, voc_path, vg_path, output_path):
    log = section_logger()

    log('Loading definitions for COCO')
    coco_definitions = COCOSet(coco_path)

    log('Converting COCO data to dataframes')
    coco_df = extract_coco_object_dataframe(coco_definitions)

    log('Loading definitions for VOC')
    voc_definitions = load_VOC(voc_path)

    log('Converting VOC data to dataframes')
    voc_df = extract_voc_object_dataframe(voc_definitions, limit=1e8)

    log('Loading definitions for Visual Genome')
    vg_definitions = load_visual_genome(vg_path)

    log('Converting VG data to dataframes')
    vg_df = extract_object_dataframe(vg_definitions, limit=1e8)

    log('Saving dataframes')
    voc_df.to_csv(os.path.join(output_path, 'voc_df.csv'))
    vg_df.to_csv(os.path.join(output_path, 'vg_df.csv'))
    coco_df.to_csv(os.path.join(output_path, 'coco_df.csv'))


generate_analysis('/home/dani/Documentos/Proyectos/Doctorado/Datasets/COCO',
                  '/home/dani/Documentos/Proyectos/Doctorado/Datasets/VOC2012/VOCdevkit/VOC2012',
                  '/home/dani/Documentos/Proyectos/Doctorado/Datasets/VisualGenome',
                  '/home/dani/Documentos/Proyectos/Doctorado/cvdatasetutils/analytics')





