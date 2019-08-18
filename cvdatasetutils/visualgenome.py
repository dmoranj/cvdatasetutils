import os
import zipfile
import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mltrainingtools.cmdlogging import section_logger
from nltk.corpus import wordnet
import glob
import requests
import cvdatasetutils.config as cf


VG_BASE = './'


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_visual_genome(vgbase):
    return {
        "objects": load_json(os.path.join(vgbase, cf.VG_OBJECTS)),
        "relationships": load_json(os.path.join(vgbase, cf.VG_RELATIONSHIPS))
    }


def get_image_info(vg, id):
    return {
        "url": vg['objects'][id]['image_url'],
        "objects": vg['objects'][id]['objects'],
        "relationships": vg['relationships'][id]['relationships']
    }


def create_image_graph(image):
    G = nx.DiGraph()

    for r in image['relationships']:
        G.add_edge(r['object']['object_id'], r['subject']['object_id'], predicate=r['predicate'], synsets=r['synsets'])

    for o in image['objects']:
        if 'synsets' in o and len(o['synsets']) > 0:
            G.add_node(o['object_id'], synsets=o['synsets'])
        else:
            G.add_node(o['object_id'], synsets=o['names'])

    return G


def visualize_image_graph(ig):
    labels = {k: v['synsets'][0] for k, v in ig._node.items() if 'synsets' in v and len(v['synsets']) > 0}
    nx.draw_networkx(ig, with_labels=True, font_weight='bold', labels=labels)
    plt.show()


def extract_relationship_data(img_id):
    def extractor(r):
        return [
            img_id,
            r['object']['object_id'],
            r['subject']['object_id'],
            r['predicate'],
            "|".join(r['synsets'])
        ]

    return extractor


def extract_relationship_dataframe(vg, limit=10, report=2e5):
    section = section_logger(1)
    data = []

    counter = 0
    for img in vg['relationships']:
        if counter > limit:
            break
        else:
            counter += 1

        if counter % report == 0:
            section("Loaded relationships in {} images".format(counter))

        rows = map(extract_relationship_data(img['image_id']), img['relationships'])

        data.extend(list(rows))

    rdf = pd.DataFrame(data, columns=['img', 'object', 'subject', 'predicate', 'synsets'])

    return rdf


def extract_object_data(img_id):
    def extractor(obj):
        return [
            obj['object_id'],
            "|".join(obj['synsets']),
            "|".join(obj['names']),
            img_id,
            obj['x'],
            obj['y'],
            obj['h'],
            obj['w']
        ]

    return extractor


def extract_object_dataframe(vg, limit=10, report=2e5):
    section = section_logger(1)
    data = []
    counter = 0

    for img in vg['objects']:
        if counter > limit:
            break
        else:
            counter += 1

        if counter % report == 0:
            section("Loaded objects in {} images".format(counter))

        rows = map(extract_object_data(img['image_id']), img['objects'])
        data.extend(rows)

    odf = pd.DataFrame(data, columns=['object_id', 'synsets', 'names', 'img', 'x', 'y', 'h', 'w'])
    return odf


def add_key_to_dict(id, collection):
    if id in collection:
        id_obj = collection[id]
    else:
        id_obj = len(collection)
        collection[id] = id_obj

    return id_obj


def extract_knowledge_graph(vg, limit=10, report=2e5):

    section = section_logger(1)

    counter = 0

    synset_ids = {}

    knowledge_graph = nx.DiGraph()

    for img in vg['relationships']:
        if counter > limit:
            break
        else:
            counter += 1

        if counter % report == 0:
            section("Loaded relationships in {} images".format(counter))

        for r in img['relationships']:
            object = r['object']['synsets']
            subject = r['subject']['synsets']
            predicate = r['synsets']

            for syn_obj in object:
                for syn_sub in subject:
                    id_obj = add_key_to_dict(syn_obj, synset_ids)
                    id_sub = add_key_to_dict(syn_sub, synset_ids)

                    if len(predicate) == 1:
                        predicate_str = predicate[0]
                    else:
                        predicate_str = 'relation.n.01'

                    knowledge_graph.add_edge(id_obj, id_sub, predicate=predicate_str)

    return knowledge_graph, synset_ids


def extract_data_analytics():
    section = section_logger()

    section('Extracting Visual Genome')
    vg = load_visual_genome()

    section('Generating DataFrames from the object files')
    odf = extract_object_dataframe(vg, int(1e6))

    section('Generating DataFrames from the relationship files')
    rdf = extract_relationship_dataframe(vg, int(1e6))

    section('Saving CSVs')
    odf.to_csv(os.path.join(cf.VG_ANALYTICS, 'vg_objects.csv'))
    rdf.to_csv(os.path.join(cf.VG_ANALYTICS, 'vg_relationships.csv'))


def extract_relations_wordnet(graph, syn_id, synset_ids, wordnet_syn, relation, predicate):
    hypernyms = wordnet_syn.__getattribute__(relation)()

    for hyp in hypernyms:
        hyp_name = hyp.name()

        if hyp_name in synset_ids:
            hyp_id = synset_ids[hyp_name]
        else:
            hyp_id = add_key_to_dict(hyp_name, synset_ids)

        graph.add_edge(syn_id, hyp_id, predicate=predicate)

    return graph, synset_ids


def add_wordnet_synsets(graph, synset_ids, report=2e5):
    old_synset_ids = dict(synset_ids)
    section = section_logger(1)

    counter = 0
    for syn_name, syn_id in old_synset_ids.items():
        counter += 1

        if counter % report == 0:
            section("Loaded relationships in {} images".format(counter))

        wordnet_syn = wordnet.synset(syn_name)

        extract_relations_wordnet(graph, syn_id, synset_ids, wordnet_syn, 'hypernyms', 'generalize.v.01')
        extract_relations_wordnet(graph, syn_id, synset_ids, wordnet_syn, 'hyponyms', 'specialize.v.01')

    return graph, synset_ids


def save_graph_info(graph, ids):
    nx.write_gml(graph, os.path.join(cf.VG_ANALYTICS, 'knowledge_graph.gml'))

    with open(os.path.join(cf.VG_ANALYTICS, 'synset_ids.json'), 'w') as outfile:
        json.dump(ids, outfile)


def compute_knowledge_graph():
    section = section_logger()

    section('Loading JSON files from dataset')
    vg = load_visual_genome()

    section('Generating Knowledge Graph')
    vi_graph, synset_ids = extract_knowledge_graph(vg, limit=1e10)

    section('Annotating with Wordnet synsets')
    wn_graph, synset_ids = add_wordnet_synsets(vi_graph, synset_ids)

    section('Saving graph')
    save_graph_info(wn_graph, synset_ids)

    return wn_graph


def load_graph_info():
    wn_graph = nx.read_gml(os.path.join(cf.VG_ANALYTICS, 'knowledge_graph.gml'))

    with open(os.path.join(cf.VG_ANALYTICS, 'synset_ids.json'), 'r') as infile:
        ids = json.load(infile)

    return wn_graph, ids


def set_base(new_path):
    global VG_BASE

    VG_BASE=new_path


def create_vg_folder_structure():
    [os.makedirs(os.path.join(VG_BASE, cf.VG_FOLDER_STRUCTURE[folder]), exist_ok=True) for folder in cf.VG_FOLDER_STRUCTURE.keys()]


def download_files(file_list, output_path, attempts=5, chunk_size=8*1024):

    for url in file_list:
        filename = url[url.rfind('/')+1:]
        target_file = os.path.join(output_path, filename)

        if os.path.isfile(target_file) or os.path.isfile(target_file[:target_file.rfind('.zip')]):
            print('Skipping [{}]'.format(filename))
            continue

        print('\nDownloading [{}] to [{}]'.format(url, output_path))

        for i in range(attempts):
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                break
            else:
                print('Connection problem []. Retrying.'.format(response.status_code))

        if response is None or not response.status_code == 200:
            print("Couldn't download file []. Please try later.".format(url))
            continue

        current_size = 0

        if 'Content-Length' in response.headers:
            total_size = int(response.headers['Content-Length'])//1024
        else:
            total_size = 'Unknown'

        with open(target_file, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=chunk_size):
                print('\r Downloading [{:,}/{:,} Kb]'.format(current_size, total_size), end="")
                fd.write(chunk)
                fd.flush()
                current_size += len(chunk)//1024

        print('\r Downloading [{:,}/{:,} Kb]'.format(total_size, total_size), end="")


def unzip_files(input_path):
    zip_file_list = glob.glob(os.path.join(input_path, '*.zip'))

    for file in zip_file_list:
        print('Uncompress [{}]'.format(file))
        zipped = zipfile.ZipFile(file, 'r')
        zipped.extractall(input_path)
        zipped.close()

        print('Removing [{}]'.format(file))
        os.remove(file)


def touch_images(folder):
    for name in ['images', 'images2']:
        with open(os.path.join(folder, name), 'w') as f1:
            f1.write('downloaded')
            f1.close()


def download():
    create_vg_folder_structure()

    data_folder = os.path.join(VG_BASE, cf.VG_DATA)
    download_files(cf.DATA_FILES, data_folder)

    images_folder = os.path.join(VG_BASE, cf.VG_IMAGES)
    download_files(cf.IMAGE_FILES, images_folder)

    unzip_files(data_folder)
    unzip_files(images_folder)

    touch_images(images_folder)

    # The merge of folders VG_100K and VG_100K_2 is still missing


