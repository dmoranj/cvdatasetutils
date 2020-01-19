from cvdatasetutils.engine import train_one_epoch, evaluate
import cvdatasetutils.utils as utils
import torch
import cvdatasetutils.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from cvdatasetutils.ad20kfrcnn import AD20kFasterRCNN
import os
from time import gmtime, strftime
from cvdatasetutils.imageutils import show_objects_in_image, IoU, save_image
import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
import numpy as np
from mltrainingtools.cmdlogging import section_logger
from mltrainingtools.metaparameters import generate_metaparameters
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import linear_sum_assignment
import networkx as nx
import json
import random
import itertools
from networkx.drawing.nx_agraph import graphviz_layout


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def save_model(output_path, model, name):
    torch.save(model.state_dict(), os.path.join(output_path, name + '.pt'))


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    # dataset = RestrictedVgFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/restrictedGenome',
    #                                  '/home/dani/Documentos/Proyectos/Doctorado/Datasets/vgtests/images',
    #                                  get_transform(train=True))
    #
    # dataset_test = RestrictedVgFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/restrictedGenome',
    #                                       '/home/dani/Documentos/Proyectos/Doctorado/Datasets/vgtests/images',
    #                                       get_transform(train=False))

    dataset = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
                              '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                              transforms=get_transform(train=True))


    dataset_test = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktest.csv',
                                   '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/validation',
                                   transforms=get_transform(train=False),
                                   is_test=True)

    num_classes = len(dataset.labels)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-10000])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-10000:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=6,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=6,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 20

    model_id = strftime("%Y%m%d%H%M", gmtime())

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        save_model('.', model, "FasterRCNN_" + model_id)


def load_frcnn(input_path, num_classes, device):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(input_path))
    model.eval()
    model.to(device)
    return model


def test(input_path, output_path, n):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
                              '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                              transforms=get_transform(train=True))

    dataset_test = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktest.csv',
                                   '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/validation',
                                   transforms=get_transform(train=False),
                                   labels=dataset.labels,
                                   is_test=True)

    num_classes = len(dataset.labels)
    model = load_frcnn(input_path, num_classes, device)

    num_examples = 0

    for id, objects in enumerate(dataset_test):
        image = objects[0].to(device)
        predictions = model([image])
        ground_truth = objects[1]
        show_objects_in_image(output_path, image, ground_truth, "{}".format(id), "FasterRCNN",
                              dataset.labels, predictions=predictions[0],
                              prediction_classes=dataset.labels)

        if num_examples < n:
            num_examples += 1
        else:
            break


def adjust_box(box, width=1, height=1):
    adjusted_box = {}
    adjusted_box['bx'] = box[0] / width
    adjusted_box['by'] = box[1] / height
    adjusted_box['w'] = (box[2] - box[0]) / width
    adjusted_box['h'] = (box[3] - box[1]) / height

    return adjusted_box


def wordnet_similarity(words_pred, words_gt):

    path_similarities = []
    wup_similarities = []

    for word_gt in words_gt.split(" "):
        synset_gt = filter_physical_synsets(wn.synsets(word_gt))

        for word_pred in words_pred.split(" "):
            synset_pred = filter_physical_synsets(wn.synsets(word_pred))
            path_similarity_list = list(filter(lambda s: s is not None, [s1.wup_similarity(s2) for s1 in synset_gt for s2 in synset_pred]))

            if len(path_similarity_list) > 0:
                path_similarities.append(max(path_similarity_list))
            else:
                path_similarities.append(0)

            wup_similarity_list = list(filter(lambda s: s is not None, [s1.path_similarity(s2) for s1 in synset_gt for s2 in synset_pred]))

            if len(wup_similarity_list) > 0:
                wup_similarities.append(max(wup_similarity_list))
            else:
                wup_similarities.append(0)

    return {
        "path": max(path_similarities),
        "wup": max(wup_similarities)
    }


def clean_stats_for_numpy(obj):
    for key in obj.keys():
        if type(obj[key]) is list and len(obj[key]) == 0:
            obj[key].append(0.0)

    return obj


def nlp_evaluation(id, objects, predictions, labels, nlp, iou_threshold=0.5, nlp_threshold=0.7, wnt_threshold=0.5):
    _, width, height = objects[0].shape
    log = section_logger(2)

    map_stats = {
        "positives": 0,
        "negatives": 0,
        "background": 0,
        "confidencesPos": [],
        "confidencesNeg": []
    }

    nlp_stats = {
        "positives": 0,
        "negatives": 0,
        "background": 0,
        "confidencesPos": [],
        "confidencesNeg": [],
        "embedding_similarities": []
    }

    wordnet_stats = {
        "positives": 0,
        "negatives": 0,
        "background": 0,
        "confidencesPos": [],
        "confidencesNeg": [],
        "path_similarities": [],
        "wup_similarities": []
    }

    for pred_id, pred_box in enumerate(predictions['boxes']):

        matched_map = False
        matched_nlp = False
        matched_wordnet = False
        confidence = float(predictions['scores'][pred_id].cpu().detach().numpy())

        for gt_id, gt_box in enumerate(objects[1]['boxes']):
            adjusted_box = adjust_box(pred_box, width, height)
            gt_box = adjust_box(gt_box, width, height)

            pred_label_id = predictions['labels'][pred_id]
            gt_label_id = objects[1]['labels'][gt_id]
            pred_label = labels[pred_label_id]
            gt_label = labels[gt_label_id]

            iou_pred = IoU(adjusted_box, gt_box)

            if iou_pred > iou_threshold and pred_label == gt_label:
                map_stats["positives"] += 1
                map_stats["confidencesPos"].append(confidence)
            else:
                map_stats["negatives"] += 1
                map_stats["confidencesNeg"].append(confidence)

            nlp_similarity = nlp_similarity_fn(nlp, gt_label, pred_label)
            nlp_stats["embedding_similarities"].append(nlp_similarity)

            if iou_pred > iou_threshold and nlp_similarity > nlp_threshold:
                nlp_stats["positives"] += 1
                nlp_stats["confidencesPos"].append(confidence)
            else:
                nlp_stats["confidencesNeg"].append(confidence)
                nlp_stats["negatives"] += 1

            wnt_similarity = wordnet_similarity(pred_label, gt_label)
            wordnet_stats["wup_similarities"].append(wnt_similarity['wup'])

            if iou_pred > iou_threshold and wnt_similarity['path'] > wnt_threshold:
                wordnet_stats["positives"] += 1
                wordnet_stats["confidencesPos"].append(confidence)
            else:
                wordnet_stats["confidencesNeg"].append(confidence)
                wordnet_stats["negatives"] += 1

            if iou_pred > iou_threshold:
                matched_map = True
                matched_nlp = True
                matched_wordnet = True

        if not matched_map:
            map_stats["background"] += 1

        if not matched_nlp:
            nlp_stats["background"] += 1

        if not matched_wordnet:
            wordnet_stats["background"] += 1

    map_stats = clean_stats_for_numpy(map_stats)
    wordnet_stats = clean_stats_for_numpy(wordnet_stats)
    nlp_stats = clean_stats_for_numpy(nlp_stats)

    image_stats = [
        id, len(predictions['boxes']), len(objects[1]['boxes']),
        np.mean(wordnet_stats['wup_similarities']),
        np.std(wordnet_stats['wup_similarities']),
        np.mean(wordnet_stats['path_similarities']),
        np.std(wordnet_stats['path_similarities']),
        np.mean(nlp_stats['embedding_similarities']),
        np.std(nlp_stats['embedding_similarities']),
        np.mean(wordnet_stats['confidencesPos']), np.std(wordnet_stats['confidencesPos']),
        np.mean(wordnet_stats['confidencesNeg']), np.std(wordnet_stats['confidencesNeg']),
        np.mean(nlp_stats['confidencesPos']), np.std(nlp_stats['confidencesPos']),
        np.mean(nlp_stats['confidencesNeg']), np.std(nlp_stats['confidencesNeg']),
        np.mean(map_stats['confidencesPos']), np.std(map_stats['confidencesPos']),
        np.mean(map_stats['confidencesNeg']), np.std(map_stats['confidencesNeg']),
        iou_threshold, nlp_threshold, wnt_threshold,
        map_stats['positives'], map_stats['negatives'], map_stats['background'],
        nlp_stats['positives'], nlp_stats['negatives'], nlp_stats['background'],
        wordnet_stats['positives'], wordnet_stats['negatives'], wordnet_stats['background']
    ]

    return pd.DataFrame([image_stats], columns=[
        'id', 'predictions', 'objects',
        'wordnet.mean_wpu', 'wordnet.sd_wpu', 'wordnet.mean_path', 'wordnet.sd_path',
        'nlp.mean_embbeding', 'nlp.sd_embedding',
        'wordnet.mean_confidence_pos', 'wordnet.sd_confidence_pos',
        'wordnet.mean_confidence_neg', 'wordnet.sd_confidence_neg',
        'nlp.mean_confidence_pos', 'nlp.sd_confidence_pos',
        'nlp.mean_confidence_neg', 'nlp.sd_confidence_neg',
        'map.mean_confidence_pos', 'map.sd_confidence_pos',
        'map.mean_confidence_neg', 'map.sd_confidence_neg',
        'iou_threshold', 'nlp_threshold', 'wnt_threshold',
        'map_pos', 'map_neg', 'map_back',
        'nlp_pos', 'nlp_neg', 'nlp_back',
        'wordnet_pos', 'wordnet_neg', 'wordnet_back'
    ])


def nlp_similarity_fn(nlp, gt_label, pred_label):
    return nlp(pred_label).similarity(nlp(gt_label))


METAPARAMETER_DEF = {
    'iou_threshold':
        {
            'base': 0,
            'range': 100,
            'default': 50,
            'type': 'integer'
        },
    'nlp_threshold':
        {
            'base': 0,
            'range': 100,
            'default': 50,
            'type': 'integer'
        },
    'wnt_threshold':
        {
            'base': 0,
            'range': 100,
            'default': 50,
            'type': 'integer'
        }
}


def alternative_evaluation(input_path, output_path, n, log_step=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log = section_logger(0)

    dataset = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
                              '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                              transforms=get_transform(train=True))

    dataset_test = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktest.csv',
                                   '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/validation',
                                   transforms=get_transform(train=False),
                                   labels=dataset.labels,
                                   is_test=True)

    nlp = spacy.load("en_core_web_md")

    num_classes = len(dataset.labels)
    model = load_frcnn(input_path, num_classes, device)

    num_examples = 0

    results = []

    metaparameters = generate_metaparameters(10, METAPARAMETER_DEF, static=False)

    for id, objects in enumerate(dataset_test):
        image = objects[0].to(device)
        predictions = model([image])

        if id % log_step == 0:
            log("Processing image [{}] with {} predictions and {} boxes".format(id,
                                                                                len(predictions[0]['boxes']),
                                                                                len(objects[1]['boxes'])))

        for meta_id in range(len(metaparameters['iou_threshold'])):

            iou_threshold = metaparameters['iou_threshold'][meta_id]
            nlp_threshold = metaparameters['iou_threshold'][meta_id]
            wnt_threshold = metaparameters['iou_threshold'][meta_id]

            partial_results = nlp_evaluation(id, objects, predictions[0], dataset.labels, nlp,
                                             iou_threshold=iou_threshold,
                                             nlp_threshold=nlp_threshold,
                                             wnt_threshold=wnt_threshold)

            results.append(partial_results)

        if num_examples < n:
            num_examples += 1
        else:
            break

    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(output_path, 'FRCNN_ad20k_evaluations.csv'))


def regular_evaluation(input_path, output_path, n):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
                              '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                              transforms=get_transform(train=True))

    dataset_test = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktest.csv',
                                   '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/validation',
                                   transforms=get_transform(train=False),
                                   labels=dataset.labels,
                                   is_test=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=True, num_workers=6,
        collate_fn=utils.collate_fn)

    num_classes = len(dataset.labels)
    model = load_frcnn(input_path, num_classes, device)

    evaluate(model, data_loader_test, device=device)


def generate_tuples(labels, number):
    tuples = [(np.random.choice(labels), np.random.choice(labels)) for _ in range(number)]

    tuples_pd = pd.DataFrame(tuples, columns=['l1', 'l2'])
    return tuples_pd.drop_duplicates()


def show_tsne(tsne_results, evaluation_path, title, labels, num_labels=800):
    tsne_coords = {'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1]}

    plt.figure(figsize=(20, 20))

    plot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        palette=sns.color_palette("hls", 10),
        data=tsne_coords,
        legend="full",
        alpha=0.2
    )

    label_sample_ids = np.random.choice(range(len(labels)), num_labels)

    for label_id in label_sample_ids:
        plot.text(tsne_coords['tsne-2d-one'][label_id], tsne_coords['tsne-2d-two'][label_id], labels[label_id],
                  horizontalalignment='left', size=4, color='black', alpha=0.3)

    fig = plot.get_figure()
    fig.savefig(os.path.join(evaluation_path, "TSNE_{}.png".format(title)), dpi=240)


def save_missing(missing):
    missing_map = { missed_word: '' for missed_word in missing }

    with open('missing_words.json', 'w') as f:
        f.write(json.dumps(missing_map, indent=4))


def load_fixes():
    if os.path.exists('word_fixes.json'):
        with open('word_fixes.json') as f:
            fix_string = f.read()
            return json.loads(fix_string)
    else:
        return {}


def create_wordnet_tree(nlp, labels, max_nodes=7000):

    hierarchy = nx.DiGraph()

    pending, missing, synsetmap = generate_synsets_from_labels(labels, nlp)
    processed = 0

    phys = wn.synset('physical_entity.n.01')

    while processed < max_nodes and len(pending) > 0:
        current_synset = pending.pop()

        for hyp_path in filter(lambda p: phys in p, current_synset.hypernym_paths()):
            previous_node = hyp_path[0].name()
            hierarchy.add_node(previous_node)

            for node in hyp_path[1:]:
                current_name = node.name()
                hierarchy.add_node(current_name)
                hierarchy.add_edge(previous_node, current_name)
                previous_node = current_name

        processed += 1

    return hierarchy, synsetmap


def is_physical_object(synset):
    phys = wn.synset('physical_entity.n.01')
    return synset.lowest_common_hypernyms(phys)[0] == phys


def generate_synsets_from_labels(labels, nlp):
    pending = set()
    synsetmap = {}
    missing = []
    fixes = load_fixes()
    for label in labels:
        if label in fixes.keys():
            label = fixes[label]

        lemma = nlp(label).print_tree()[0]['lemma']
        synsets = wn.synsets(lemma)

        if len(synsets) == 0:
            print('Synset not found for label [{}]'.format(label))
            missing.append(label)

        filtered_synsets = filter_physical_synsets(synsets)
        pending.update(filtered_synsets)
        synsetmap[label] = filtered_synsets

    save_missing(missing)

    return pending, missing, synsetmap


def filter_physical_synsets(synsets):
    return [synset for synset in synsets if synset.name().split('.')[1] == 'n' and is_physical_object(synset)]


def draw_subgraphs(output_path, l1, l2, trees, synsetmap):
    filename = 'wnet_subgraph_{}_to_{}.png'.format(l1, l2)
    filepath = os.path.join(output_path, filename)

    options = {
        'node_color': 'green',
        'node_size': 10,
        'width': 1,
        'alpha': 0.7,
        'font-size': 10
    }

    _ = plt.figure(dpi=1200, figsize=(20, 15))

    for id, tree in enumerate(trees):
        plt.subplot(3, 3, id + 1)
        title = " - ".join([x for x in tree.nodes() if tree.out_degree(x) == 0 and tree.in_degree(x) == 1])
        plt.title(title)
        pos = graphviz_layout(tree, prog='dot')
        plt.margins(0.2)
        nx.draw_networkx(tree, pos, with_labels=True, **options)

        if id == 5:
            break

    first_paragraph = "[{}]: {}".format(l1, ", ".join([s.name() for s in synsetmap[l1]]))
    second_paragraph = "[{}]: {}".format(l2, ", ".join([s.name() for s in synsetmap[l2]]))

    ax = plt.subplot(3, 1, 3)

    ax.text(0.01, 0.9, first_paragraph,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, wrap=True,
            fontsize=21)

    ax.text(0.01, 0.4, second_paragraph,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, wrap=True,
            fontsize=21)

    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.99)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches = "tight")
    plt.close()


def expand_nodes(filtered_nodes, wordnet_tree):
    """
    Given a set of nodes and a tree, return a subgraph containing those nodes and all their predecessors
    in the tree

    :param filtered_nodes:          Set of nodes to restrict
    :param wordnet_tree:            DiGraph (without directed cycles)
    :return:                        Subgraph of the nodes and all their ancestors
    """
    pending = set(filtered_nodes)
    final_set = set(filtered_nodes)

    while len(pending) > 0:
        node = pending.pop()
        predecessors = [p for p in wordnet_tree.predecessors(node) if p not in pending and p not in final_set]

        if len(predecessors) > 0:
            pending.update(predecessors)
            final_set.update(predecessors)

    return final_set, wordnet_tree.subgraph(final_set)


def get_children(filtered_tree, wordnet_tree):
    leaves = {node for node in filtered_tree.nodes if len(list(filtered_tree.successors(node))) == 0}
    children = {child for leaf in leaves for child in wordnet_tree.successors(leaf)}
    both = leaves.union(children)

    node_subgraph = wordnet_tree.subgraph(both)
    children_subgraph = wordnet_tree.subgraph(children)

    return node_subgraph, children_subgraph


def draw_full_hierarchy(output_path, wordnet_tree, max_out_degree=10):
    filename = 'global_wordnet_tree.png'
    filepath = os.path.join(output_path, filename)

    filtered_nodes = [n for n in wordnet_tree.nodes() if wordnet_tree.degree(n) > max_out_degree]
    filtered_tree_nodes, filtered_tree = expand_nodes(filtered_nodes, wordnet_tree)
    direct_children, direct_children_labels = get_children(filtered_tree, wordnet_tree)
    full_tree = wordnet_tree.subgraph(filtered_tree_nodes.union(direct_children))

    options = {
        'node_color': 'grey',
        'node_size': 50,
        'width': 0.5,
        'alpha': 0.5,
        'font-size': 2
    }

    _ = plt.figure(dpi=600, figsize=(30, 15))
    plt.margins(0.2)

    pos = graphviz_layout(filtered_tree, prog='dot')
    nx.draw_networkx(filtered_tree, pos, with_labels=False, **options)
    text = nx.draw_networkx_labels(filtered_tree, pos, alpha=0.4, fontsize=2)

    for _, t in text.items():
        t.set_rotation(20)

    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.99)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def save_wordnet_data(root_output_path, wordnet_tree, synsetmap, examples=5, max_combinations=10):
    output_path = os.path.join(root_output_path, 'wordnedSubgraphs')
    os.makedirs(output_path, exist_ok=True)

    labels = list(synsetmap.keys())

    for i in range(examples):
        l1 = labels[random.randint(0, len(labels) - 1)]
        l2 = labels[random.randint(0, len(labels) - 1)]

        trees = []

        raw_combinations = list(itertools.product(synsetmap[l1], synsetmap[l2]))
        combinations = random.sample(raw_combinations, min(len(raw_combinations), max_combinations))

        for pair in combinations:
            s1, s2 = pair
            ancestor = nx.lowest_common_ancestor(wordnet_tree, s1.name(), s2.name())
            shortest_path_s1 = nx.shortest_path(wordnet_tree, ancestor, s1.name())
            shortest_path_s2 = nx.shortest_path(wordnet_tree, ancestor, s2.name())
            subgraph_nodes = set(shortest_path_s2).union(set(shortest_path_s1))
            trees.append(wordnet_tree.subgraph(subgraph_nodes))

        draw_subgraphs(output_path, l1, l2, trees, synsetmap)


def compute_path_to_ancestor(nlp):
    def compute(row):
        lemma1 = nlp(str(row['l1'])).print_tree()[0]['lemma']
        lemma2 = nlp(str(row['l2'])).print_tree()[0]['lemma']
        s1 = filter_physical_synsets(wn.synsets(lemma1))
        s2 = filter_physical_synsets(wn.synsets(lemma2))

        values = []
        mins = []

        for syn1 in s1:
            for syn2 in s2:
                common_ancestor = syn1.lowest_common_hypernyms(syn2)
                path1 = syn1.shortest_path_distance(common_ancestor[0])
                path2 = syn2.shortest_path_distance(common_ancestor[0])
                values.append(path1 + path2)
                mins.append(min(path1, path2))

        return "{}_{}".format(0 if len(values)== 0 else min(values), 0 if len(mins) == 0 else min(mins))

    return compute

def semantic_similarity_evaluation(root_output_path, max_labels_hierarchy=800):
    log = section_logger(0)
    output_path = os.path.join(root_output_path, 'semantic_similarity')
    os.makedirs(output_path, exist_ok=True)

    log('Loading the dataset')
    dataset = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
                              '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                              transforms=get_transform(train=True))

    nlp = spacy.load("en_core_web_md")
    labels = dataset.labels

    log('Creating Wordnet tree')
    wordnet_tree, synsetmap = create_wordnet_tree(nlp, labels)

    log('Drawing Wordnet data')
    save_wordnet_data(output_path, wordnet_tree, synsetmap, examples=200, max_combinations=2000)
    draw_full_hierarchy(output_path, wordnet_tree)

    log('Generating tuples')
    label_tuples = generate_tuples(labels, 10000)

    log('Computing NLP similarities')
    label_tuples['nlp_cosine'] = label_tuples.apply(lambda row: nlp_similarity_fn(nlp, str(row['l1']), str(row['l2'])), axis=1)

    log('Computing Wordnet similarities')
    label_tuples['wordnet_wup'] = label_tuples.apply(lambda row: wordnet_similarity(str(row['l1']), str(row['l2']))['wup'], axis=1)
    label_tuples['wordnet_path'] = label_tuples.apply(lambda row: wordnet_similarity(str(row['l1']), str(row['l2']))['path'], axis=1)
    label_tuples['path_to_ancestor'] = label_tuples.apply(compute_path_to_ancestor(nlp), axis=1)

    label_tuples.to_csv(os.path.join(output_path, 'distances.csv'))

    log('Computing tSNE Clustering')

    embeddings = np.array([nlp(label).vector for label in labels])

    for perplexity in [50, 100, 150, 200]:
        tsne_embedding = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeddings)
        show_tsne(tsne_embedding, output_path, "ADE20k_{}".format(perplexity), labels)


def get_tsne_results(predictions, tsne_embedding, labels, used_labels):
    """
    For a set of detections in an image, generates two datasets of labeled points, where each point
    has the coordinates of its label in the provided embedding.

    :param image:                   predicted and ground labels for a given image
    :param tsne_embedding:          embedding of the labels of the dataset that is being analyzed
    :return:
    """
    prediction_ids = [used_labels.index(i) for i in predictions['predictions']['labels']]
    gt_ids = [used_labels.index(i) for i in predictions['groundTruth']['labels']]

    results = {
        'x': list(tsne_embedding[:, 0][prediction_ids]),
        'y': list(tsne_embedding[:, 1][prediction_ids]),
        'confidence': list(predictions['predictions']['scores']),
        'type': ['prediction' for i in range(len(predictions['predictions']['labels']))],
        'labels': [labels[label_id] for label_id in predictions['predictions']['labels']]
    }

    gt = {
        'x': list(tsne_embedding[:, 0][gt_ids]),
        'y': list(tsne_embedding[:, 1][gt_ids]),
        'confidence': list([1 for i in range(len(predictions['groundTruth']['labels']))]),
        'type': ['truth' for i in range(len(predictions['groundTruth']['labels']))],
        'labels': [labels[label_id] for label_id in predictions['groundTruth']['labels']]
    }

    return {k: results[k] + gt[k] for k in results.keys()}


def generate_semantic_map(index, predictions, output_path, tsne_embedding, labels, used_labels):
    tsne_coords = get_tsne_results(predictions, tsne_embedding, labels, used_labels)

    plt.figure(figsize=(16, 10))

    plot = sns.scatterplot(
        x="x", y="y", hue="type",
        data=tsne_coords,
        legend="full",
        alpha=.4
    )

    for label_id in range(len(tsne_coords['x'])):
        plot.text(tsne_coords['x'][label_id], tsne_coords['y'][label_id], tsne_coords['labels'][label_id],
                  horizontalalignment='left', size=10, color='black', alpha=0.3)

    fig = plot.get_figure()
    fig.savefig(os.path.join(output_path, "{}_TSNE.png".format(index)), dpi=240)


def compute_semantic_similarity(predictions_list, output_path, dataset, perplexity=40):
    """ Computes a map of the semantic similarity of the predictions and the ground truth for each image. """
    nlp = spacy.load("en_core_web_md")
    labels = dataset.labels

    gt_labels = {e for pl in predictions_list for e in pl['groundTruth']['labels']}
    pred_labels = {e for pl in predictions_list for e in pl['predictions']['labels']}
    used_labels = list(gt_labels.union(pred_labels))

    #embeddings = np.array([nlp(label).vector for label in [labels[l] for l in used_labels]])
    embeddings = np.array([nlp(label).vector for label in labels])
    tsne_embedding = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeddings)

    for index, image in enumerate(predictions_list):
        generate_semantic_map(index, image, output_path, tsne_embedding, labels, used_labels)


def compute_lrp_metric(images):
    return None


def compute_spatial_quality(gt, prediction):
    gt_area = np.sum(gt['mask'])

    fg_probability = gt['mask'] * prediction['area']
    fg_probability_sum = np.sum(np.log(fg_probability, where=gt['mask'] > 0), where=gt['mask'] > 0)

    bg_mask = prediction['mask'] * np.abs(gt['mask'] -1)
    bg_probability = bg_mask * (1 - prediction['area'])
    bg_probability_sum = np.sum(np.log(bg_probability, where=bg_mask > 0), where=bg_mask > 0)

    Lfg = - (1/gt_area) * fg_probability_sum
    Lbg = - (1/gt_area) * bg_probability_sum

    return np.exp(-(Lfg + Lbg))


def compute_label_quality(gt, prediction):
    return prediction['score'] if prediction['label'] == gt['label'] else 0


def find_optimal_assignment(pairs):
    matrix = - np.array(pairs)
    results = linear_sum_assignment(matrix)

    return [(results[0][i], results[1][i], pairs[results[0][i]][results[1][i]]) for i in range(results[0].shape[0])]


def compute_confusion_metrics(predictions, optimal_assignment):
    Ntp = len([1 for e in optimal_assignment if e[2] >= 1e-5])
    Nfn = len(predictions['groundTruth']['labels']) - Ntp
    Nfp = len(predictions['predictions']['labels']) - Ntp
    return Ntp, Nfn, Nfp


def compute_uncertainty_metrics(predictions_list, output_path, labels, epsilon=1e-2):
    # TODOlog = section_logger(0): add logs to detect the slow step
    log = section_logger(2)

    accumulator_q = 0
    accumulator_quot = 0

    for pred_id, predictions in enumerate(predictions_list):
        log('Processing uncertainty for image {}'.format(pred_id))

        _, height, width = predictions['image'].shape
        pairs = []
        pairs_spatial = []
        
        ground_truth_list = predictions['groundTruth']
        detections_list = predictions['predictions']

        log('Computing spatial and label Qs')
        for gt_id in range(len(ground_truth_list['labels'])):
            pair_row = []
            pair_spatial_row = []
            gt = extract_boxes(ground_truth_list, gt_id, width, height, epsilon)

            for det_id in range(len(detections_list['labels'])):
                prediction = extract_boxes(detections_list, det_id, width, height, epsilon)

                spatial_q = compute_spatial_quality(gt, prediction)
                label_q = compute_label_quality(gt, prediction)
                pPDQ = np.sqrt(spatial_q * label_q)
                pair_row.append(pPDQ)
                pair_spatial_row.append(spatial_q)

            pairs.append(pair_row)
            pairs_spatial.append(pair_spatial_row)

        log('Finding optimal assignment')
        optimal_assignment = find_optimal_assignment(pairs)
        optimal_spatial_assignment = find_optimal_assignment(pairs_spatial)

        optimally_assigned_predictions = {key: [predictions['predictions'][key][assignment[1]] for assignment in optimal_assignment] for key in ['scores', 'labels', 'boxes']}
        optimally_assigned_spatial_predictions = {key: [predictions['predictions'][key][assignment[1]] for assignment in optimal_spatial_assignment] for key in ['scores', 'labels', 'boxes']}

        log('Computing confusion metrics')
        confusion_metrics = compute_confusion_metrics(predictions, optimal_assignment)

        log('Save image with the assignment')
        postfix = "_".join([str(metric) for metric in confusion_metrics])

        integral_image = torch.tensor(predictions['image'] * 255).numpy().astype(np.int32)
        integral_image = np.transpose(integral_image, (1, 2, 0))

        log('Saving images')
        show_objects_in_image(output_path, integral_image, predictions['groundTruth'],
                              postfix, "{}_uncertainty_both".format(pred_id),
                              labels, predictions=optimally_assigned_predictions,
                              prediction_classes=labels)

        show_objects_in_image(output_path, integral_image, predictions['groundTruth'],
                              postfix, "{}_uncertainty_spatial".format(pred_id),
                              labels, predictions=optimally_assigned_spatial_predictions,
                              prediction_classes=labels)

        # Add to accumulators
        accumulator_quot += sum(confusion_metrics)
        accumulator_q += 1

    # Compute global PDQ
    return accumulator_q/accumulator_quot, optimal_assignment


def extract_boxes(target_list, gt_id, width, height, epsilon):
    box = {
        'box': tuple(int(e) for e in target_list['boxes'][gt_id]),
        'label': target_list['labels'][gt_id]
    }

    box['mask'] = np.transpose(cv2.rectangle(np.zeros((height, width)), box['box'][:2], box['box'][2:], 1, -1))

    if 'scores' in target_list.keys():
        box['score'] = target_list['scores'][gt_id]
        box['area'] = np.abs(np.zeros((width, height)) + box['mask'] - epsilon)
    else:
        box['score'] = [1]

    return box


def compute_map(images):
    return None


def compute_semantic_similarity_metric(ssm):
    return None


def save_full_analysis(images, lrp_metric, uncertainty, map, ssm):
    return None


def complete_evaluation(input_path, output_path, max_results=20):
    log = section_logger(0)

    log('Initializing data')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    log('Load dataset')

    dataset = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktrain.csv',
                          '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/training',
                          transforms=get_transform(train=True))

    labels = dataset.labels

    dataset_test = AD20kFasterRCNN('/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/ade20ktest.csv',
                               '/home/dani/Documentos/Proyectos/Doctorado/Datasets/ADE20K/ADE20K_2016_07_26/images/validation',
                               transforms=get_transform(train=False),
                               labels=labels,
                               is_test=True)

    log('Initializing model')
    num_classes = len(dataset.labels)
    model = load_frcnn(input_path, num_classes, device)
    images = []

    log('Compute model results')
    current_results = 0

    for id, objects in enumerate(dataset_test):
        image = objects[0].to(device)
        predictions = model([image])

        clean_data = lambda data: {k: data[k].tolist() for k in data.keys() if k in ['boxes', 'labels', 'scores']}

        images.append({
            'image': image.cpu().detach().numpy(),
            'groundTruth': clean_data(objects[1]),
            'predictions': clean_data(predictions[0])
        })

        del image
        del predictions

        if current_results > max_results:
            break
        else:
            current_results += 1

    log('Perform full evaluation')
    full_evaluation(images, output_path, dataset, labels)


def save_originals(output_path, images, labels):
    for pred_id, predictions in enumerate(images):
        image = predictions['image']
        integral_image = torch.tensor(image * 255).numpy().astype(np.int32)
        integral_image = np.transpose(integral_image, (1, 2, 0))

        save_image(output_path, integral_image, '', '{}_original'.format(pred_id))

        show_objects_in_image(output_path, integral_image, predictions['groundTruth'],
                              '', "{}_predictions".format(pred_id),
                              labels, predictions=predictions['predictions'],
                              prediction_classes=labels)


def full_evaluation(images, root_output_path, dataset, labels):
    log = section_logger(1)
    evaluation_name = strftime("%Y%m%d%H%M", gmtime())
    output_path = os.path.join(root_output_path, evaluation_name)
    os.makedirs(output_path, exist_ok=True)

    log('Save original images and predictions')
    save_originals(output_path, images, labels)

    log('Compute semantic similarity T-SNE visualizations')
    compute_semantic_similarity(images, output_path, dataset)

    log('Compute LRP metric')
    lrp_metric = compute_lrp_metric(images)

    log('Compute Uncertainty metrics')
    uncertainty, _ = compute_uncertainty_metrics(images, output_path, labels)

    log('Compute mAP')
    map = compute_map(images)

    log('Compute semantic similarity metrics')
    ssm = compute_semantic_similarity_metric(images)

    log('Save results')
    save_full_analysis(images, lrp_metric, uncertainty, map, ssm)


if __name__== "__main__":
    # alternative_evaluation('./FasterRCNN_ADE20k_20191123.pt', '../images/', 40)
    # semantic_similarity_evaluation('../images/')

    complete_evaluation('./FasterRCNN_ADE20k_20191123.pt', '../images/')

