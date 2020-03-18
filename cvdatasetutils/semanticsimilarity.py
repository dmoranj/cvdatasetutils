from cvdatasetutils.ad20kfrcnn import AD20kFasterRCNN
import os
import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
import numpy as np
from mltrainingtools.cmdlogging import section_logger
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import json
import random
import itertools
from networkx.drawing.nx_agraph import graphviz_layout
from cvdatasetutils.dnnutils import get_transform


def generate_tuples(labels, number):
    tuples = [(np.random.choice(labels), np.random.choice(labels)) for _ in range(number)]

    tuples_pd = pd.DataFrame(tuples, columns=['l1', 'l2'])
    return tuples_pd.drop_duplicates()


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


def filter_physical_synsets(synsets):
    return [synset for synset in synsets if synset.name().split('.')[1] == 'n' and is_physical_object(synset)]


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


def nlp_similarity_fn(nlp, gt_label, pred_label):
    return nlp(pred_label).similarity(nlp(gt_label))


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

    #analyze_wordnet_distance(labels, log, nlp, output_path)
    analyze_semantic_distance(labels, log, nlp, output_path)


def analyze_wordnet_distance(labels, log, nlp, output_path):
    log('Creating Wordnet tree')
    wordnet_tree, synsetmap = create_wordnet_tree(nlp, labels)

    log('Drawing Wordnet data')
    save_wordnet_data(output_path, wordnet_tree, synsetmap, examples=200, max_combinations=2000)
    draw_full_hierarchy(output_path, wordnet_tree)


def analyze_semantic_distance(labels, log, nlp, output_path):
    log('Generating tuples')
    label_tuples = generate_tuples(labels, 10000)
    log('Computing NLP similarities')
    label_tuples['nlp_cosine'] = label_tuples.apply(lambda row: nlp_similarity_fn(nlp, str(row['l1']), str(row['l2'])),
                                                    axis=1)
    log('Computing Wordnet similarities')
    label_tuples['wordnet_wup'] = label_tuples.apply(
        lambda row: wordnet_similarity(str(row['l1']), str(row['l2']))['wup'], axis=1)
    label_tuples['wordnet_path'] = label_tuples.apply(
        lambda row: wordnet_similarity(str(row['l1']), str(row['l2']))['path'], axis=1)
    label_tuples['path_to_ancestor'] = label_tuples.apply(compute_path_to_ancestor(nlp), axis=1)
    label_tuples.to_csv(os.path.join(output_path, 'distances.csv'))
    log('Computing tSNE Clustering')
    embeddings = np.array([nlp(label).vector for label in labels])
    for perplexity in [50, 100, 150, 200]:
        tsne_embedding = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeddings)
        show_tsne(tsne_embedding, output_path, "ADE20k_{}".format(perplexity), labels)


if __name__== "__main__":
    semantic_similarity_evaluation('../images/')

