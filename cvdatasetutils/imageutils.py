from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import torch


def get_intersection(obj1, obj2, pos, size):
    order = lambda o1, o2, pos, size: (o1, o2) if o1[pos] - o1[size]/2 < o2[pos] - o2[size]/2 else (o2, o1)

    obj1, obj2 = order(obj1, obj2, pos, size)
    length = min((obj1[pos] + obj1[size]/2), (obj2[pos] + obj1[size]/2)) - (obj2[pos] - obj2[size]/2)

    return max(length, 0)


def IoU(obj1, obj2):
    area = lambda o: o['h']*o['w']

    intersection = get_intersection_area(obj1, obj2)

    union = max(area(obj1) - area(obj2) - intersection, 1)

    return intersection/union


def get_intersection_area(obj1, obj2):
    intersection_w = get_intersection(obj1, obj2, 'bx', 'w')
    intersection_h = get_intersection(obj1, obj2, 'by', 'h')
    intersection = intersection_h * intersection_w
    return intersection


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_image(image_path):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    return image, image_np


def save_image(evaluation_path, image, name, title):
    fig = plt.figure(frameon=False)
    plt.imshow(image, interpolation='nearest')
    fig.savefig(os.path.join(evaluation_path, "ex_{}_{}.png".format(title, name)), dpi=120)
    plt.close(fig)


def show_objects_in_image(evaluation_path, image, ground_truth, name, title, classes, colormap=None,
                          predictions=None, prediction_classes=None, segmentation_alpha=0.3):
    """
    Draw the objects and segmentation masks over the passed image (for separate images if containing the predictions)

    :param evaluation_path:
    :param image:
    :param ground_truth:
    :param name:
    :param title:
    :param classes:
    :param colormap:
    :param predictions:
    :param prediction_classes:
    :param segmentation_alpha:
    :return:
    """

    draw_objects_in_image("ex_{}_{}_gt.png", evaluation_path, image, ground_truth, name, title, classes,
                          colormap, segmentation_alpha)

    if predictions is not None:
        predictions['masks'] = predictions['masks'].squeeze()
        draw_objects_in_image("ex_{}_{}_pred.png", evaluation_path, image, predictions, name, title,
                              prediction_classes, colormap, segmentation_alpha, test=True)


def draw_objects_in_image(img_name, evaluation_path, image, ground_truth, name, title,
                          classes, colormap=None, segmentation_alpha=0.8, test=False):

    fig = plt.figure(frameon=False)

    plt.imshow(image, interpolation='nearest')
    ax = plt.Axes(fig, [0., 0., 1., 0.9])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.title(title)

    colors = [value for value in mcolors.TABLEAU_COLORS.values()]
    draw_objects(ax, classes, ground_truth, colors)

    if 'masks' in ground_truth.keys():
        mask = generate_mask(ground_truth, colors, test)
    else:
        mask = None

    if colormap:
        ax.imshow(image, cmap=colormap)

        if mask is not None:
            ax.imshow(mask, cmap=colormap, alpha=segmentation_alpha)
    else:
        ax.imshow(image)

        if mask is not None:
            ax.imshow(mask, alpha=segmentation_alpha)

    ax.set_axis_off()
    fig.savefig(os.path.join(evaluation_path, img_name.format(title, name)), dpi=120)
    plt.close(fig)


def generate_mask(ground_truth, colors, test):

    if test and ground_truth['labels'].shape == (1,):
        masks = ground_truth['masks']
        img_shape = masks.shape
        mask = np.zeros((*img_shape, 3))
        masks = [(masks > 0.5).int().detach().cpu().numpy()]
    else:
        masks = ground_truth['masks']
        img_shape = masks[0].shape
        mask = np.zeros((*img_shape, 3))
        masks = masks.int()

        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

    for mask_id, obj_mask in enumerate(masks):
        color = mcolors.to_rgb(colors[mask_id % len(colors)])

        for color_index, color_value in enumerate(color):
            try:
                mask[:, :, color_index] += obj_mask * color_value
            except:
                print("Cua")

    return mask


def draw_objects(ax, classes, objects, colors, ground_truth=True, threshold=0.3):

    for obj_id in range(len(objects['boxes'])):
        if ground_truth:
            color_line = 'g'
            position = 'top'
        else:
            color_line = 'r'
            position = 'bottom'

            if objects['scores'][obj_id] < threshold:
                continue

        obj = objects['boxes'][obj_id]
        label = objects['labels'][obj_id]
        w = obj[2] - obj[0]
        h = obj[3] - obj[1]

        box = patches.Rectangle((obj[0], obj[1]), w, h,
                                linewidth=2, edgecolor=color_line, facecolor='none', alpha=0.3)

        ax.add_patch(box)

        ax.text(obj[0], obj[1], classes[label],
                horizontalalignment='left',
                verticalalignment=position,
                bbox=dict(facecolor=colors[obj_id % len(colors)], alpha=0.3))

