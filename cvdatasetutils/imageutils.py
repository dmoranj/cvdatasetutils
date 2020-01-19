from PIL import Image
import PIL
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms


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
                          predictions=None, prediction_classes=None):

    fig = plt.figure(frameon=False)

    #if not isinstance(image, PIL.Image.Image):
    #    image = transforms.ToPILImage()(image)

    plt.imshow(image, interpolation='nearest')
    ax = plt.Axes(fig, [0., 0., 1., 0.9])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.title(title)

    draw_objects(ax, classes, ground_truth)

    if predictions is not None:
        draw_objects(ax, prediction_classes, predictions, ground_truth=False)

    if colormap:
        ax.imshow(image, cmap=colormap)
    else:
        ax.imshow(image)

    ax.set_axis_off()
    fig.savefig(os.path.join(evaluation_path, "ex_{}_{}.png".format(title, name)), dpi=120)
    plt.close(fig)


def draw_objects(ax, classes, objects, ground_truth=True, threshold=0.3):
    for obj_id in range(len(objects['boxes'])):
        if ground_truth:
            color = 'g'
            position = 'top'
        else:
            color = 'r'
            position = 'bottom'

            if objects['scores'][obj_id] < threshold:
                continue

        obj = objects['boxes'][obj_id]
        label = objects['labels'][obj_id]
        w = obj[2] - obj[0]
        h = obj[3] - obj[1]

        box = patches.Rectangle((obj[0], obj[1]), w, h,
                                linewidth=2, edgecolor=color, facecolor='none', alpha=0.3)

        ax.add_patch(box)

        ax.text(obj[0], obj[1], classes[label],
                horizontalalignment='left',
                verticalalignment=position,
                bbox=dict(facecolor=color, alpha=0.3))
