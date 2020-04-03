from cvdatasetutils.imageutils import IoU, adjust_box


def evaluate_map(predictions, ground_truth, width, height):
    recall_steps = [round(i * 0.1, 2) for i in range(0, 11)]
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(0, 10)]

    positive_labels = set(ground_truth['labels'].tolist())

    label_scores = {label: {iou: {recall: [] for recall in recall_steps} for iou in iou_thresholds} for label in positive_labels}
    totals = 0

    for positive_label in positive_labels:
        for iou_threshold in iou_thresholds:
            for recall_step in recall_steps:
                positive_indexes = [i for i in range(len(ground_truth['labels'])) if ground_truth['labels'][i] == positive_label]
                prediction_indexes = [i for i in range(len(predictions['labels'])) if predictions['labels'][i] == positive_label]

                true_positives = 0
                false_positives = 0

                for positive_index in positive_indexes:
                    for prediction_index in prediction_indexes:
                        adjusted_prediction = adjust_box(predictions['boxes'][prediction_index], width, height)
                        adjusted_gt = adjust_box(ground_truth['boxes'][positive_index], width, height)
                        iou = IoU(adjusted_prediction, adjusted_gt)

                        if iou > iou_threshold:
                            if predictions['scores'][prediction_index] > recall_step:
                                true_positives += 1
                                break
                            else:
                                false_positives += 1

                precision = true_positives / len(positive_indexes)
                label_scores[positive_label][iou_threshold][positive_index] = precision
                totals += precision

    average = totals/(len(positive_labels) + len(iou_thresholds) + len(recall_steps))

    return average, label_scores


def evaluate_masks(predictions, ground_truth, width, height):
    return evaluate_map(predictions, ground_truth, width, height)
