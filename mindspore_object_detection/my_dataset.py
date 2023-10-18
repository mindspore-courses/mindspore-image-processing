import numpy as np
import cv2
from src import GeneratDefaultBoxes
from mindspore.dataset import MindDataset, VOCDataset
from mindspore.dataset.vision import Decode, HWC2CHW, Normalize, RandomColorAdjust

default_boxes_tlbr = GeneratDefaultBoxes().default_boxes_tlbr
default_boxes = GeneratDefaultBoxes().default_boxes

y1, x1, y2, x2 = np.split(default_boxes_tlbr[:, :4], 4, axis=-1)
vol_anchors = (x2 - x1) * (y2 - y1)
matching_threshold = 0.5


def create_dataset(mindrecord_file, batch_size=32, device_num=1, rank=0,
                   is_training=True, num_parallel_workers=1, use_multiprocessing=True):
    """Create SSD dataset with VOCDataset."""
    dataset = MindDataset(mindrecord_file, columns_list=["img_id", "image", "annotation"], num_shards=device_num,
                          shard_id=rank, num_parallel_workers=num_parallel_workers, shuffle=is_training)

    decode = Decode()
    dataset = dataset.map(operations=decode, input_columns=["image"])

    change_swap_op = HWC2CHW()
    # Computed from random subset of ImageNet training images
    normalize_op = Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    color_adjust_op = RandomColorAdjust(
        brightness=0.4, contrast=0.4, saturation=0.4)
    compose_map_func = (lambda img_id, image, annotation: preprocess_fn(
        img_id, image, annotation, is_training))

    if is_training:
        output_columns = ["image", "box", "label", "num_match"]
        trans = [color_adjust_op, normalize_op, change_swap_op]
    else:
        output_columns = ["img_id", "image", "image_shape"]
        trans = [normalize_op, change_swap_op]

    dataset = dataset.map(operations=compose_map_func, input_columns=["img_id", "image", "annotation"],
                          output_columns=output_columns, python_multiprocessing=use_multiprocessing,
                          num_parallel_workers=num_parallel_workers)

    dataset = dataset.map(operations=trans, input_columns=["image"], python_multiprocessing=use_multiprocessing,
                          num_parallel_workers=num_parallel_workers)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


def intersect(box_a, box_b):
    """Compute the intersect of two sets of boxes."""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union


def random_sample_crop(image, boxes):
    """Crop images and boxes randomly."""
    height, width, _ = image.shape
    min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])

    if min_iou is None:
        return image, boxes

    for _ in range(50):
        image_t = image
        w = _rand(0.3, 1.0) * width
        h = _rand(0.3, 1.0) * height
        # aspect ratio constraint b/t .5 & 2
        if h / w < 0.5 or h / w > 2:
            continue

        left = _rand() * (width - w)
        top = _rand() * (height - h)
        rect = np.array([int(top), int(left), int(top + h), int(left + w)])
        overlap = jaccard_numpy(boxes, rect)

        # dropout some boxes
        drop_mask = overlap > 0
        if not drop_mask.any():
            continue

        if overlap[drop_mask].min() < min_iou and overlap[drop_mask].max() > (min_iou + 0.2):
            continue

        image_t = image_t[rect[0]:rect[2], rect[1]:rect[3], :]
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

        # mask in that both m1 and m2 are true
        mask = m1 * m2 * drop_mask

        # have any valid boxes? try again if not
        if not mask.any():
            continue

        # take only matching gt boxes
        boxes_t = boxes[mask, :].copy()
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2])
        boxes_t[:, :2] -= rect[:2]
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4])
        boxes_t[:, 2:4] -= rect[:2]

        return image_t, boxes_t
    return image, boxes


def ssd_bboxes_encode(boxes):
    """Labels anchors with ground truth inputs."""

    def jaccard_with_anchors(bbox):
        """Compute jaccard score a box and the anchors."""
        # Intersection bbox and volume.
        ymin = np.maximum(y1, bbox[0])
        xmin = np.maximum(x1, bbox[1])
        ymax = np.minimum(y2, bbox[2])
        xmax = np.minimum(x2, bbox[3])
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors + \
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        jaccard = inter_vol / union_vol
        return np.squeeze(jaccard)

    pre_scores = np.zeros((8732), dtype=np.float32)
    t_boxes = np.zeros((8732, 4), dtype=np.float32)
    t_label = np.zeros((8732), dtype=np.int64)
    for bbox in boxes:
        label = int(bbox[4])
        scores = jaccard_with_anchors(bbox)
        idx = np.argmax(scores)
        scores[idx] = 2.0
        mask = (scores > matching_threshold)
        mask = mask & (scores > pre_scores)
        pre_scores = np.maximum(pre_scores, scores * mask)
        t_label = mask * label + (1 - mask) * t_label
        for i in range(4):
            t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i]

    index = np.nonzero(t_label)

    # Transform to tlbr.
    bboxes = np.zeros((8732, 4), dtype=np.float32)
    bboxes[:, [0, 1]] = (t_boxes[:, [0, 1]] + t_boxes[:, [2, 3]]) / 2
    bboxes[:, [2, 3]] = t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]

    # Encode features.
    bboxes_t = bboxes[index]
    default_boxes_t = default_boxes[index]
    bboxes_t[:, :2] = (bboxes_t[:, :2] - default_boxes_t[:, :2]
                       ) / (default_boxes_t[:, 2:] * 0.1)
    tmp = np.maximum(bboxes_t[:, 2:4] / default_boxes_t[:, 2:4], 0.000001)
    bboxes_t[:, 2:4] = np.log(tmp) / 0.2
    bboxes[index] = bboxes_t

    num_match = np.array([len(np.nonzero(t_label)[0])], dtype=np.int32)
    return bboxes, t_label.astype(np.int32), num_match


def preprocess_fn(img_id, image, box, is_training):
    """Preprocess function for dataset."""
    cv2.setNumThreads(2)

    def _infer_data(image, input_shape):
        img_h, img_w, _ = image.shape
        input_h, input_w = input_shape

        image = cv2.resize(image, (input_w, input_h))

        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        return img_id, image, np.array((img_h, img_w), np.float32)

    def _data_aug(image, box, is_training, image_size=(300, 300)):
        ih, iw, _ = image.shape
        h, w = image_size
        if not is_training:
            return _infer_data(image, image_size)
        # Random crop
        box = box.astype(np.float32)
        image, box = random_sample_crop(image, box)
        ih, iw, _ = image.shape
        # Resize image
        image = cv2.resize(image, (w, h))
        # Flip image or not
        flip = _rand() < .5
        if flip:
            image = cv2.flip(image, 1, dst=None)
        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)
        box[:, [0, 2]] = box[:, [0, 2]] / ih
        box[:, [1, 3]] = box[:, [1, 3]] / iw
        if flip:
            box[:, [1, 3]] = 1 - box[:, [3, 1]]
        box, label, num_match = ssd_bboxes_encode(box)
        return image, box, label, num_match

    return _data_aug(image, box, is_training, image_size=[300, 300])
