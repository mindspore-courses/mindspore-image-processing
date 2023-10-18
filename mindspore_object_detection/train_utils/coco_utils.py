import json
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mindspore as ms
from mindspore import Tensor, nn, ops


train_cls = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
             'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
             'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
             'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
             'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
             'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
             'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
             'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
             'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']

train_cls_dict = {}
for i, cls in enumerate(train_cls):
    train_cls_dict[cls] = i


def apply_eval(eval_param_dict):
    net = eval_param_dict["net"]
    net.set_train(False)
    ds = eval_param_dict["dataset"]
    anno_json = eval_param_dict["anno_json"]
    coco_metrics = COCOMetrics(anno_json=anno_json,
                               classes=train_cls,
                               num_classes=81,
                               max_boxes=100,
                               nms_threshold=0.6,
                               min_score=0.1)
    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_id = data['img_id']
        img_np = data['image']
        image_shape = data['image_shape']

        output = net(Tensor(img_np))

        for batch_idx in range(img_np.shape[0]):
            pred_batch = {
                "boxes": output[0].asnumpy()[batch_idx],
                "box_scores": output[1].asnumpy()[batch_idx],
                "img_id": int(np.squeeze(img_id[batch_idx])),
                "image_shape": image_shape[batch_idx]
            }
            coco_metrics.update(pred_batch)
    eval_metrics = coco_metrics.get_metrics()
    return eval_metrics


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep


class COCOMetrics:
    """Calculate mAP of predicted bboxes."""

    def __init__(self, anno_json, classes, num_classes, min_score, nms_threshold, max_boxes):
        self.num_classes = num_classes
        self.classes = classes
        self.min_score = min_score
        self.nms_threshold = nms_threshold
        self.max_boxes = max_boxes

        self.val_cls_dict = {i: cls for i, cls in enumerate(classes)}
        self.coco_gt = COCO(anno_json)
        cat_ids = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        self.class_dict = {cat['name']: cat['id'] for cat in cat_ids}

        self.predictions = []
        self.img_ids = []

    def update(self, batch):
        pred_boxes = batch['boxes']
        box_scores = batch['box_scores']
        img_id = batch['img_id']
        h, w = batch['image_shape']

        final_boxes = []
        final_label = []
        final_score = []
        self.img_ids.append(img_id)

        for c in range(1, self.num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > self.min_score
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(
                    class_boxes, class_box_scores, self.nms_threshold, self.max_boxes)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [self.class_dict[self.val_cls_dict[c]]
                                ] * len(class_box_scores)

        for loc, label, score in zip(final_boxes, final_label, final_score):
            res = {}
            res['image_id'] = img_id
            res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
            res['score'] = score
            res['category_id'] = label
            self.predictions.append(res)

    def get_metrics(self):
        with open('predictions.json', 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f)

        coco_dt = self.coco_gt.loadRes('predictions.json')
        E = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        E.params.imgIds = self.img_ids
        E.evaluate()
        E.accumulate()
        E.summarize()
        return E.stats[0]


class SsdInferWithDecoder(nn.Cell):
    """
SSD Infer wrapper to decode the bbox locations."""

    def __init__(self, network, default_boxes, ckpt_path):
        super(SsdInferWithDecoder, self).__init__()
        param_dict = ms.load_checkpoint(ckpt_path)
        ms.load_param_into_net(network, param_dict)
        self.network = network
        self.default_boxes = default_boxes
        self.prior_scaling_xy = 0.1
        self.prior_scaling_wh = 0.2

    def construct(self, x):
        pred_loc, pred_label = self.network(x)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * \
            default_bbox_wh + default_bbox_xy
        pred_wh = ops.exp(pred_loc[..., 2:] *
                          self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = ops.concat((pred_xy_0, pred_xy_1), -1)
        pred_xy = ops.maximum(pred_xy, 0)
        pred_xy = ops.minimum(pred_xy, 1)
        return pred_xy, pred_label
