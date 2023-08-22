'''评价指标'''
# pylint:disable=E0401, W0613, R1710
import contextlib
import io
import mindspore
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
import numpy as np


class MeanAveragePrecision(mindspore.train.Metric):
    '''实现基于mindspore的map评价指标'''

    def __init__(self, box_format="xyxy",
                 iou_type="bbox",
                 iou_thresholds=None,
                 rec_thresholds=None,
                 max_detection_thresholds=None,
                 class_metrics: bool = False,
                 **kwargs):
        super().__init__()

        self.box_format = box_format
        self.iou_type = iou_type
        self.iou_thresholds = iou_thresholds or mindspore.numpy.linspace(
            0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()
        self.rec_thresholds = rec_thresholds or mindspore.numpy.linspace(
            0.0, 1.00, round(1.00 / 0.01) + 1).tolist()
        max_det_thr, _ = mindspore.ops.sort(mindspore.Tensor(
            max_detection_thresholds or [1, 10, 100], dtype=mindspore.int32))
        self.max_detection_thresholds = max_det_thr.asnumpy().tolist()
        self.class_metrics = class_metrics

        self.detections = []
        self.detection_scores = []
        self.detection_labels = []
        self.groundtruths = []
        self.groundtruth_labels = []
        self.groundtruth_crowds = []
        self.groundtruth_area = []

        self.coco_eval = 0

    def clear(self):
        """初始化变量列表"""
        self.detections = []
        self.detection_scores = []
        self.detection_labels = []
        self.groundtruths = []
        self.groundtruth_labels = []
        self.groundtruth_crowds = []
        self.groundtruth_area = []

    def update(self, preds, target):
        """更新"""
        for item in preds:
            detections = self._get_safe_item_values(item)

            self.detections.append(detections)
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            groundtruths = self._get_safe_item_values(item)
            self.groundtruths.append(groundtruths)
            self.groundtruth_labels.append(item["labels"])
            self.groundtruth_crowds.append(
                item.get("iscrowd", mindspore.ops.zeros_like(item["labels"])))
            self.groundtruth_area.append(
                item.get("area", mindspore.ops.zeros_like(item["labels"])))

    def eval(self):
        """计算最终评估结果"""
        coco_target, coco_preds = COCO(), COCO()

        coco_target.dataset = self._get_coco_format(
            self.groundtruths, self.groundtruth_labels, crowds=self.groundtruth_crowds, area=self.groundtruth_area
        )
        coco_preds.dataset = self._get_coco_format(
            self.detections, self.detection_labels, scores=self.detection_scores)

        with contextlib.redirect_stdout(io.StringIO()):
            coco_target.createIndex()
            coco_preds.createIndex()

            self.coco_eval = COCOeval(
                coco_target, coco_preds, iouType=self.iou_type)
            self.coco_eval.params.iouThrs = np.array(
                self.iou_thresholds, dtype=np.float64)
            self.coco_eval.params.recThrs = np.array(
                self.rec_thresholds, dtype=np.float64)
            self.coco_eval.params.maxDets = self.max_detection_thresholds

            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            self.coco_eval.summarize()
            stats = self.coco_eval.stats

        # if class mode is enabled, evaluate metrics per class
        if self.class_metrics:
            map_per_class_list = []
            mar_100_per_class_list = []
            for class_id in self._get_classes():
                self.coco_eval.params.catIds = [class_id]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.coco_eval.evaluate()
                    self.coco_eval.accumulate()
                    self.coco_eval.summarize()
                    class_stats = self.coco_eval.stats

                map_per_class_list.append(mindspore.Tensor([class_stats[0]]))
                mar_100_per_class_list.append(
                    mindspore.Tensor([class_stats[8]]))

            map_per_class_values = mindspore.Tensor(
                map_per_class_list, dtype=mindspore.float32)
            mar_100_per_class_values = mindspore.Tensor(
                mar_100_per_class_list, dtype=mindspore.float32)
        else:
            map_per_class_values = mindspore.Tensor(
                [-1], dtype=mindspore.float32)
            mar_100_per_class_values = mindspore.Tensor(
                [-1], dtype=mindspore.float32)

        return {
            "map": mindspore.Tensor([stats[0]], dtype=mindspore.float32),
            "map_50": mindspore.Tensor([stats[1]], dtype=mindspore.float32),
            "map_75": mindspore.Tensor([stats[2]], dtype=mindspore.float32),
            "map_small": mindspore.Tensor([stats[3]], dtype=mindspore.float32),
            "map_medium": mindspore.Tensor([stats[4]], dtype=mindspore.float32),
            "map_large": mindspore.Tensor([stats[5]], dtype=mindspore.float32),
            "mar_1": mindspore.Tensor([stats[6]], dtype=mindspore.float32),
            "mar_10": mindspore.Tensor([stats[7]], dtype=mindspore.float32),
            "mar_100": mindspore.Tensor([stats[8]], dtype=mindspore.float32),
            "mar_small": mindspore.Tensor([stats[9]], dtype=mindspore.float32),
            "mar_medium": mindspore.Tensor([stats[10]], dtype=mindspore.float32),
            "mar_large": mindspore.Tensor([stats[11]], dtype=mindspore.float32),
            "map_per_class": map_per_class_values,
            "mar_100_per_class": mar_100_per_class_values,
            "classes": mindspore.Tensor(self._get_classes(), dtype=mindspore.int32),
        }

    def _get_classes(self):
        """Return a list of unique classes found in ground truth and detection data."""
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            return mindspore.ops.cat(self.detection_labels + self.groundtruth_labels).unique().cpu().asnumpy().tolist()
        return []

    def _get_coco_format(
        self,
        boxes,
        labels,
        scores=None,
        crowds=None,
        area=None,
    ):
        """Transforms and returns all cached targets or predictions in COCO format.

        Format is defined at
        https://cocodataset.org/#format-data

        """
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for image_id, (image_boxes, image_labels) in enumerate(zip(boxes, labels)):
            if self.iou_type == "segm" and len(image_boxes) == 0:
                continue

            if self.iou_type == "bbox":
                image_boxes = image_boxes.cpu().tolist()
            image_labels = image_labels.cpu().tolist()

            images.append({"id": image_id})
            if self.iou_type == "segm":
                images[-1]["height"], images[-1]["width"] = image_boxes[0][0][0], image_boxes[0][0][1]

            for k, (image_box, image_label) in enumerate(zip(image_boxes, image_labels)):
                if self.iou_type == "bbox" and len(image_box) != 4:
                    raise ValueError(
                        f"Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})"
                    )

                if isinstance(image_label, int):
                    raise ValueError(
                        f"Invalid input class of sample {image_id}, element {k}"
                        f" (expected value of type integer, got type {type(image_label)})"
                    )

                stat = image_box if self.iou_type == "bbox" else {
                    "size": image_box[0], "counts": image_box[1]}

                if area is not None and area[image_id][k].cpu().tolist() > 0:
                    area_stat = area[image_id][k].cpu().tolist()
                else:
                    area_stat = image_box[2] * \
                        image_box[3] if self.iou_type == "bbox" else mask_utils.area(
                            stat)

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox" if self.iou_type == "bbox" else "segmentation": stat,
                    "area": area_stat,
                    "category_id": image_label,
                    "iscrowd": crowds[image_id][k].cpu().tolist() if crowds is not None else 0,
                }

                if scores is not None:
                    score = scores[image_id][k].cpu().tolist()
                    if isinstance(score, float):
                        raise ValueError(
                            f"Invalid input score of sample {image_id}, element {k}"
                            f" (expected value of type float, got type {type(score)})"
                        )
                    annotation["score"] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [{"id": i, "name": str(i)} for i in self._get_classes()]
        return {"images": images, "annotations": annotations, "categories": classes}

    def _get_safe_item_values(self, item):
        """Convert and return the boxes or masks from the item depending on the iou_type.

        Args:
            item: input dictionary containing the boxes or masks

        Returns:
            boxes or masks depending on the iou_type

        """
        masks = []
        if self.iou_type == "bbox":
            masks = _fix_empty_tensors(item["boxes"])
            if masks.numel() > 0:
                masks = box_convert(
                    masks, in_fmt=self.box_format, out_fmt="xywh")
            return masks
        if self.iou_type == "segm":
            for i in item["masks"].cpu().numpy():
                rle = mask_utils.encode(np.asfortranarray(i))
                masks.append((tuple(rle["size"]), rle["counts"]))
            return tuple(masks)


def box_convert(boxes, in_fmt="xyxy", out_fmt="xywh"):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) which will be converted.

    Returns:
        boxes (Tensor[N, 4]): boxes in (x, y, w, h) format.
    """
    if in_fmt == "xyxy" and out_fmt == "xywh":
        x1, y1, x2, y2 = boxes.unbind(-1)
        w = x2 - x1  # x2 - x1
        h = y2 - y1  # y2 - y1
        boxes = mindspore.ops.stack((x1, y1, w, h), axis=-1)
        return boxes


def _fix_empty_tensors(boxes):
    """Empty tensors can cause problems in DDP mode, this methods corrects them."""
    if boxes.numel() == 0 and boxes.ndim == 1:
        return boxes.unsqueeze(0)
    return boxes
