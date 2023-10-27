'''初始化'''
# pylint: disable=E0401
from .train_eval_utils import train_one_epoch, get_lr, evaluate
from .loss import KpLoss
from .coco_eval import EvalCOCOMetric
