from coco_utils import apply_eval


def ssd_eval(dataset, net, anno_json):
    """SSD evaluation."""
    batch_size = 1

    print("Load Checkpoint!")
    net.set_train(False)
    total = dataset.get_dataset_size() * batch_size
    print("\n========================================\n")
    print("total images num: ", total)
    eval_param_dict = {"net": net, "dataset": dataset, "anno_json": anno_json}
    mAP = apply_eval(eval_param_dict)
    print("\n========================================\n")
    print(f"mAP: {mAP}")
