'''模型训练工具类'''
import sys

from tqdm import tqdm
import mindspore


def train_one_epoch(model, optimizer, data_loader, epoch):
    '''训练'''
    model.set_train(True)
    criterion = mindspore.nn.CrossEntropyLoss()
    mean_loss = mindspore.ops.zeros(1)  # 平均损失

    # 前向传播
    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits

    # 梯度函数
    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # 更新，训练
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        return loss, logits

    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        loss, _ = train_step(images, labels)
        mean_loss = (mean_loss * step + loss.detach()) / \
            (step + 1)  # update mean losses

        # 打印平均loss
        data_loader.desc = "[epoch {}] mean loss {}".format(
            epoch, round(mean_loss.item(), 3))

    return mean_loss.item()


def evaluate(model, data_loader):
    '''验证'''
    model.set_train(False)

    # 用于存储预测正确的样本个数
    sum_num = mindspore.ops.zeros(1)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...", file=sys.stdout)

    for _, data in enumerate(data_loader):
        images, labels = data
        pred = model(images)
        pred = mindspore.ops.max(pred, axis=1)[1]
        sum_num += mindspore.ops.equal(pred, labels).sum()

    # 计算预测正确的比例
    acc = sum_num.item() / num_samples

    return acc
