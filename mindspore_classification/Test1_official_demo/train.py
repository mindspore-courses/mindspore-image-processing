'''模型训练'''
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from model import LeNet


def main():
    '''主函数'''
    transform = ds.transforms.Compose(
        [ds.vision.ToTensor(),
         ds.vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar10_dataset_dir = "/path/to/cifar10_dataset_directory"
    # 50000张训练图片
    train_set = ds.Cifar10Dataset(
        dataset_dir=cifar10_dataset_dir, shuffle=True)
    train_loader = ds.GeneratorDataset(
        train_set, shuffle=True, num_parallel_workers=0)
    train_loader = train_loader.map(operations=transform)
    train_loader = train_loader.batch(36)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, shuffle=False)
    val_loader = ds.GeneratorDataset(
        val_set, shuffle=False, num_wonum_parallel_workersrkers=0)
    val_loader = val_loader.map(operations=transform)
    val_loader = val_loader.batch(5000)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = nn.Adam(net.get_parameters(), learning_rate=0.001)

    # 前向传播
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_function(logits, label)
        return loss, logits

    # 梯度函数
    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # 更新，训练
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        return loss, logits

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            loss, outputs = train_step(inputs, labels)

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                outputs = net(val_image)  # [batch, 10]
                predict_y = mindspore.ops.max(outputs, axis=1)[1]
                accuracy = mindspore.ops.equal(
                    predict_y, val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.ckpt'
    mindspore.save_checkpoint(net, save_path)


if __name__ == '__main__':
    main()
