'''模型预测'''
# pylint:disable=E0401
import mindspore
import mindspore.dataset as ds
from PIL import Image

from model import LeNet


def main():
    '''主函数'''
    transform = ds.transforms.Compose(
        [ds.vision.Resize((32, 32)),
         ds.vision.ToTensor(),
         ds.vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    mindspore.load_param_into_net(net, mindspore.load_checkpoint('Lenet.ckpt'))

    im = Image.open('1.jpg')
    im = transform(im)  # [C, H, W]
    im = mindspore.ops.unsqueeze(im, dim=0)  # [N, C, H, W]

    outputs = net(im)
    predict = mindspore.ops.max(outputs, axis=1)[1].asnumpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
