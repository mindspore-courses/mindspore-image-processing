'''浮点数分析'''
# pylint:disable=E0401, W0611
import mindspore
import mindspore.nn as nn
from vit_model import Attention


def count_flops(layer, input_size):
    '''计算给定层的FLOPs'''
    if isinstance(layer, nn.Conv2d):
        output_size = layer(input_size).size()
        kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
        flops = output_size[0] * output_size[1] * output_size[2] * \
            output_size[3] / (4 * kernel_size[0] * kernel_size[1])
    elif isinstance(layer, nn.Dense):
        output_size = layer(input_size).size()
        flops = output_size[0] * output_size[1]
    else:
        flops = 0
    return flops


def flop_count_analysis(model, input_size):
    '''结果累加'''
    total_flops = 0
    for layer in model.cells():
        if isinstance(layer, (nn.Conv2d, nn.Dense)):
            total_flops += count_flops(layer, input_size)
    return total_flops


def main():
    '''主函数'''
    # Self-Attention
    a1 = Attention(dim=512, num_heads=1)
    a1.proj = mindspore.nn.Identity()  # remove Wo

    # Multi-Head Attention
    a2 = Attention(dim=512, num_heads=8)

    # [batch_size, num_tokens, total_embed_dim]
    t = (mindspore.ops.rand(32, 1024, 512),)

    flops1 = flop_count_analysis(a1, t)
    print("Self-Attention FLOPs:", flops1.total())

    flops2 = flop_count_analysis(a2, t)
    print("Multi-Head Attention FLOPs:", flops2.total())


if __name__ == '__main__':
    main()
