'''主程序'''
# pylint:disable=E0401
import typing
from collections import defaultdict

import mindspore
import mindspore.nn as nn

import tabulate
from prettytable import PrettyTable
from model import efficientnetv2_s


def parameter_count(model: nn.Cell):
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    r = defaultdict(int)
    for name, prm in model.get_parameters():
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r


def parameter_count_table(model: nn.Cell, max_depth: int = 3) -> str:
    """
    Args:
        model: a torch module
        max_depth (int): maximum depth to recursively print submodules or
            parameters

    Returns:
        str: the table to be printed
    """
    count: typing.DefaultDict[str, int] = parameter_count(model)
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.get_parameters()
    }

    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        if x > 1e8:
            return f"{(x / 1e9):.1f}G"
        if x > 1e5:
            return f"{(x / 1e6):.1f}M"
        if x > 1e2:
            return f"{(x / 1e3):.1f}K"
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append(
                        (indent + name, indent + str(param_shape[name])))
                else:
                    table.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop(""))))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(
        table, headers=["name", "#elements or shape"], tablefmt="pipe"
    )
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab


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
    model = efficientnetv2_s()

    # option1
    for name, para in model.trainable_params():
        # 除head外，其他权重全部冻结
        if "head" not in name:
            para.requires_grad = False
        else:
            print(f"training {name}")

    complexity = model.complexity(224, 224, 3)
    table = PrettyTable()
    table.field_names = ["params", "freeze-params",
                         "train-params", "FLOPs", "acts"]
    table.add_row([complexity["params"],
                   complexity["freeze"],
                   complexity["params"] - complexity["freeze"],
                   complexity["flops"],
                   complexity["acts"]])
    print(table)

    # option2
    tensor = (mindspore.ops.rand(1, 3, 224, 224),)
    flops = flop_count_analysis(model, tensor)
    print(flops.total())

    print(parameter_count_table(model))


if __name__ == '__main__':
    main()
