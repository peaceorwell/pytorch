import torch

from torch._dynamo import register_backend
from torch.fx import subgraph_rewriter
from torch._functorch.aot_autograd import aot_module_simplified

def search_pattern(x, y):
    #return (x + y) * y
    x = torch.ops.aten.add.Tensor(x, y)
    x = torch.ops.aten.mul.Tensor(x, y)
    return x


def replacement(x, y):
    return torch.ops.aten.sub.Tensor(x, y)


def compile_fx(model_, example_inputs, *args, **kwargs):
    def fw_compiler(gm, sample_inputs):
        subgraph_rewriter.replace_pattern_with_filters(gm, search_pattern, replacement)
        return gm
    return aot_module_simplified(model_, example_inputs, fw_compiler=fw_compiler)


@register_backend
def bytedance(*args, **kwargs):
    return compile_fx(*args, **kwargs)
