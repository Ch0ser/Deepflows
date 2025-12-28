from .module import Module
from collections import OrderedDict
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from typing_extensions import Self


__all__ = ['Sequential', 'ModuleList', 'ModuleDict']


class Sequential(Module):
    r"""A sequential container
    
    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.
    (上面注释搬运自'https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/container.py')

    Example::
    1.
        models = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
    2.
        models = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    二者功效相同
    """

    _modules: Dict[str, Module]
    
    """
    @overload -> 重载
    参考
    https://zhuanlan.zhihu.com/p/489767633
    https://zhuanlan.zhihu.com/p/677070644
    """
    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...
    
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        # 对应上面Example的2
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        # 对应上面Example的1
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __add__(self, other):
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError('add operator supports only objects '
                             f'of Sequential class')

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> 'Sequential':
        r"""Append a given module to the end.
        类似于list的append方法
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, sequential) -> 'Sequential':
        r"""Append a given sequential to the end
        类似于list的extend方法
        """
        for layer in sequential:
            self.append(layer)
        return self
        

class ModuleList(Module):
    pass


class ModuleDict(Module):
    pass
