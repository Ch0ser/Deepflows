import itertools
from collections import OrderedDict
from copy import deepcopy
from ..parameter import Parameter
from ...autograd import set_grad_enabled
from ...tensor import Tensor
from ...backend_selection import *
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing_extensions import Self


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


T = TypeVar('T', bound='Module')


def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f"Module is missing the required \"forward\" function")


class Module:
    r"""下面引用Pytorch的Module类的注释
    Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    """

    training: bool  # 控制training/testing状态
    _parameters: Dict[str, Optional[Parameter]]  # 在训练过程中会随BP而更新的参数
    _buffers: Dict[str, Optional[Tensor]]  # 在训练过程中不会随着BP而更新的参数
    _non_persistent_buffers_set: Set[str]  # 不保存在模型中的buffer放到这里面
    _modules: Dict[str, Optional['Module']]  # 子神经网络模块

    r"""
    buffers与parameters唯一的区别就是前者不会随BP更新而后者会
    至于为什么要将参数分为buffers和parameters
    而不直接将所有参数都作为parameters，不随BP更新的require_grad设置为False
    关于buffers与parameters可以参考：
    https://zhuanlan.zhihu.com/p/89442276
    """

    def __init__(self) -> None:
        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_buffers', OrderedDict())
        super().__setattr__('_non_persistent_buffers_set', set())
        super().__setattr__('_modules', OrderedDict())
        super().__setattr__('device', backend_api.Device('cuda'))

    """
    Attention Please!!!
    在这里声明了'forward'方法
    Callable[..., Any] 表明它是一个可调用对象，接受任意数量和类型的参数，并返回任意类型的值
    Callable[..., Any] 的作用等价于

    def __call__(self, *input):
        return self.forward(*input)

    而forward方法默认值为 方法_forward_unimplemented
    因为forward本就是需要继承Module类的子类去实现的
    如果子类没有实现forward就调用
    那么就会调用默认的_forward_unimplemented方法
    从而引发报错

    总的来说，下面这段代码等价于

    class Module:
        def __call__(self, *input):
            return self.forward(*input)

        def forward(self, *input):
            raise NotImplementedError

    下面这种写法明显感觉更高级，其他优点不知道
    """

    #  forward: Callable[..., Any] = _forward_unimplemented
    def __call__(self, *input):
        return self.forward(*input)

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a models parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): buffer的名字
            tensor (Tensor or None): buffer的值. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool)：这个buffer是否是这个模型的一部分，决定了在保存模型时是否要保存这个buffer

        Example::

            > # xdoctest: +SKIP("undefined vars")
            > self.register_buffer('running_mean', zeros(num_features))

        """

        """对buffer进行一些简单的检查"""
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")  # 不能在模型初始化前注册buffer
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string")
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.
        没有persistent这个参数是因为凡是为parameter的参数都是必定要保存在模型中的
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string")
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"cannot assign this param to parameter '{name}' "
                            "(torch.nn.Parameter or None required)"
                            )
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        r"""Add a child module to the current module.
        将子模型添加到当前模型

        Args:
             name(str): 子模型的名字
             module (Module): 父模型
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"this module is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif '.' in name:
            raise KeyError(f"module name can't contain \".\", got: {name}")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def register_module(self, name: str, module: Optional['Module']) -> None:
        """与add_module功能相同"""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.
        通俗来说，就是通过查询现有模型中是否有子模型target

        下面举例说明

        设模型"A"，其结构如下:
        A(
           (net_b): Module(
               (net_c): Module(
                   (conv):Conv2d(16, 33, kernel_size=(3, 3), stride=(2,2))
               )
               (linear): Linear(in_features=100, out_features=200, bias=True)
           )
        )
        "A"有一个嵌套的子模型"net_b"，"net_b"又有两个子模型"net_c"和"linear"."net_c"又有一个子模型"conv"

        要检查"A"中是否有"linear"这个子模型，可以使用函数
        ``get_submodule("net_b.linear")``
        要检查是否有"conv"这个子模型，使用函数
        ``get_submodule("net_b.net_c.conv")``
        """
        if target == "":
            return self

        """
        对atoms，设有target=a.b.c
        则atoms=[a,b,c]
        """
        atoms: List[str] = target.split(".")
        # mod就是self，是该模型本身
        mod: Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                                       "attribute `" + item + "`")

            """
            使用'getattr'返回当前item的属性值
            """
            mod = getattr(mod, item)

            if not isinstance(mod, Module):
                raise AttributeError("`" + item + "` is not "
                                                  "an nn.Module")

        return mod

    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.
        """
        module_path, _, param_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + param_name + "`")

        param: Parameter = getattr(mod, param_name)

        if not isinstance(param, Parameter):
            raise AttributeError("`" + param_name + "` is not an "
                                                    "nn.Parameter")

        return param

    def get_buffer(self, target: str) -> "Tensor":
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + buffer_name + "`")

        buffer: Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.
        对模型的所有子模型应用函数'fn'

        Typical use includes initializing the parameters of a models.
        最常用的应用是初始化模型中的所有参数

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            > @torch.no_grad()
            > def init_weights(m):
            >     print(m)
            >     if type(m) == nn.Linear:
            >         m.weight.fill_(1.0)
            >         print(m.weight)
            > net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            > net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )

        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    # __getattr__用于访问类中是否有'name'代表的属性值
    # 有则返回，无则报错
    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    """
    ！！！attention please ！！！
    下面是Module类中一个究极重要的方法
    '__setattr__'的作用就是正确地将parameter,buffer,module添加到Module中
    """

    """
    '__setattr__'方法语法本身比较简单，该方法常与'__dict__'联系在一起
    关于'__setattr__'参考：
    https://zhuanlan.zhihu.com/p/101004827?from_voters_page=true
    关于'__dict__'参考：
    https://c.biancheng.net/view/2374.html
    """

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        """
        下面这个方法'remove_from'
        实现 删除 '*dice_or_sets'中一堆列表或元组中key为'name'的元素
        这在为parameter,buffer,module赋值时使用
        举个例子：
        假设要将'name'为'weight'的类型为'parameter'的参数添加进模型
        那么'weight'这个名字的参数就只能为parameter，不能是buffer,module
        所以要删除buffer,module中的'name'为'weight'的参数

        这说明 self.dict，self._buffers，self._parameters，self._modules 中的属性应该是互斥的
        """

        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        # __dict__本质上就是一个普通的字典

        # 对parameter
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            # 下面这段代码检查了继承 nn.Module 的自定义模块是否有正确地初始化父类 nn.Module
            # 这也说明了 super().__init__() 的必要性
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        # value并不是Parameter实例，但该name代表的参数已在模型中存在，说明这个value的赋值是修改之前该参数的赋值
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign this value as parameter '{name}' "
                                "(Parameter or None expected)"
                                )
            self.register_parameter(name, value)
        else:
            # 对module
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign this as child module '{name}' "
                                    "(nn.Module or None expected)"
                                    )
                modules[name] = value
            else:
                # 对buffer
                # 注意到buffer与parameter与module不同
                # buffer没有'if isinstance(value, Tensor):...'
                # 这说明想要给Module增加buffer,self.register_buffer是唯一的方式
                # __setattr__ 只能将 self._buffers 中已有的buffer重新赋值为None或者tensor
                # 目前没搞懂为什么要这样 :(

                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Tensor):
                        raise TypeError(f"cannot assign this as buffer '{name}' "
                                        "(Tensor or None expected)"
                                        )
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)

    # 在每次删除操作时会被调用
    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def params_and_buffers_saved(self):
        """
        限于技术，目前无法实现像pytorch那样的保存与加载功能
        所以在此处整个简单的方法来实现保存与加载

        这个方法的功能主要是将Module中的parameter和persistent为true的buffers合到一起
        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        # 关于itertools.chain()，可参考chatgpt
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}
        return local_state

    def _named_members(self, get_members_fn, prefix='', recurse=True, remove_duplicate: bool = True):
        r"""Help yield various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Return an iterator over module parmeters.

        经常在使用优化器更新模型参数时使用

        Yields:
            Parameter: module parameter

        Example::

            > for param in models.parameters():
            >     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        r"""Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): True:还会返回子模块的Parameter
                            Fasle:只返回该模块的Parameter
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            > for name, param in self.named_parameters():
            >     if name in ['bias']:
            >         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        yield from gen

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Return an iterator over module buffers.

        Args:
            recurse (bool): True:还会返回子模块的buffer
                            Fasle:只返回该模块的buffer

        Yields:
            Tensor: module buffer

        Example::

            > for buf in models.buffers():
            >     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[
        Tuple[str, Tensor]]:
        r"""Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): True:还会返回子模块的buffer
                            Fasle:只返回该模块的buffer
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, Tensor): Tuple containing the name and buffer

        Example::

            > for name, buf in self.named_buffers():
            >     if name in ['running_var']:
            >         print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        yield from gen

    def children(self) -> Iterator['Module']:
        r"""Return an iterator over immediate children modules.
        返回该模块的子模块

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        r"""Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            > for name, module in models.named_children():
            >     if name in ['conv4', 'conv5']:
            >         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        r"""Return an iterator over all modules in the network.
        返回该模型的所有模块，不返回模块名

        Yields:
            Module: a module in the network

        Note:
        重复的module只返回一次，例如下面的例子中，'l'只返回一次
        即没有输出:
        2 -> Linear(in_features=2, out_features=2, bias=True)

        Example::

            > l = nn.Linear(2, 2)
            > net = nn.Sequential(l, l)
            > for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""返回该模型的所有模块，包括名字和模型本身(both the name of the module as well as the module itself)

        Args:
            memo: 记录已经抛出(即已经yield)的模块
            prefix: 加在模块名字前的前缀
            remove_duplicate: 是否返回重复的模块

        Yields:
            (str, Module): Tuple of name and module

        Note:
        重复的module只返回一次，例如下面的例子中，'l'只返回一次
        即没有输出:
        2 -> ('1', Linear(in_features=2, out_features=2, bias=True))

        Example::

            > l = nn.Linear(2, 2)
            > net = nn.Sequential(l, l)
            > for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            # 返回 self._modules 下的 name 和 module 元组
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                # 递归调用和返回 module.named_modules
                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

    def train(self: T, mode: bool = True) -> T:
        r"""Set the module in training mode.
        将模型设置为训练模式

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        set_grad_enabled(self.training)
        # 要将子模型的模式也设置为训练
        # 这个时候就体现出上面一堆实现属性访问方法的作用
        for module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        """
        设置为评估模式
        """
        return self.train(False)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Reset gradients of all models parameters.
        重置所有参数Parameter的梯度

        Args:
            set_to_none (bool): True -> 将所有 Parameter 的梯度设为 None
                                False -> 将所有 Parameter 的梯度设为 0
        """
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    # 判断p是否是计算图中的叶子节点
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    """
    下面是一堆实现转移的函数
    """

    def to(self, device):
        if self.device.name == device:
            return self
        else:
            module = deepcopy(self)
            module.move(device)
            return module

    def move(self, device):
        self.device = backend_api.Device(device)
        for module in self._modules.values():
            if isinstance(module, Module):
                module.move(device)

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

