import numpy as np
import numpy

from .autograd import is_grad_enable, no_grad
from typing import Any, List, Tuple, Union, Optional, Type
from .backend_selection import Device, backend_api, BackendTensor, default_device


class Graph:
    """
    动态计算图
    """
    node_list: list = list()

    @classmethod
    def add(cls, node):
        """ 将结点添加进计算图中"""
        cls.node_list.append(node)

    @classmethod
    def clear(cls):
        cls.node_list.clear()

    @classmethod
    def free_graph(cls):
        """
        释放计算图。
        关键修正：必须保留叶子节点（如模型权重），并清理它们的 children 引用，
        否则权重的 children 列表会无限膨胀导致内存泄漏。
        """
        keep_list = []
        for node in Graph.node_list:
            # 1. 在清理之前判断是否为叶子节点（权重/输入）
            # 必须现在判断，因为后面 clear() 之后 parents 就没了，所有节点都会变成“伪叶子”
            is_leaf_node = node.is_leaf
            
            # 2. 彻底切断引用
            node.children.clear() # <--- 这一步最关键！防止权重挂着旧的计算图
            node.parents.clear()
            
            # 3. 如果是叶子节点（权重），让它留在列表里，下一轮继续接受管理
            if is_leaf_node:
                keep_list.append(node)

        # 4. 更新全局列表，只保留权重，丢弃中间计算结果
        Graph.node_list = keep_list

    @classmethod
    def free_graph_all(cls):
        for node in Graph.node_list:
            node.children.clear()
            node.parents.clear()
        Graph.node_list = list() # 全部清空


_tensor_count = 0


class Tensor:
    """
    将数据(NumPy数组)包装成可微分张量

    Parameters
    ----------
    array :
        张量数据
    requires_grad : bool, default=False
        是否需要求梯度;
    dtype : default=None
        数据类型,和numpy数组的dtype等价

    Attributes
    ----------
    data : numpy.ndarray
        核心数据,为NumPy数组;
    requires_grad : bool
        是否需要求梯度;
    grad : numpy.ndarray
        梯度数据,为和data相同形状的数组(初始化为全0);
    children : list[Tensor]
        下游节点列表；
    parents : list[Tensor]
        上游节点列表.
    """

    def __init__(
            self,
            array,
            dtype="float32",
            device: Optional[Device] = None,
            name: Optional[str] = None,
            requires_grad: bool = False
    ) -> None:

        # 赋名
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        # 创建与backend关联的存储数据的类
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            # device和dtype跟传进来的array这个Tensor一样,那么直接可以用array内部的BackendTensor
            if device == array.device and dtype == array.dtype:
                self.data = array.data
            # 指定的device和dtype与array的不一样,那么就抽出array底部的具体数据,根据指定的device与dtype来重新创建BackendTensor
            else:
                self.data = Tensor._from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        elif isinstance(array, backend_api.BackendTensor):
            self.data = array
            # self.data = Tensor._from_numpy(array.numpy(), device=array.device, dtype=array.dtype)
        # 根据numpy来创建
        else:
            device = device if device else default_device()
            self.data = Tensor._from_numpy(array, device=device, dtype=dtype)

        # 对于一个tensor的是否需要梯度,还需要设置的此时整个代码区是否需要求梯度,即"with no_grad:"or"with enable_grad:"
        self.requires_grad: bool = requires_grad and is_grad_enable()

        """
        设置tensor的梯度grad
        如果requires_grad=True则初始化为跟data相同shape的全零array
        否则为None
        """
        self.grad = backend_api.zeros_like(self.data) if self.requires_grad else None

        # 初始化子节点和父节点列表
        self.children = list()
        self.parents = list()

        if self.requires_grad:
            # 在动态计算图中不需要求梯度的结点不出现在计算图中
            Graph.add(self)

    @staticmethod
    def _from_numpy(numpy_array, device, dtype):
        if backend_api is numpy:
            return numpy.array(numpy_array)
        return backend_api.Btensor(numpy_array, dtype=dtype, device=device)
    
    @staticmethod
    def make_const(self_Tensor, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor.__init__(self_Tensor, requires_grad=False)
        return tensor

    """
    @property 是 Python 中的一个装饰器,主要用于将一个方法变成属性调用。
    这使得我们可以在不调用方法的情况下访问其返回的值,就像访问属性一样。
    
    这一点在pytorch中有体现
    """

    @property
    def is_leaf(self) -> bool:
        # 判断是否为叶节点:需要求导且无上游节点的节点为叶节点
        return self.requires_grad and len(self.parents) == 0

    @property
    def shape(self) -> Tuple[int]:
        """张量的形状,用法同NumPy.

        Example
        -------
        > Tensor([[2, 2]]).shape
        (1, 2)
        """
        return self.data.shape

    @property
    def ndim(self):
        """张量的维度,用法同NumPy.

        Example
        -------
        > Tensor([[2, 2]]).ndim
        2
        """
        return self.data.ndim

    @property
    def dtype(self):
        """张量的数据类型,用法同NumPy.

        Example
        -------
        > Tensor([[2, 2]]).dtype
        dtype('int64')
        """
        return self.data.dtype

    @property
    def size(self):
        """张量的元素个数,用法同NumPy.

        Example
        -------
        >Tensor([[1, 1]]).size
        2
        """
        return self.data.size
    
    @property
    def device(self):
        if backend_api is numpy:
            return default_device()
        return self.data.device 
    
    def numpy(self):
        data = self.data
        if backend_api is numpy:
            return data
        return data.numpy() if not isinstance(data, tuple) else [x.numpy() for x in data]
    
    def detach(self):
        return Tensor.make_const(self)

    def dispose(self):
        if self.grad is not None and not self.is_leaf:
            self.grad = None
        self.children.clear()
        self.parents.clear()
        try:
            Graph.node_list.remove(self)
        except ValueError:
            pass

    @property
    def T(self):
        """张量的转置"""
        return self.transpose()

    def reshape(self, *new_shape):
        return Reshape(self, new_shape)
    
    def transpose(self, *axes):
        return transpose(self, axes if len(axes) != 0 else None)

    #def swapaxes(self, axis1: int, axis2: int):
        #return swapaxes(self, axis1, axis2)

    def max(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return max(self, axis, keepdims)

    #def min(
        #self,
        #axis: Union[int, Tuple, None] = None,
        #keepdims: bool = False,
    #):
        #return min(self, axis, keepdims)

    def sum(
            self,
            axis: Union[int, Tuple, None] = None,
            keepdims: bool = False,
    ):
        return sum(self, axis, keepdims)

    def build_edge(self, node):
        # 构建两节点的有向边
        self.children.append(node)
        node.parents.append(self)

    """
    下面的都是用魔法方法实现运算符重载
    简单地说,就是使非常数的那种操作数能像常数那样进行运算
    比如说,对两个tensor a和b,肯定不能直接像1+2那样运算
    如果想要实现a+b这样直接算的话,就需要使用魔法方法进行运算符重载
    比如说加法重载,就要在tensor类中定义重载加法: def __add__():
    然后在函数里面写代码具体实现这个加法
    最后,就可以实现两个自定义类a+b这样直接算
    """

    def __add__(self, x):
        return add(self, x)

    def __radd__(self, x):
        return add(x, self)

    def __sub__(self, x):
        return sub(self, x)

    def __rsub__(self, x):
        return sub(x, self)

    def __mul__(self, x):
        return mul(self, x)

    def __rmul__(self, x):
        return mul(x, self)

    def __matmul__(self, x):
        return matmul(self, x)

    def __rmatmul__(self, x):
        return matmul(x, self)

    def __truediv__(self, x):
        return div(self, x)

    def __rtruediv__(self, x):
        return div(x, self)

    def __pow__(self, x):
        return pow(self, x)

    def __rpow__(self, x):
        return pow(x, self)

    def __len__(self) -> int:
        return len(self.data)

    def __pos__(self):
        return self * 1

    def __neg__(self):
        return self * -1

    # def __abs__(self):
    #   return abs(self)

    def __getitem__(self, key):
        return get_slice(self, key)

    def __setitem__(self, key, value):
        """
        重载了切片/索引赋值的操作,我们不允许self允许求导,否则将出现错误
        """
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(key, Tensor):
            key = key.data
        if not isinstance(value, Tensor):
            self.data[key] = value
        else:
            self.data[key] = value.data

    def __iadd__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data += other
        return self

    def __isub__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data -= other
        return self

    def __imul__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data *= other
        return self

    def __itruediv__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data /= other
        return self

    def __imatmul__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data @= other
        return self

    @no_grad()
    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data < other)

    @no_grad()
    def __le__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data <= other)

    @no_grad()
    def eq(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data == other)

    @no_grad()
    def ne(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data != other)

    @no_grad()
    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data > other)

    @no_grad()
    def __ge__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data >= other)

    def backward(self, retain_graph: bool = False):
        if self not in Graph.node_list:
            return

        if self.data.ndim != 1:
            raise ValueError("backward should be called only on a scalar.")

        from .autograd import no_grad 
        
        with no_grad():
            # 1. 初始梯度必须是纯数据 (BackendTensor)
            # 使用 backend_api.ones_like(self.data) 返回的可能是 Tensor，必须取 .data
            init_grad = backend_api.ones_like(self.data)
            self.grad = init_grad.data if isinstance(init_grad, Tensor) else init_grad

            for i in range(len(Graph.node_list) - 1, -1, -1):
                if Graph.node_list[i] is self:
                    id = i
                    break

            for node in Graph.node_list[id::-1]:
                grad = node.grad
                if grad is None: 
                    continue
                
                # 确保流入的 grad 是纯数据，防止污染后续计算
                if isinstance(grad, Tensor):
                    grad = grad.data

                for parent in [p for p in node.parents if p.requires_grad]:
                    # 计算梯度
                    add_grad = node.grad_fn(parent, grad)

                    # ================= 核心修复 =================
                    # 无论 grad_fn 返回什么（Tensor 还是 BackendTensor），
                    # 我们统统只要 .data！
                    if isinstance(add_grad, Tensor):
                        add_grad = add_grad.data
                    # ===========================================

                    # 处理形状不匹配 (这部分逻辑保持你原有的，稍微优化类型判断)
                    if add_grad.shape != parent.shape:
                        # 转换为 numpy 处理广播 (假设 backend_api 能处理)
                        np_grad = add_grad.numpy() if hasattr(add_grad, 'numpy') else add_grad
                        
                        # 处理 expand_dims
                        if np_grad.ndim > parent.ndim:
                             np_grad = np.sum(
                                np_grad,
                                axis=tuple(range(np_grad.ndim - parent.ndim))
                             )
                        
                        # 处理 broadcast_to (keepdims=True)
                        # 简单的轴匹配逻辑
                        axis = tuple(i for i, (a, b) in enumerate(zip(np_grad.shape, parent.shape)) if a != b)
                        if axis:
                            np_grad = np.sum(np_grad, axis=axis, keepdims=True)
                        
                        # 转回 BackendTensor (确保是纯数据)
                        add_grad = backend_api.Btensor(np_grad, device=self.device)
                        # 如果 Btensor 返回的是 Tensor 包装器，再次剥离
                        if isinstance(add_grad, Tensor):
                            add_grad = add_grad.data
                    
                    # 累加梯度：此时 parent.grad 和 add_grad 都是纯 BackendTensor
                    if parent.grad is None:
                         parent.grad = add_grad
                    else:
                         # 如果 parent.grad 意外变成了 Tensor，这里强制纠正
                         if isinstance(parent.grad, Tensor):
                             parent.grad.data += add_grad
                         else:
                             parent.grad += add_grad

                if not node.is_leaf:
                    node.grad = None

        if not retain_graph:
            Graph.free_graph()

    def zero_grad(self):
        """梯度归零"""
        # 方案A：最安全，直接置空，节省内存（推荐）
        self.grad = None 
        
        # 方案B：如果你需要它一直是张量，确保它是 BackendTensor
        # temp = backend_api.zeros(self.shape, device=self.device)
        # self.grad = temp.data if isinstance(temp, Tensor) else temp

    # def item(self):
        # return self.data.item()

    def to(self, device):
        if self.device.name == device:
            return self
        elif device.name == "cpu":  # cuda -> cpu
            device = backend_api.cpu()
            return Tensor(self.data, dtype=self.dtype, device=device)
        else:  # cpu -> cuda
            device = backend_api.cuda()
            return Tensor(self.data, dtype=self.dtype, device=device)

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def __repr__(self) -> str:
        return "{}({}, requires_grad={}".format(
            "Tensor",
            self.data,
            self.requires_grad,
        ) + (", device={}".format(self.device)) + ")"
    
    def __str__(self):
        return self.data.__str__()
    

"""
Tensor的基本运算算子
"""


class UnaryOperator(Tensor):
    """
    一元操作的基类
    """

    def __init__(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        super().__init__(
            array=self.forward(x),
            device=x.device,
            requires_grad=is_grad_enable() and x.requires_grad,
        )

        if self.requires_grad:
            x.build_edge(self)

    def forward(self, x: Tensor):
        """前向传播函数,即定义该算子的具体运算操作"""
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad):
        """
        反向传播函数
        x : Tensor
            下游节点
        grad : ndarray
            上游流入该节点的梯度
        """
        raise NotImplementedError

    # 定义该类实例化信息
    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


class BinaryOperator(Tensor):
    """
    二元操作的基类
    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        if isinstance(x, Tensor) and isinstance(y, BackendTensor):
            assert x.device == y.device
            self.x = x.data
            self.y = y
            y = Tensor(y)
        elif isinstance(x, Tensor) and isinstance(y, Tensor):
            assert x.device == y.device
            self.x = x.data
            self.y = y.data
        else:
            self.x = x.data
            self.y = np.float32(y)
            y = Tensor(np.array([y]), device=x.device)

        super().__init__(
            array=self.forward(self.x, self.y),
            device=x.device,
            requires_grad=is_grad_enable()
            and (x.requires_grad or y.requires_grad)
        )

        if self.requires_grad:
            x.build_edge(self)
            y.build_edge(self)

    def forward(self, x, y):
        raise NotImplementedError
    
    def gard_fn(self, x, grad):
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


class add(BinaryOperator):
    """
    加法算子
    """
    def forward(self, x, y):
        return x + y

    def grad_fn(self, node: Tensor, grad):
        return grad


class sub(BinaryOperator):
    def forward(self, x, y):
        return x - y

    def grad_fn(self, node: Tensor, grad):
        if node is self.parents[0]:
            return grad
        return -grad


class mul(BinaryOperator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self, x, y):
        return x * y

    def grad_fn(self, node: Tensor, grad):
        if node is self.parents[0]:
            return grad * self.parents[1].data
        return grad * self.parents[0].data


class div(BinaryOperator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self, x, y):
        return x / y

    def grad_fn(self, node: Tensor, grad):
        temp = grad / self.parents[1].data
        if node is self.parents[0]:
            return temp
        return -self.data * temp


class pow(BinaryOperator):
    """
    幂运算算子
    """

    def __init__(self, x, y) -> None:
        super().__init__(x, y)

    def forward(self, x, y):
        return x ** y

    def grad_fn(self, node: Tensor, grad):
        if node is self.parents[0]:
            return (self.data * self.parents[1].data / node.data) * grad
        else:
            return self.data * backend_api.log(self.parents[0].data) * grad


class matmul(BinaryOperator):
    """
    矩阵乘法算子
    """

    def __init__(self, x, y) -> None:
        super().__init__(x, y)

    def forward(self, x, y):
        return x @ y

    def grad_fn(self, node: Tensor, grad):
        if node is self.parents[0]:
            if self.parents[1].ndim == 1:
                return backend_api.expand_dims(grad, -1) @ backend_api.expand_dims(
                    self.parents[1].data, -2)
            elif self.parents[1].ndim > 2:
                shape = list(range(self.parents[1].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return grad @ self.parents[1].data.transpose(*shape)
            return grad @ self.parents[1].data.transpose()
        else:
            if self.parents[0].ndim == 1:
                return backend_api.expand_dims(self.parents[0].data, -1) @ backend_api.expand_dims(grad, -2)
            elif self.parents[0].ndim > 2:
                shape = list(range(self.parents[0].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return self.parents[0].data.transpose(*shape) @ grad
            return self.parents[0].data.transpose() @ grad

"""
class abs(UnaryOperator):

    def forward(self, x: Tensor):
        return self.xp.abs(x)

    def grad_fn(self, x: Tensor, grad):
        mask = self.xp.zeros(x.shape)
        mask[x > 0] = 1.
        mask[x < 0] = -1.
        return grad * mask
"""


class sum(UnaryOperator):

    def __init__(self, x: Tensor, axes: Optional[tuple] = None, keepdims=False) -> None:
        self.axes = axes
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor):
        a = x.data
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis=axis)
            return a
        return backend_api.summation(a, axis=self.axes, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad):
        if not (self.axes is None or self.keepdims):
            gard = backend_api.expand_dims(grad, axis=self.axes)
        return backend_api.ones(x.shape, device=x.device) * grad


class mean(UnaryOperator):

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor):
        return backend_api.mean(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad):
        if not (self.axis is None or self.keepdims):
            grad = backend_api.expand_dims(grad, axis=self.axis)
        return backend_api.ones(x.shape, device=x.device) * grad * self.data.size / x.data.size


class max(UnaryOperator):

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor):
        return backend_api.max(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad):
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 1. 补回被max压缩的维度
            full_dim_y = backend_api.expand_dims(self.data, axis=self.axis)
            grad = backend_api.expand_dims(grad, axis=self.axis)

        # 2. 关键修复：强制full_dim_y广播到x.data的形状（确保完全匹配）
        full_dim_y = full_dim_y.broadcast_to(x.data.shape)

        # 3. 计算梯度（此时形状一致，无广播错误）
        return (full_dim_y == x.data) * grad

"""
class min(UnaryOperator):

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.min(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = self.xp.expand_dims(self.data, axis=self.axis)
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad



class argmax(Tensor):
    def __init__(self, x: Tensor, axis=None) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        self.device = x.device
        super().__init__(self.forward(x), device=self.device)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.argmax(x.data, axis=self.axis)


class argmin(Tensor):
    def __init__(self, x: Tensor, axis=None) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        self.device = x.device
        super().__init__(self.forward(x), device=self.device)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.argmin(x.data, axis=self.axis)
"""


class exp(UnaryOperator):
    """指数运算

    Example
    -------
    > x = Tensor(1.)
    > y = exp(x)
    """

    def forward(self, x: Tensor):
        return backend_api.exp(x.data)

    def grad_fn(self, x: Tensor, grad):
        return self.data * grad


class log(UnaryOperator):
    """对数运算

    Example
    -------
    > x = Tensor(1.)
    > y = log(x)
    """

    def forward(self, x: Tensor):
        return backend_api.log(x.data)

    def grad_fn(self, x: Tensor, grad):
        return grad / x.data


class maximum(BinaryOperator):
    def forward(self, x, y):
        return backend_api.maximum(x, y)

    def grad_fn(self, x: Tensor, grad):
        return (self.data == x.data) * grad


"""
class minimum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return self.xp.minimum(x, y)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x) * grad
"""


def sqrt(x: Tensor):
    """平方根函数"""
    return x ** 0.5


def square(x: Tensor):
    """平方函数"""
    return x * x


# 非计算函数
class Reshape(UnaryOperator):
    """
    张量形状变换算子

    Parameters
    ----------
    new_shape : tuple
        变换后的形状,用法同NumPy
    """
    def __init__(self, x: Tensor, new_shape):
        self.new_shape = new_shape
        super().__init__(x)

    def forward(self, x: Tensor):
        return x.data.compact().reshape(self.new_shape)

    def grad_fn(self, x: Tensor, grad):
        return grad.reshape(x.shape)


class transpose(UnaryOperator):
    """
    张量转置算子

    Parameters
    ----------
    axes : tuple
        转置的轴变换,用法同NumPy
    """
    def __init__(self, x: Tensor, axes: Optional[tuple] = None):
        self.axes = axes
        super().__init__(x)

    def forward(self, x: Tensor):
        return x.data.transpose(self.axes)

    def grad_fn(self, x: Tensor, grad):
        if self.axes is None:
            return grad.transpose()
        return grad.transpose(tuple(np.argsort(self.axes)))


class get_slice(UnaryOperator):
    """
    切片算子,为Tensor类提供索引和切片接口

    Example
    -------
    > x = Tensor(
            np.arange(12).reshape(3, 4).astype(float),
            requires_grad=True,
        )
    > y = x[:2, :2].sum()
    > y.backward()
    > x.grad
    [[1. 1. 0. 0.]
     [1. 1. 0. 0.]
     [0. 0. 0. 0.]]
    """

    def __init__(self, x: Tensor, key):
        if isinstance(key, Tensor):
            self.key = key.data
        else:
            self.key = key
        super().__init__(x)

    def forward(self, x: Tensor):
        return x.data[self.key]

    def grad_fn(self, x: Tensor, grad):
        full_grad = backend_api.zeros(x.shape)
        full_grad[self.key] = grad
        return full_grad


"""
class swapaxes(UnaryOperator):
    # 张量交换算子
    def __init__(self, input: Tensor, axis1: int, axis2: int) -> None:
        self.axis1 = axis1
        self.axis2 = axis2
        super().__init__(input)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.swapaxes(self.axis1, self.axis2)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad.swapaxes(self.axis1, self.axis2)
"""


"""
class concatenate(Tensor):
    # 对多个张量进行连接,用法类似于`numpy.concatenate`

    Parameters
    ----------
    tensors :
        待连接的张量: 
    axis : default=0
        连接轴,默认是沿着第一个轴拼接.

    def __init__(self, tensors: List[Tensor], axis=0) -> None:
        requires_grad = False
        self.tensors = tensors
        self.axis = axis
        self.indices = [0]

        for i in range(len(self.tensors)):
            assert isinstance(
                tensors[i],
                Tensor), "Concatenate elements in 'tensors' must be 'Tensor'"
            if i == 0:
                device = tensors[i].device
            else:
                assert tensors[i].device == device
            requires_grad = requires_grad or self.tensors[i].requires_grad
            self.indices.append(self.indices[-1] +
                                self.tensors[i].shape[self.axis])
        self.device = device
        super().__init__(self.forward(),
                         requires_grad=requires_grad and is_grad_enable(),
                         device=device)
        if self.requires_grad:
            for i in range(len(self.tensors)):
                self.tensors[i].build_edge(self)

    def forward(self):
        return self.xp.concatenate([t.data for t in self.tensors],
                                   axis=self.axis)

    def grad_fn(self, x, grad: np.ndarray):
        x_id = self.tensors.index(x)
        start = self.indices[x_id]
        end = self.indices[x_id + 1]
        slc = [slice(None)] * grad.ndim
        slc[self.axis] = slice(start, end)
        return grad[tuple(slc)]


def unsqueeze(x: Tensor, axis: Any):
    # 等价于numpy的expand_dims
    if type(axis) not in (tuple, list):
        axis = (axis, )

    out_ndim = len(axis) + x.ndim
    axis = normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(x.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return x.reshape(*shape)


def normalize_axis_tuple(axis, ndim):
    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    axis = tuple(np.arange(ndim)[np.asarray(axis) % ndim])
    return axis
"""


# 一些包装的特殊矩阵
def empty(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.empty(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def zeros(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.zeros(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def ones(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.ones(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def randn(*shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.random.randn(*shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def rand(*shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.random.rand(*shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def uniform(low: float,
            high: float,
            shape=None,
            dtype=None,
            device=None,
            requires_grad=False):
    return Tensor(np.random.uniform(low, high, size=shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)

