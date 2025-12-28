import operator
import math
from functools import reduce
import numpy as np


def prod(x):       
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device which wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name 
    
    def __repr__(self):
        return self.name + "()"
    
    def __getattr__(self, name):
        return getattr(self.mod, name)
    
    def enabled(self):
        return self.mod is not None
    
    def randn(self, *shape, dtype="float32"):
        return BackendTensor(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return BackendTensor(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return BackendTensor(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return BackendTensor.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr
    

def cuda():
    """Return cuda device"""
    try:
        from DeepFlows.backend.backend_src.build.Release import  CUDA_BACKEND

        return BackendDevice("cuda", CUDA_BACKEND)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    class NumpyBackend:
        def Array(self, size):
            return np.zeros(int(size), dtype=np.float32)

        def fill(self, handle, value):
            handle[...] = np.float32(value)

        def from_numpy(self, np_array, handle):
            handle[:] = np_array.reshape(-1).astype(np.float32)

        def to_numpy(self, handle, shape, strides, offset):
            base = handle
            sh = np.array(shape, dtype=np.int64)
            st = np.array(strides, dtype=np.int64)
            if sh.size == 0:
                return np.array([], dtype=np.float32)
            extent = int(np.sum((sh - 1) * st)) if len(shape) > 0 else 0
            if offset >= 0 and offset + extent < base.size:
                base2 = base[offset:]
                byte_strides = tuple(int(s) * base2.itemsize for s in st)
                return np.lib.stride_tricks.as_strided(base2, shape=tuple(sh.tolist()), strides=byte_strides).copy()
            idx = np.indices(tuple(sh.tolist()), dtype=np.int64)
            lin = offset + np.sum(idx * st.reshape((len(shape),) + (1,) * len(shape)), axis=0)
            out = base[lin]
            return out.copy()

        def ewise_setitem(self, other_handle, view_handle, shape, strides, offset):
            view = self.to_numpy(view_handle, shape, strides, offset)
            view[...] = other_handle.reshape(view.shape)
            view_handle[:view.size] = view.reshape(-1)

        def scalar_setitem(self, size, scalar, view_handle, shape, strides, offset):
            view = self.to_numpy(view_handle, shape, strides, offset)
            view[...] = np.float32(scalar)
            view_handle[:view.size] = view.reshape(-1)

        def compact(self, src_handle, dst_handle, shape, strides, offset):
            arr = self.to_numpy(src_handle, shape, strides, offset)
            dst_handle[:] = arr.reshape(-1)

        def ewise_add(self, a, b, out):
            out[:] = a + b

        def scalar_add(self, a, s, out):
            out[:] = a + np.float32(s)

        def ewise_mul(self, a, b, out):
            out[:] = a * b

        def scalar_mul(self, a, s, out):
            out[:] = a * np.float32(s)

        def ewise_div(self, a, b, out):
            out[:] = a / b

        def scalar_div(self, a, s, out):
            out[:] = a / np.float32(s)

        def scalar_power(self, a, s, out):
            out[:] = a ** np.float32(s)

        def ewise_maximum(self, a, b, out):
            out[:] = np.maximum(a, b)

        def scalar_maximum(self, a, s, out):
            out[:] = np.maximum(a, np.float32(s))

        def ewise_eq(self, a, b, out):
            out[:] = (a == b).astype(np.float32)

        def scalar_eq(self, a, s, out):
            out[:] = (a == np.float32(s)).astype(np.float32)

        def ewise_ge(self, a, b, out):
            out[:] = (a >= b).astype(np.float32)

        def scalar_ge(self, a, s, out):
            out[:] = (a >= np.float32(s)).astype(np.float32)

        def ewise_log(self, a, out):
            out[:] = np.log(a)

        def ewise_exp(self, a, out):
            out[:] = np.exp(a)

        def ewise_tanh(self, a, out):
            out[:] = np.tanh(a)

        def matmul(self, a, b, out, m, n, p):
            A = a.reshape(int(m), int(n))
            B = b.reshape(int(n), int(p))
            C = A @ B
            out[:] = C.reshape(-1)

        def reduce_sum(self, view_handle, out_handle, last_dim):
            last = int(last_dim)
            outer = int(view_handle.size // last)
            view = view_handle.reshape(outer, last)
            out_handle[:outer] = view.sum(axis=1).astype(np.float32)

        def reduce_max(self, view_handle, out_handle, last_dim):
            last = int(last_dim)
            outer = int(view_handle.size // last)
            view = view_handle.reshape(outer, last)
            out_handle[:outer] = view.max(axis=1).astype(np.float32)

    return BackendDevice("cpu", NumpyBackend())


def gpu_cupy():
    """Return cupy device"""
    pass


def cpu():
    """Return cpu device"""
    return cpu_numpy()


def default_device():
    return cpu_numpy()


def all_devices():
    """Return a dict of all available devices"""
    return {"cpu": cpu(), "cuda": cuda(),
            "cpu_numpy": cpu_numpy(), "gpu_cupy": gpu_cupy()}


def Device(device_name=None):
    return all_devices()[device_name]


class BackendTensor:
    """ A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.
    """

    def __init__(self, other, device=None) -> None:
        """ Create by copying another BackendTensor, or form numpy"""
        if isinstance(other, BackendTensor):
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other.astype(np.float32)), array._handle)
            self._init(array)
        else:
            array = BackendTensor(np.array(other), device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        """ Utility function to compute compact strides """
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """
        Create a new BackendTensor with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing array.
        """
        array = BackendTensor.__new__(BackendTensor)
        array._shape = tuple(shape)
        array._strides = BackendTensor.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array._device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """ Return number of dimensions. """
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)
    
    def __repr__(self):
        return "BackendTensor(" + self.numpy().__str__() + f", device={self.device})"
    
    def __str__(self):
        return self.numpy().__str__()
    
    ### Basic array manipulation
    def fill(self, value):
        """ Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device):
        """ Convert between devices, using to/from numpy calls as the unifying bridge. """
        if device == self.device:
            return self
        else:
            return BackendTensor(self.numpy(), device=device)
        
    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        return(
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        if self.is_compact():
            return self
        else:
            out = BackendTensor.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """ Restride the matrix without copying memory. 
            在make的过程中传递了被复制对象的handle,所以这两者的handle指向同一片内存区域
        """
        assert len(shape) == len(strides)
        return BackendTensor.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )
    
    @property
    def flat(self):
        """展平成一维的"""
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            BackendTensor : reshaped array; this will point to thep
        """

        if new_shape[0] == -1:
            _new_shape = list(new_shape)
            shapes = 1
            for dim in _new_shape[1:]:
                shapes = dim * shapes
            _new_shape[0] = int(prod(self._shape) / shapes)
            new_shape = tuple(_new_shape)
        elif new_shape[-1] == -1:
            _new_shape = list(new_shape)
            shapes = 1
            for dim in _new_shape[:-1]:
                shapes = dim * shapes
            _new_shape[-1] = int(prod(self._shape) / shapes)
            new_shape = tuple(_new_shape)

        if prod(new_shape) != prod(self._shape):
            raise ValueError("Product of current shape is not equal to \
                              the product of the new shape!")
        
        if not self.is_compact():
            raise ValueError("The matrix is not compact!")
        return BackendTensor.make(new_shape, BackendTensor.compact_strides(new_shape), self._device, self._handle)
    
    def transpose(self, new_axes=None):
        permute_axes = list(range(self.ndim))
        if new_axes:
            for i in range(self.ndim):
                permute_axes[i] = new_axes[i]
        else:
            for i in range(self.ndim):
                permute_axes[i] = self.ndim - i - 1
        return self.permute(permute_axes)

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            BackendTensor : new BackendTensor object with permuted dimensions, pointing
            to the same memory as the original BackendTensor (i.e., just shape and
            strides changed).
        """

        new_shape = tuple(np.array(self._shape)[list(new_axes)])
        new_strides = tuple(np.array(self._strides)[list(new_axes)])
        return BackendTensor.make(new_shape, new_strides, self._device, self._handle)

    def broadcast_to(self, new_shape):
        orig_shape = list(self._shape)
        target_dim = len(new_shape)
        orig_dim = len(orig_shape)

        # 补前导1，使orig_shape维度数与目标一致
        pad = target_dim - orig_dim
        if pad > 0:
            orig_shape = [1] * pad + orig_shape  # 补pad个前导1
        assert len(orig_shape) == target_dim, f"Cannot broadcast {self._shape} to {new_shape}"

        # 检查每个维度是否可广播
        for x, y in zip(orig_shape, new_shape):
            assert x == y or x == 1, f"Dimension mismatch: {x} vs {y}"

        # 计算新步长（核心修复部分）
        new_strides = []
        for i in range(target_dim):
            if i < pad:
                # 前导补的维度：步长为0（不移动内存）
                new_strides.append(0)
            else:
                # 原张量自身的维度（索引为i - pad）
                orig_idx = i - pad
                if orig_shape[i] == 1:
                    # 原维度被广播（值为1），步长设为0
                    new_strides.append(0)
                else:
                    # 原维度未被广播，沿用原步长
                    new_strides.append(self._strides[orig_idx])

        return BackendTensor.make(new_shape, tuple(new_strides), self._device, self._handle)
    
    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)
    
    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            BackendTensor: a new BackendTensor object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """

        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        # 搞懂下面三个计算的原理
        new_shape = [(sl.stop - sl.start + sl.step - 1) // sl.step for sl in idxs]
        offset = sum([sl.start * st for sl, st in zip(idxs, self._strides)])
        new_strides = tuple([st * sl.step for st, sl in zip(self._strides, idxs)])
        return BackendTensor.make(new_shape, new_strides, self._device, self._handle, offset)

    def __setitem__(self, idxs, other):
        """Set the value of a view into an array, using the same semantics 
        as __getitem__()"""
        view = self.__getitem__(idxs)
        if isinstance(other, BackendTensor):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    def __len__(self):
        return self.shape[0]

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        out = BackendTensor.make(self.shape, device=self.device)
        if isinstance(other, BackendTensor):
            if self.shape != other.shape:
                # 修复：删除强制reshape((1,1))的逻辑，让broadcast_to自动补维度
                other = other.broadcast_to(self.shape)  # 直接调用修复后的broadcast_to
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )
    
    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )
    
    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = BackendTensor.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)
    
    def __ne__(self, other):
        return 1 - (self == other)
    
    def __lt__(self, other):
        return 1 - (self >= other)
    
    def __le__(self, other):
        return 1 - (self > other)
    
    def log(self):
        out = BackendTensor.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = BackendTensor.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = BackendTensor.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    def __matmul__(self, other):
        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        out = BackendTensor.make((m, p), device=self.device)
        self.device.matmul(
            self.compact()._handle, other.compact()._handle, out._handle, m, n, p
        )
        return out

    def reduce_view_out(self, axis, keepdims=False):
        """ Return a view to the array set up for reduction functions and output array. """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")
        
        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = BackendTensor.make((1,) * (self.ndim if keepdims else 1), device=self.device)
        
        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = BackendTensor.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return view, out
    
    def sum(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def mean(self, axis=None, keepdims=False):
        sum = self.sum(axis, keepdims=keepdims)
        total_num = prod(self.shape)
        return sum / total_num


    def flip(self, axes):
        """
        Flip this BackendTensor along the specified axes.
        Note: compact() before returning.
        """
        assert len(axes) <= len(self.shape)
        new_strides = list(self.strides)
        for axis in axes:
            new_strides[axis] = - new_strides[axis]
        new_strides = tuple(new_strides)
        new_offset = sum([(self.shape[axis] - 1) * self.strides[axis] for axis in axes])
        return BackendTensor.make(self.shape, new_strides, self._device, self._handle, new_offset).compact()

    def pad(self, axes):
        """
        Pad this BackendTensor by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        assert len(axes) == len(self.shape)
        new_shape = tuple([l + r + n for (l, r), n in zip(axes, self.shape)])
        arr = self.device.full(new_shape, 0)
        access = tuple([slice(l, l + n) for (l, _), n in zip(axes, self.shape)])
        arr[access] = self
        return arr


def Btensor(a, dtype="float32", device=None):
    dtype = "float32" if dtype is None else dtype
    return BackendTensor(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def zeros(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, 0., dtype)


def ones(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, 1., dtype)


def zeros_like(data):
    device = data.device
    dtype = data.dtype
    shape = data.shape
    return device.full(shape, 0., dtype)


def ones_like(data):
    device = data.device
    dtype = data.dtype
    shape = data.shape
    return device.full(shape, 1., dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def max(a, axis=None, keepdims=False):
    return a.max(axis, keepdims)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def flip(a, axes):
    return a.flip(axes)


def summation(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)


def mean(a, axis=None, keepdims=False):
    return a.mean(axis=axis, keepdims=keepdims)


def pad(a, axes):
    return a.pad(axes)


def expand_dims(a, axis):
    if abs(axis) > a.ndim:
        raise ValueError("axis {} is out of bounds for Btensor of dimension {}".format(axis, a.ndim))
    new_shape = a.shape[:axis] + (1,) + a.shape[axis:]
    return a.reshape(new_shape)

