import functools

grad_enable = True


def is_grad_enable():
    return grad_enable


def set_grad_enabled(mode: bool):
    """
    'global'
    使多个函数共享grad_enable并且能在函数内引用它
    不使用该关键字处理会引发异常
    具体可见以下连接
    https://zhuanlan.zhihu.com/p/111284408
    """
    global grad_enable
    grad_enable = mode


"""
no_grad 临时关闭梯度计算
通过下面这个类就可以明白用Pytorch编码时写的 'with no_grad:'
"""


class no_grad:
    def __enter__(self) -> None:
        # 保存先前的 grad_enable
        self.prev = is_grad_enable()
        # 因为是no_grad，所以在此处要将grad_enable设置为False
        set_grad_enabled(False)

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        # 恢复先前的 grad_enable
        set_grad_enabled(self.prev)

    """
    下面是一个装饰器并且应用了functools
    这方面的python语法可以参考如下链接
    https://zhuanlan.zhihu.com/p/45458873
    关于__class__的知识，参考
    https://blog.csdn.net/weixin_44799217/article/details/126111905
    """
    def __call__(self, func):
        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)

        return decorate_context


"""
下面是与no_grad相反的类
"""


class enable_grad:
    def __enter__(self):
        self.prev = is_grad_enable()
        set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback):
        set_grad_enabled(self.prev)

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)

        return decorate_context
