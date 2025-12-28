from ..tensor import Tensor
from .. import backend_api


class Parameter(Tensor):
    def __init__(self, data: Tensor):
        super().__init__(
            array=data.data,
            dtype=data.dtype,
            device=data.device,
            requires_grad=True
        )

    def to(self, device):
        if self.device.name == device:
            return self
        elif device.device == "cpu":  # cuda -> cpu
            return self.__class__(
                Tensor(
                    self.data.numpy(),
                    dtype=self.dtype,
                    device=backend_api.Device(device),
                ))
        else:  # cpu -> cuda
            return self.__class__(
                Tensor(
                    self.data.numpy(),
                    dtype=self.dtype,
                    device=backend_api.Device(device),
                ))

    def __repr__(self) -> str:
        return "Parameter : \n{}".format(self.data) + (",\ndevice={}".format(
            self.device) if self.device.device != "cpu" else "")
