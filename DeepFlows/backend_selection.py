import os

# BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")
BACKEND = "nd"

if BACKEND == "nd":
    from . import backend as backend_api
    from .backend import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        gpu_cupy,
        default_device,
        BackendDevice as Device,
    )

    BackendTensor = backend_api.BackendTensor
elif BACKEND == "np":
    import numpy as backend_api
    # from .backend_numpy import all_devices, cpu, default_device, Device

    BackendTensor = backend_api.ndarray
else:
    raise RuntimeError("Unknown needle array backend %s" % BACKEND)
