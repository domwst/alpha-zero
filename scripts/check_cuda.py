import ctypes
import traceback

import torch


print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_cuda_is_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")

try:
    driver = ctypes.CDLL("libcuda.so.1")
    driver_result = driver.cuInit(0)
    driver_count = ctypes.c_int(-1)
    driver_count_result = driver.cuDeviceGetCount(ctypes.byref(driver_count))
    print(f"cuInit={driver_result}")
    print(f"cuDeviceGetCount={driver_count_result}, count={driver_count.value}")
except Exception:
    traceback.print_exc()

try:
    torch.cuda.init()
    tensor = torch.zeros(1, device="cuda")
    print(f"cuda_tensor={tensor}")
    convolution = torch.nn.Conv2d(2, 32, 3, padding=1, device="cuda")
    convolution_output = convolution(torch.zeros(8, 2, 19, 19, device="cuda"))
    torch.cuda.synchronize()
    print(f"cuda_convolution_shape={tuple(convolution_output.shape)}")
except Exception:
    traceback.print_exc()
