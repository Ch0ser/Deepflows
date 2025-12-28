import sys
import numpy as np

# 1. 添加模块路径（基于当前脚本相对路径，稳健）
from pathlib import Path
build_dir = Path(__file__).resolve().parent / ".." / "DeepFlows" / "backend" / "backend_src" / "build" / "Release"
sys.path.insert(0, str(build_dir.resolve()))
# 2. 诊断与导入（优先动态加载 .pyd，兼容带 ABI 后缀）
try:
    import os, platform
    import importlib.machinery as _machinery
    import importlib.util as _import_util
    from glob import glob as _glob
    # Windows 3.8+ 需要显式加入依赖 DLL 目录（如 CUDA 的 bin）
    if hasattr(os, "add_dll_directory"):
        cuda_env = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_PATH_V12_2") or os.environ.get("CUDA_PATH_V11_8")
        if cuda_env:
            dll_dir = os.path.join(cuda_env, "bin")
            if os.path.isdir(dll_dir):
                os.add_dll_directory(dll_dir)
                print(f"[Info] 已加入 CUDA DLL 目录: {dll_dir}")
        # 也将本地 Release 目录加入 DLL 搜索路径，便于依赖解析
        os.add_dll_directory(str(build_dir.resolve()))
        # 动态查找并按绝对路径加载任意 CUDA_BACKEND*.pyd
    candidates = _glob(str(build_dir / 'CUDA_BACKEND*.pyd'))
    pyd_path = candidates[0]
    loader = _machinery.ExtensionFileLoader('CUDA_BACKEND', pyd_path)
    spec = _import_util.spec_from_file_location('CUDA_BACKEND', pyd_path, loader=loader)
    if spec is None or spec.loader is None:
        raise ImportError('无法创建加载规范 spec')
    cuda = _import_util.module_from_spec(spec)
    spec.loader.exec_module(cuda)
    print("✅ 模块导入成功！")
except Exception as e:
    print(f"❌ 模块导入失败：{e}")
# 3. 测试1：创建CUDA数组 + fill填充值
print("\n=== 测试 fill 函数（填充标量）===")
try:
    # 创建一个包含 5 个元素的 CUDA 数组
    cuda_arr = cuda.Array(5)
    print(cuda_arr)
    # 填充值为 3.14
    cuda.fill(cuda_arr, 3.14)
    # 传输到NumPy查看结果
    np_arr = cuda.to_numpy(cuda_arr, shape=[5], strides=[1], offset=0)
    print(f"CUDA数组填充后的值：{np_arr}")
    assert np.allclose(np_arr, [3.14, 3.14, 3.14, 3.14, 3.14]), "fill 测试失败"
    print("✅ fill 函数测试通过！")
except Exception as e:
    print(f"❌ fill 函数测试失败：{e}")

# 4. 测试2：element-wise加法（ewise_add）
print("\n=== 测试 ewise_add 函数（元素级加法）===")
try:
    # 生成两个NumPy数组
    np_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    np_b = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)

    # 创建CUDA数组并从NumPy拷贝数据
    cuda_a = cuda.Array(len(np_a))
    cuda_b = cuda.Array(len(np_b))
    cuda_out = cuda.Array(len(np_a))  # 输出数组
    cuda.from_numpy(np_a, cuda_a)
    cuda.from_numpy(np_b, cuda_b)

    # 调用元素级加法
    cuda.ewise_add(cuda_a, cuda_b, cuda_out)

    # 结果回传NumPy并验证
    np_out = cuda.to_numpy(cuda_out, shape=[5], strides=[1], offset=0)
    print(f"输入a：{np_a}")
    print(f"输入b：{np_b}")
    print(f"输出a+b：{np_out}")
    assert np.allclose(np_out, np_a + np_b), "ewise_add 测试失败"
    print("✅ ewise_add 函数测试通过！")
except Exception as e:
    print(f"❌ ewise_add 函数测试失败：{e}")

# 5. 测试3：标量加法（scalar_add）
print("\n=== 测试 scalar_add 函数（标量+数组）===")
try:
    np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scalar_val = 5.0  # 标量值

    cuda_a = cuda.Array(len(np_a))
    cuda_out = cuda.Array(len(np_a))
    cuda.from_numpy(np_a, cuda_a)

    # 标量 + 数组
    cuda.scalar_add(cuda_a, scalar_val, cuda_out)

    np_out = cuda.to_numpy(cuda_out, shape=[3], strides=[1], offset=0)
    print(f"输入数组：{np_a}")
    print(f"标量值：{scalar_val}")
    print(f"输出数组+标量：{np_out}")
    assert np.allclose(np_out, np_a + scalar_val), "scalar_add 测试失败"
    print("✅ scalar_add 函数测试通过！")
except Exception as e:
    print(f"❌ scalar_add 函数测试失败：{e}")

print("\n🎉 所有测试执行完毕！")