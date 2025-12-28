cmake_minimum_required(VERSION 3.28)
project(CUDA_BACKEND LANGUAGES CXX CUDA)

# 1. ���� CUDA �����ҹ��߰�
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# 2. �ִ���ʽ���� Python3���Զ��� Conda ������ʶ�����������
find_package(Python3 REQUIRED
    COMPONENTS Interpreter Development
)

# 3. pybind11 ʹ�����ҵ��� Python3
set(PYBIND11_FINDPYTHON ON)

# 4. ָ�� Conda �� pybind11 ������Ŀ¼��ȷ��·����ȷ��
find_package(pybind11 REQUIRED)

# 5. ���� CUDA ��չ��
add_library(CUDA_BACKEND SHARED ndarray_backend_cuda.cu)
add_definitions(-DPYBIND11_NO_STD_PAIR)
# 6. ����ѡ����� CUDA 11.5 + ������뾯�棩
target_compile_options(CUDA_BACKEND PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:
        /EHsc /utf-8 /std:c++14 /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        -arch=sm_52
        -std=c++14
        --expt-relaxed-constexpr
        -Xcompiler=/EHsc,/utf-8,/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    >
)

# 7. �������������� Python3::Python�������ֶ�·����
target_link_libraries(CUDA_BACKEND PRIVATE
    Python3::Python                # �Զ����� Conda ������ Python
    pybind11::module               # pybind11 ����Ŀ�꣨�Ѹ��� Python3 ���ã�
    CUDA::cudart                   # CUDA ����ʱ��
)

# 8. ���� Python ģ�����ԣ����� .pyd �ļ���
set_target_properties(CUDA_BACKEND  PROPERTIES
    PREFIX ""                      # ȥ��Ĭ��ǰ׺
    SUFFIX ".pyd"                  # Windows �� Python ģ���׺
    CUDA_SEPARABLE_COMPILATION OFF # �رշ�����룬���ټ�������
)