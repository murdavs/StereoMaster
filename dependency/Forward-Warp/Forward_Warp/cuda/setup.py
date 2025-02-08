from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='forward_warp_cuda',
    ext_modules=[
        CUDAExtension(
            name='forward_warp_cuda',
            sources=[
                'forward_warp_cuda.cpp',
                'forward_warp_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': [
                    '/std:c++17',  # Fuerza C++17 en MSVC
                    '/MD',        # Runtime multithreaded DLL
                ],
                'nvcc': [
                    '-std=c++17',                 # Fuerza C++17 en NVCC
                    '-allow-unsupported-compiler',# Ignora verificación de versión MSVC
                    '-Xcompiler', '/MD',
                    '--expt-relaxed-constexpr',
                    # Opcionalmente, especifica gencode según tu GPU:
                    # '-gencode=arch=compute_75,code=sm_75',
                ],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
