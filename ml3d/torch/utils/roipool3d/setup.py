from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='roipool3d',
      ext_modules=[
          CUDAExtension('roipool3d_cuda', [
              'src/roipool3d.cpp',
              'src/roipool3d_kernel.cu',
          ],
                        extra_compile_args={
                            'cxx': ['-g'],
                            'nvcc': ['-O2']
                        })
      ],
      cmdclass={'build_ext': BuildExtension})
