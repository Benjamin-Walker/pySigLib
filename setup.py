from setuptools import find_packages, setup, Extension
from setuptools.command.install import install
from pathlib import Path
import shutil
import os
import platform
from b2_utils import get_b2, build_cpp

GETDLL = True
USE_CUDA = False

SYSTEM = platform.system()
if SYSTEM == 'Darwin':
    DATA = ['libcpsig.dylib']
else:
    DATA = ['cpsig.dll']
    if USE_CUDA:
        DATA += ['cusig.dll']

class CustomInstall(install):
    def run(self):
        if GETDLL:
            get_b2()
            build_cpp()
            parent_dir = Path(__file__).parent
            dir_ = parent_dir / 'pysiglib'

            if SYSTEM == "Windows":
                cpsig_dll_path = parent_dir / 'siglib' / 'dist' / 'release' / 'cpsig.dll'
                shutil.copy(cpsig_dll_path, dir_)
                if USE_CUDA:
                    cusig_dll_path = parent_dir / 'siglib' / 'dist' / 'release' / 'cusig.dll'
                    shutil.copy(cusig_dll_path, dir_)
            elif SYSTEM == "Darwin":
                cpsig_dll_path = parent_dir / 'siglib' / 'dist' / 'release' / 'libcpsig.dylib'
                shutil.copy(cpsig_dll_path, dir_)
            else:
                raise Exception("Unsupported OS during setup.py")

        super().run()

setup(
    name='pysiglib',
    version="1.0",
    packages=['pysiglib'],
    include_package_data=True,
    package_data={'': DATA},
    cmdclass={
        'install': CustomInstall,
    },
)
