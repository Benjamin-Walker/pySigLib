from numba.cuda.cudadrv.runtime import Runtime
from setuptools import find_packages, setup, Extension
from setuptools.command.install import install
from pathlib import Path
import shutil
import os
import platform
from build_utils import get_b2, build_cpsig, build_cusig

REBUILD = True
USE_CUDA = 'CUDA_PATH' in os.environ
if 'CUSIG' in os.environ and os.environ['CUSIG'] == 0:
    USE_CUDA = False

# Only support Windows, Linux and MacOS
SYSTEM = platform.system()
if SYSTEM not in ['Windows', 'Linux', 'Darwin']:
    raise RuntimeError("Error while installing pySigLib: unsupported system '" + SYSTEM + "'")

# Don't support CUDA on MacOS
if SYSTEM == 'Darwin':
    USE_CUDA = False

if USE_CUDA:
    print("Found CUDA_PATH, attempting to build with CUDA. To build without CUDA, run 'CUSIG=0 pip install pysiglib' instead.")
else:
    print("Building without CUDA.")

# Get lib extension
if SYSTEM == 'Windows':
    LIB_EXT = '.dll'
elif SYSTEM == 'Linux':
    LIB_EXT = '.so'
else:
    LIB_EXT = '.dylib'

# Get lib names
LIBS = ['cpsig' + LIB_EXT]
if USE_CUDA:
    LIBS += ['cusig' + LIB_EXT]

class CustomInstall(install):
    def run(self):
        if REBUILD:
            get_b2(SYSTEM)
            build_cpsig(SYSTEM)
            if USE_CUDA:
                build_cusig(SYSTEM)
            parent_dir = Path(__file__).parent
            dir_ = parent_dir / 'pysiglib'

            for file in LIBS:
                path = parent_dir / 'siglib' / 'x64' / 'Release' / file
                shutil.copy(path, dir_)

        super().run()

setup(
    name='pysiglib',
    version="0.1.0",
    packages=['pysiglib'],
    include_package_data=True,
    package_data={'': LIBS},
    cmdclass={
        'install': CustomInstall,
    },
)
