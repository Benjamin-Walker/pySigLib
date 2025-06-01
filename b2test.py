import os
import sys
import subprocess

from build_utils import get_b2, build_cpp, nvcc_compile_and_link, get_paths, build_cusig

if __name__ == "__main__":
    build_cpp()

