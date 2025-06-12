set -e
set -x

CUDA_PATH="${CUDA_PATH}"
NVCC_EXE="${CUDA_PATH}/bin/nvcc"
VS_PATH0="$1"
VS_PATH="$2"  # Not used on Linux

SIGLIB_DIR="$(pwd)/siglib"
BUILD_DIR="${SIGLIB_DIR}/cusig/x64/Release"

mkdir -p "${BUILD_DIR}"

cd "${SIGLIB_DIR}/cusig"

# Build precompiled header equivalent (g++ only supports it with gch or precompiled modules in C++20)
# Optional step – comment out if unnecessary
g++ -std=c++20 -O2 -Wall -Wextra -DNDEBUG -DCUSIG_EXPORTS -fPIC -I"${CUDA_PATH}/include" -c cupch.cpp -o "${BUILD_DIR}/cupch.o"

echo "*** Compile CUDA files with nvcc ***"
NVCC_ARGS="-gencode=arch=compute_52,code=sm_52 --use-local-env -rdc=true -I${CUDA_PATH}/include --keep-dir ${BUILD_DIR} --machine 64 --compile -cudart static -lineinfo -DNDEBUG -DCUSIG_EXPORTS -Xcompiler -fPIC"

#${NVCC_EXE} ${NVCC_ARGS} -o "${BUILD_DIR}/cu_sig_kernel.cu.o" cu_sig_kernel.cu
#${NVCC_EXE} ${NVCC_ARGS} -o "${BUILD_DIR}/cu_sig_kernel.h.o" cu_sig_kernel.h

echo "*** Linking ***"

${NVCC_EXE} -shared -Xcompiler -fPIC -DNDEBUG -DCUSIG_EXPORTS \
    cu_sig_kernel.cu \
    -o ${SIGLIB_DIR}/x64/Release/libcusig.so

echo "*** Build complete. ***"
