#!/usr/bin/env bash
# Cross-compile vad-runtime for Android (arm64-v8a, ONNX).
# If build-android/ already exists and is configured, only runs make.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build-android"

ANDROID_NDK="${ANDROID_NDK:-/home/jingsz/other/android-ndk-r19c}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-21}"
JOBS="${JOBS:-$(nproc)}"

TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
ORT_SO="${PROJECT_ROOT}/third_party/onnxruntime-android/lib/${ANDROID_ABI}/libonnxruntime.so"

if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
    echo "Error: NDK toolchain not found: ${TOOLCHAIN_FILE}" >&2
    echo "Set ANDROID_NDK (current: ${ANDROID_NDK})" >&2
    exit 1
fi

if [[ ! -f "${ORT_SO}" ]]; then
    echo "==> ONNX Runtime Android not found, running setup..."
    bash "${SCRIPT_DIR}/setup_onnxruntime_android.sh"
fi

if [[ ! -f "${ORT_SO}" ]]; then
    echo "Error: ${ORT_SO} still missing after setup" >&2
    exit 1
fi

if [[ -d "${BUILD_DIR}" && -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "==> ${BUILD_DIR} exists, skip cmake (use rm -rf build-android to reconfigure)"
    cd "${BUILD_DIR}"
    make -j"${JOBS}"
else
    echo "==> Configure Android build in ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DANDROID_ABI="${ANDROID_ABI}" \
        -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
        -DANDROID_STL=c++_shared \
        -DENABLE_ONNX=ON \
        -DENABLE_RKNN=OFF
    make -j"${JOBS}"
fi

echo "==> Done:"
echo "    ${BUILD_DIR}/libfsmn_vad.so"
echo "    ${BUILD_DIR}/vad_example_onnx"
