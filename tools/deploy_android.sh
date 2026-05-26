#!/usr/bin/env bash
# Push build-android artifacts and run vad_example_onnx on device via adb.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANDROID_NDK="${ANDROID_NDK:-/home/jingsz/other/android-ndk-r19c}"
BUILD_DIR="${PROJECT_ROOT}/build-android"
DEVICE_DIR="/data/local/tmp/vad"
PCM_FILE="${PROJECT_ROOT}/examples/xmov.pcm"

if [[ ! -d "${BUILD_DIR}" ]] || [[ ! -f "${BUILD_DIR}/vad_example_onnx" ]]; then
    echo "==> build-android missing, running build_android.sh"
    bash "${SCRIPT_DIR}/build_android.sh"
fi

ORT_SO="${PROJECT_ROOT}/third_party/onnxruntime-android/lib/arm64-v8a/libonnxruntime.so"
if [[ ! -f "${ORT_SO}" ]]; then
    echo "Error: ${ORT_SO} not found. Run: bash tools/setup_onnxruntime_android.sh" >&2
    exit 1
fi

CXX_SHARED="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [[ ! -f "${CXX_SHARED}" ]]; then
    echo "Error: libc++_shared.so not found under ANDROID_NDK=${ANDROID_NDK}" >&2
    exit 1
fi

if ! adb get-state >/dev/null 2>&1; then
    echo "Error: no adb device (adb devices)" >&2
    exit 1
fi

echo "==> Pushing to ${DEVICE_DIR}"
adb shell mkdir -p "${DEVICE_DIR}/model"
adb push "${BUILD_DIR}/vad_example_onnx" "${DEVICE_DIR}/"
adb push "${BUILD_DIR}/libfsmn_vad.so" "${DEVICE_DIR}/"
adb push "${ORT_SO}" "${DEVICE_DIR}/"
adb push "${CXX_SHARED}" "${DEVICE_DIR}/"
adb push "${PROJECT_ROOT}/model/." "${DEVICE_DIR}/model/"
if [[ -f "${PCM_FILE}" ]]; then
    adb push "${PCM_FILE}" "${DEVICE_DIR}/xmov.pcm"
    PCM_NAME="xmov.pcm"
else
    echo "Warning: ${PCM_FILE} not found, skip PCM push"
    PCM_NAME=""
fi

echo "==> Run (if model_quant.onnx is on device)"
if [[ -f "${PROJECT_ROOT}/model/model_quant.onnx" ]] && [[ -n "${PCM_NAME}" ]]; then
    adb shell "cd ${DEVICE_DIR} && chmod +x vad_example_onnx && LD_LIBRARY_PATH=. ./vad_example_onnx ./model ./${PCM_NAME}"
else
    echo "Skip run: need model/model_quant.onnx and ${PCM_FILE}"
    echo "Manual: adb shell 'cd ${DEVICE_DIR} && LD_LIBRARY_PATH=. ./vad_example_onnx ./model ./xmov.pcm'"
fi
