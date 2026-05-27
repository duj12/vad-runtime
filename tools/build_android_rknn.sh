#!/usr/bin/env bash
# Cross-compile vad-runtime for RK3588 Android (arm64-v8a, RKNN NPU).
# Produces install/rk3588_android_<abi>/android_vad_rknn/ bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-android-rknn}"

ANDROID_NDK="${ANDROID_NDK:-${ANDROID_NDK_PATH:-/home/jingsz/other/android-ndk-r19c}}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-28}"
JOBS="${JOBS:-$(nproc)}"

TARGET_ARCH="${ANDROID_ABI}"
INSTALL_DIR="${INSTALL_DIR:-${PROJECT_ROOT}/install/rk3588_android_${TARGET_ARCH}/android_vad_rknn}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/android_vad_rknn}"

if [[ -z "${RKNN_RKNPU2_ROOT:-}" ]]; then
    if [[ -z "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
        if [[ -d "${HOME}/workspace/rknn_model_zoo" ]]; then
            export RKNN_MODEL_ZOO_ROOT="${HOME}/workspace/rknn_model_zoo"
        else
            echo "Error: set RKNN_RKNPU2_ROOT or RKNN_MODEL_ZOO_ROOT" >&2
            exit 1
        fi
    fi
fi

TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
    echo "Error: NDK toolchain not found: ${TOOLCHAIN_FILE}" >&2
    echo "Set ANDROID_NDK or ANDROID_NDK_PATH (current: ${ANDROID_NDK})" >&2
    exit 1
fi

CMAKE_EXTRA=()
if [[ -n "${RKNN_RKNPU2_ROOT:-}" ]]; then
    CMAKE_EXTRA+=(-DRKNN_RKNPU2_ROOT="${RKNN_RKNPU2_ROOT}")
fi
if [[ -n "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
    CMAKE_EXTRA+=(-DRKNN_MODEL_ZOO_ROOT="${RKNN_MODEL_ZOO_ROOT}")
fi

echo "ANDROID_NDK=${ANDROID_NDK}"
echo "RKNN_MODEL_ZOO_ROOT=${RKNN_MODEL_ZOO_ROOT:-}"
echo "RKNN_RKNPU2_ROOT=${RKNN_RKNPU2_ROOT:-}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_DIR=${INSTALL_DIR}"

if [[ -d "${BUILD_DIR}" && -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "==> ${BUILD_DIR} exists, skip cmake (rm -rf build-android-rknn to reconfigure)"
    cd "${BUILD_DIR}"
    make -j"${JOBS}"
else
    echo "==> Configure Android RKNN build in ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DANDROID_ABI="${ANDROID_ABI}" \
        -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
        -DANDROID_STL=c++_static \
        -DENABLE_ONNX=OFF \
        -DENABLE_RKNN=ON \
        "${CMAKE_EXTRA[@]}"
    make -j"${JOBS}"
fi

# Resolve librknnrt.so for install bundle
if [[ -n "${RKNN_RKNPU2_ROOT:-}" ]]; then
    _RKNN_BASE="${RKNN_RKNPU2_ROOT}"
elif [[ -n "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
    _RKNN_BASE="${RKNN_MODEL_ZOO_ROOT}/3rdparty/rknpu2"
else
    echo "Error: RKNN SDK root unknown" >&2
    exit 1
fi
if [[ -f "${_RKNN_BASE}/Android/${TARGET_ARCH}/librknnrt.so" ]]; then
    _RKNN_RT="${_RKNN_BASE}/Android/${TARGET_ARCH}/librknnrt.so"
elif [[ -f "${_RKNN_BASE}/runtime/Android/librknn_api/${TARGET_ARCH}/librknnrt.so" ]]; then
    _RKNN_RT="${_RKNN_BASE}/runtime/Android/librknn_api/${TARGET_ARCH}/librknnrt.so"
else
    echo "librknnrt.so not found under ${_RKNN_BASE}" >&2
    exit 1
fi

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/lib" "${INSTALL_DIR}/model"

cp "${BUILD_DIR}/vad_example_rknn" "${INSTALL_DIR}/"
cp "${BUILD_DIR}/libfsmn_vad.so" "${INSTALL_DIR}/lib/"
cp "${_RKNN_RT}" "${INSTALL_DIR}/lib/"

if [[ -d "${PROJECT_ROOT}/model" ]]; then
    cp "${PROJECT_ROOT}/model/am.mvn" "${INSTALL_DIR}/model/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/model/config.yaml" "${INSTALL_DIR}/model/" 2>/dev/null || true
    if [[ -f "${PROJECT_ROOT}/model/model.rknn" ]]; then
        cp "${PROJECT_ROOT}/model/model.rknn" "${INSTALL_DIR}/model/"
    else
        echo "Warning: model/model.rknn not found; run tools/convert_onnx_to_rknn.py before board run" >&2
    fi
fi

_pcm=""
for _c in "${PROJECT_ROOT}/examples/xmov.raw" "${PROJECT_ROOT}/examples/xmov.pcm"; do
    if [[ -f "${_c}" ]]; then
        _pcm="${_c}"
        break
    fi
done
if [[ -n "${_pcm}" ]]; then
    cp "${_pcm}" "${INSTALL_DIR}/xmov.raw"
fi

cat > "${INSTALL_DIR}/README.txt" <<EOF
Android RKNN VAD bundle (RK3588).

Push to device:

  bash tools/deploy_android_rknn.sh

Run on board:

  adb shell 'cd ${DEVICE_ROOT} && export LD_LIBRARY_PATH=.:./lib && \\
    ./vad_example_rknn ./model ./xmov.raw'

Expect: VAD initialized, backend: 2
EOF

echo ""
echo "==> Build complete. Install bundle: ${INSTALL_DIR}"
echo "    vad_example_rknn, libfsmn_vad.so, lib/librknnrt.so"
echo ""
echo "  bash tools/deploy_android_rknn.sh"
