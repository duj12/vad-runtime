#!/usr/bin/env bash
# Push android_vad_rknn install bundle to RK3588 Android device and run demo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_ARCH="${TARGET_ARCH:-arm64-v8a}"
INSTALL_DIR="${INSTALL_DIR:-${PROJECT_ROOT}/install/rk3588_android_${TARGET_ARCH}/android_vad_rknn}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/android_vad_rknn}"
BUILD_BIN="${PROJECT_ROOT}/build-android-rknn/vad_example_rknn"
INSTALL_BIN="${INSTALL_DIR}/vad_example_rknn"
PCM_NAME="xmov.raw"

die() { echo "[deploy] ERROR: $*" >&2; exit 1; }
ok() { echo "[deploy] $*"; }

require_file() {
    [[ -f "$1" ]] || die "missing in install bundle: $1 (run bash tools/build_android_rknn.sh)"
}

if [[ ! -d "${INSTALL_DIR}" ]]; then
    echo "==> install bundle missing, running build_android_rknn.sh"
    bash "${SCRIPT_DIR}/build_android_rknn.sh"
fi

ok "checking install bundle: ${INSTALL_DIR}"
require_file "${INSTALL_BIN}"
require_file "${INSTALL_DIR}/lib/libfsmn_vad.so"
require_file "${INSTALL_DIR}/lib/librknnrt.so"
require_file "${INSTALL_DIR}/model/am.mvn"
require_file "${INSTALL_DIR}/model/config.yaml"

if [[ ! -f "${INSTALL_DIR}/model/model.rknn" ]]; then
    die "missing model/model.rknn (run: python3 tools/convert_onnx_to_rknn.py ...)"
fi

if [[ -f "${BUILD_BIN}" && -f "${INSTALL_BIN}" ]]; then
    if ! cmp -s "${BUILD_BIN}" "${INSTALL_BIN}"; then
        if [[ "${SYNC_BINARY_FROM_BUILD:-0}" == "1" ]]; then
            cp -f "${BUILD_BIN}" "${INSTALL_BIN}"
            cp -f "${PROJECT_ROOT}/build-android-rknn/libfsmn_vad.so" "${INSTALL_DIR}/lib/libfsmn_vad.so"
            ok "synced binaries from build-android-rknn"
        else
            echo "[deploy] WARN: install binary differs from ${BUILD_BIN}" >&2
            echo "[deploy]       re-run build_android_rknn.sh or: SYNC_BINARY_FROM_BUILD=1 $0" >&2
        fi
    fi
fi

command -v adb >/dev/null 2>&1 || die "adb not in PATH"
adb get-state >/dev/null 2>&1 || die "no adb device (adb devices)"

if [[ "${SKIP_ADB_ROOT:-0}" != "1" ]]; then
    adb root 2>/dev/null || true
    adb remount 2>/dev/null || true
fi

ok "pushing to ${DEVICE_ROOT}"
adb shell "rm -rf ${DEVICE_ROOT} && mkdir -p ${DEVICE_ROOT}"
adb push "${INSTALL_DIR}/." "${DEVICE_ROOT}/"

device_check_file() {
    adb shell "test -f ${DEVICE_ROOT}/$1" 2>/dev/null || die "on device missing file: ${DEVICE_ROOT}/$1"
}

ok "verifying on device"
device_check_file "vad_example_rknn"
device_check_file "lib/libfsmn_vad.so"
device_check_file "lib/librknnrt.so"
device_check_file "model/model.rknn"
device_check_file "model/am.mvn"
device_check_file "model/config.yaml"

if adb shell "test -d ${DEVICE_ROOT}/android_vad_rknn" 2>/dev/null; then
    die "nested ${DEVICE_ROOT}/android_vad_rknn/ detected (push with trailing /. )"
fi

ok "deploy OK -> ${DEVICE_ROOT}"
echo ""

if [[ -f "${INSTALL_DIR}/${PCM_NAME}" ]] || adb shell "test -f ${DEVICE_ROOT}/${PCM_NAME}" 2>/dev/null; then
    echo "==> Running vad_example_rknn on device..."
    adb shell "cd ${DEVICE_ROOT} && chmod +x vad_example_rknn && \
        ${VAD_DEBUG:+VAD_DEBUG=${VAD_DEBUG}} LD_LIBRARY_PATH=.:./lib ./vad_example_rknn ./model ./${PCM_NAME}"
else
    echo "Manual run:"
    echo "  adb shell 'cd ${DEVICE_ROOT} && LD_LIBRARY_PATH=.:./lib ./vad_example_rknn ./model ./xmov.raw'"
fi
