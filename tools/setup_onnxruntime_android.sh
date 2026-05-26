#!/usr/bin/env bash
# Download and extract ONNX Runtime Android AAR into third_party/onnxruntime-android/
set -euo pipefail

ORT_VERSION="1.16.3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${PROJECT_ROOT}/third_party/onnxruntime-android"
TMP_DIR="${PROJECT_ROOT}/build/.onnxruntime-android-download"

AAR_URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar"

echo "==> ONNX Runtime Android ${ORT_VERSION}"
echo "    Destination: ${DEST_DIR}"

mkdir -p "${TMP_DIR}"
AAR_FILE="${TMP_DIR}/onnxruntime-android-${ORT_VERSION}.aar"

if [[ ! -f "${AAR_FILE}" ]]; then
    echo "==> Downloading ${AAR_URL}"
    if command -v wget >/dev/null 2>&1; then
        wget -q -O "${AAR_FILE}" "${AAR_URL}"
    elif command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "${AAR_FILE}" "${AAR_URL}"
    else
        echo "Error: wget or curl required" >&2
        exit 1
    fi
else
    echo "==> Using cached ${AAR_FILE}"
fi

EXTRACT_DIR="${TMP_DIR}/extracted"
rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
unzip -q -o "${AAR_FILE}" -d "${EXTRACT_DIR}"

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}/include" "${DEST_DIR}/lib"

cp -a "${EXTRACT_DIR}/headers/." "${DEST_DIR}/include/"

for abi_dir in "${EXTRACT_DIR}"/jni/*/; do
    abi="$(basename "${abi_dir}")"
    if [[ -f "${abi_dir}/libonnxruntime.so" ]]; then
        mkdir -p "${DEST_DIR}/lib/${abi}"
        cp "${abi_dir}/libonnxruntime.so" "${DEST_DIR}/lib/${abi}/"
        echo "    Copied libonnxruntime.so for ${abi}"
    fi
done

if [[ ! -f "${DEST_DIR}/include/onnxruntime_cxx_api.h" ]]; then
    echo "Error: headers not found after extract" >&2
    exit 1
fi

if [[ ! -f "${DEST_DIR}/lib/arm64-v8a/libonnxruntime.so" ]]; then
    echo "Warning: arm64-v8a/libonnxruntime.so not found (check AAR contents)" >&2
fi

echo "==> Done. Layout:"
echo "    ${DEST_DIR}/include/"
echo "    ${DEST_DIR}/lib/<abi>/libonnxruntime.so"
