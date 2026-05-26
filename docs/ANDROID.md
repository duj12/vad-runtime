# Android 开发板部署指南（ONNX CPU）

在瑞芯微等 **Android ARM64** 开发板上运行 FSMN VAD，推理后端为 **ONNX Runtime CPU**（与 Linux `vad_example_onnx` 一致）。

完整流程：**环境准备 → ONNX Runtime Android → 模型准备 → NDK 交叉编译 → adb 推送运行 → 调试**。

## 1. 环境准备

| 组件 | 说明 |
|------|------|
| NDK | `export ANDROID_NDK=/home/jingsz/other/android-ndk-r19c`（也可用环境变量覆盖） |
| CMake | 3.16+ |
| adb | Android platform-tools，`adb devices` 能看到设备 |
| ABI | `arm64-v8a`（RK3588 / RK3568 等 64 位板） |
| API | `android-21`（NDK r19c 对应 `aarch64-linux-android21-clang++`） |

```bash
export ANDROID_NDK=/home/jingsz/other/android-ndk-r19c
adb devices
adb shell getprop ro.product.cpu.abi   # 期望 arm64-v8a
```

## 2. 获取 ONNX Runtime（Android）

**不能**使用 `onnxruntime-linux-aarch64-*.tgz`：那是 **glibc Linux** 库，在 Android **bionic** 上会加载失败或崩溃。

### 方式 A：一键脚本（推荐）

在项目根目录执行：

```bash
bash tools/setup_onnxruntime_android.sh
```

生成目录：

```
third_party/onnxruntime-android/
├── include/                    # onnxruntime_cxx_api.h 等
└── lib/
    └── arm64-v8a/
        └── libonnxruntime.so
```

### 方式 B：手动下载 AAR

```bash
# 在仓库根目录执行（下面命令均假设已 cd 到 vad-runtime 根目录）
ORT_VER=1.16.3
wget https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VER}/onnxruntime-android-${ORT_VER}.aar
mkdir -p build/.ort-aar && unzip -q onnxruntime-android-${ORT_VER}.aar -d build/.ort-aar

mkdir -p third_party/onnxruntime-android/include third_party/onnxruntime-android/lib/arm64-v8a
cp -a build/.ort-aar/headers/. third_party/onnxruntime-android/include/
cp build/.ort-aar/jni/arm64-v8a/libonnxruntime.so third_party/onnxruntime-android/lib/arm64-v8a/
```

与桌面版 `third_party/onnxruntime/`（Linux x64/aarch64）**分开存放**，避免混用。

## 3. 模型文件准备

运行需要模型目录下三个文件：

| 文件 | 说明 |
|------|------|
| `model_quant.onnx` | ONNX 量化模型（5 输入 5 输出） |
| `am.mvn` | CMVN 参数 |
| `config.yaml` | VAD 配置 |

仓库 `model/` 目录已包含 `am.mvn`、`config.yaml`，**需自行准备** `model_quant.onnx`（可从 [FunASR](https://github.com/modelscope/FunASR) FSMN VAD 导出或拷贝到 `model/`）。

```bash
ls model/
# 应看到: am.mvn  config.yaml  model_quant.onnx
```

### 测试音频

- 16 kHz、16 bit、单声道 raw PCM（无 WAV 头）
- 可用仓库 `examples/xmov.raw`，或由 WAV 转换：

```bash
ffmpeg -y -i examples/xmov.wav -f s16le -acodec pcm_s16le -ac 1 -ar 16000 examples/xmov.raw
```

### 附录：ONNX → RKNN 转换（非本次 Android 主路径）

若日后在板端使用 RKNN NPU，可在 **x86 主机** 安装 [RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) 后转换：

```bash
python3 tools/convert_onnx_to_rknn.py model/model_quant.onnx model/model.rknn --target rk3588
```

当前 Android 部署仍使用 `model_quant.onnx` + ONNX 后端。

## 4. 交叉编译

### 方式 A：一键脚本（推荐）

在仓库根目录执行。若 `build-android/` 已存在且已配置过 CMake，**只执行 `make`**，不会删除目录、不会重新 `cmake`：

```bash
export ANDROID_NDK=/home/jingsz/other/android-ndk-r19c
cd ~/workspace/vad-runtime
bash tools/build_android.sh
```

需要完全重新配置时：`rm -rf build-android` 后再运行上述脚本。

脚本会自动检查并下载 ONNX Runtime Android（调用 `setup_onnxruntime_android.sh`）。可选环境变量：`ANDROID_ABI`、`ANDROID_PLATFORM`、`JOBS`。

### 方式 B：手动 cmake

```bash
export ANDROID_NDK=/home/jingsz/other/android-ndk-r19c
cd ~/workspace/vad-runtime
bash tools/setup_onnxruntime_android.sh

mkdir -p build-android && cd build-android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21 \
  -DANDROID_STL=c++_shared \
  -DENABLE_ONNX=ON \
  -DENABLE_RKNN=OFF
make -j$(nproc)
```

> 一般无需 `-DONNXRUNTIME_DIR`，默认使用 `third_party/onnxruntime-android/`。勿使用占位路径 `/path/to/vad-runtime`。

产物（在 `build-android/`）：

- `libfsmn_vad.so` — VAD 动态库
- `vad_example_onnx` — 示例可执行文件

验证为 Android ELF：

```bash
file vad_example_onnx
# 期望: ELF 64-bit LSB shared object, ARM aarch64, ... Android
```

## 5. adb 部署与运行

### 5.1 推送文件

在开发板上使用可写目录 `/data/local/tmp/vad/`（一般无需 root）：

```bash
export ANDROID_NDK=/home/jingsz/other/android-ndk-r19c
VAD_ROOT=~/workspace/vad-runtime   # 改成你的仓库实际路径
BUILD=$VAD_ROOT/build-android

adb shell mkdir -p /data/local/tmp/vad/model

adb push $BUILD/vad_example_onnx /data/local/tmp/vad/
adb push $BUILD/libfsmn_vad.so /data/local/tmp/vad/
adb push $VAD_ROOT/third_party/onnxruntime-android/lib/arm64-v8a/libonnxruntime.so /data/local/tmp/vad/
adb push $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so /data/local/tmp/vad/

adb push $VAD_ROOT/model/ /data/local/tmp/vad/model/
adb push $VAD_ROOT/examples/xmov.raw /data/local/tmp/vad/
```

`libc++_shared.so` 必须与可执行文件同目录或位于 `LD_LIBRARY_PATH` 中（使用 `c++_shared` STL 时必需）。

### 5.2 运行

```bash
adb shell 'cd /data/local/tmp/vad && chmod +x vad_example_onnx && \
  LD_LIBRARY_PATH=. ./vad_example_onnx ./model ./xmov.raw'
```

期望输出类似：

```
VAD initialized, backend: 1
Read 83968 samples, duration: 5248ms

   0ms: BEGIN    ...
4800ms: END    ...

=== Summary ===
RTF: ...
```

### 5.3 一键部署脚本（可选）

```bash
export ANDROID_NDK=/home/jingsz/other/android-ndk-r19c
cd ~/workspace/vad-runtime && bash tools/deploy_android.sh
```

## 6. 调试

| 现象 | 排查 |
|------|------|
| `cannot locate symbol` / `dlopen failed` | 是否推送 `libonnxruntime.so`、`libc++_shared.so`；`LD_LIBRARY_PATH=.` |
| `Illegal instruction` | 是否误用 Linux aarch64 的 ORT |
| `Failed to init VAD` | 模型路径、`model_quant.onnx` 是否存在 |
| `Permission denied` | `chmod +x vad_example_onnx`；路径用 `/data/local/tmp` |
| 崩溃无输出 | `adb logcat \| grep -iE 'vad\|onnx\|fatal'` |

常用命令：

```bash
# 确认 ABI
adb shell getprop ro.product.cpu.abi

# 确认 ELF
adb shell file /data/local/tmp/vad/vad_example_onnx

# 查看依赖（在主机上对 push 前的 .so）
readelf -d third_party/onnxruntime-android/lib/arm64-v8a/libonnxruntime.so | grep NEEDED

# 拉取目录排查
adb pull /data/local/tmp/vad ./vad-device-copy
```

## 7. 常见问题

1. **`ONNX Runtime Android library not found: /path/to/vad-runtime/...`**  
   把文档占位路径 `/path/to/vad-runtime` 原样写进了 `cmake` 会导致该错误。解决：在仓库根目录执行 `bash tools/setup_onnxruntime_android.sh`，重新配置时不要传错误的 `ONNXRUNTIME_DIR`（或改为真实绝对路径）。

2. **误用 Linux ORT**  
   `onnxruntime-linux-aarch64-*.tgz` 仅适用于 RK3588 **Linux** 系统，不能用于 Android。

3. **缺少 `libc++_shared.so`**  
   CMake 使用 `-DANDROID_STL=c++_shared` 时，必须把 NDK 中的 `libc++_shared.so` 推到设备。

4. **模型未就绪**  
   `vad_init` 返回 NULL 时检查 `model/` 下三文件是否齐全。

5. **RKNN**  
   当前 CMake **不支持** Android 上链接 RKNN；请使用 ONNX 后端（`VAD_BACKEND_ONNX`）。

## 参考

- [ONNX Runtime Android 安装说明](https://onnxruntime.ai/docs/install/#install-on-android)
- 项目 [README.md](../README.md) — API 与流式输出说明
