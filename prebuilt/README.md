# 预编译 libfsmn_vad.so

本目录提供已经编译好的 `libfsmn_vad.so`，不想自己编译时可以直接取用。

| 目录 | 架构 | 适用平台 | 编译环境 |
|------|------|----------|----------|
| `linux-x86_64/` | x86_64 | 普通 Linux PC / 服务器 | Ubuntu 20.04, gcc 9.4, glibc 2.31 |
| `linux-aarch64/` | aarch64 | Jetson 系列（Xavier / Orin / Nano）| 交叉编译，gcc 9.4 (aarch64-linux-gnu), glibc 2.31 |
| `android-arm64-v8a/` | arm64-v8a | Android 真机（RK3588 等） | NDK r26d, clang, c++_shared |

`linux-aarch64` 已在以下设备实测通过（ONNX 后端，xmov.raw 输出一致）：

- 192.168.101.120 — Ubuntu 20.04.6 / glibc 2.31 / 5.10.104-tegra
- 192.168.100.178 — Ubuntu 22.04.5 / glibc 2.35 / 5.15.148-tegra (Orin)

## 运行时依赖

### Linux（x86_64 / aarch64）

`libfsmn_vad.so` 的 RUNPATH 是 `$ORIGIN`，所以只要把对应平台的
`libonnxruntime.so.1.16.3` 放在**同一目录**即可：

```bash
PLATFORM=linux-aarch64     # 或 linux-x86_64
mkdir -p /opt/vad/lib
cp prebuilt/$PLATFORM/libfsmn_vad.so /opt/vad/lib/
cp third_party/onnxruntime/$PLATFORM/lib/libonnxruntime.so.1.16.3 /opt/vad/lib/
```

然后 `LD_LIBRARY_PATH=/opt/vad/lib`（或把该目录加入 ldconfig）即可链接使用。

示例验证：

```bash
cd /opt/vad/lib
ldd ./libfsmn_vad.so | grep onnxruntime
# libonnxruntime.so.1.16.3 => .../libonnxruntime.so.1.16.3
```

### Android（arm64-v8a）

Android 是 bionic 运行时，需要把三个库都推到设备同一目录：

```bash
adb push prebuilt/android-arm64-v8a/libfsmn_vad.so /data/local/tmp/vad/
adb push third_party/onnxruntime/android/lib/arm64-v8a/libonnxruntime.so /data/local/tmp/vad/
# c++_shared 来自 NDK；未使用 NDK 时见 docs/ANDROID.md 的说明
adb push $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so /data/local/tmp/vad/
adb shell "cd /data/local/tmp/vad && LD_LIBRARY_PATH=. ./your_app"
```

> Android 版只做了编译验证（NDK r26d / arm64-v8a），未在真机上运行；
> 逻辑与 linux-aarch64 完全相同，ONNX Runtime 也是同一 AAR 里的库。

## 自己重新编译 Android

```bash
export ANDROID_NDK=$HOME/ndk/android-ndk-r26d
make android
# 或: bash tools/build_android_onnx.sh
```

## 文件校验

| 文件 | SHA-256 |
|------|---------|
| `linux-x86_64/libfsmn_vad.so` | `b08989767a283e47cd4155dfe5a006a109d9711a37be604af145251435950b81` |
| `linux-aarch64/libfsmn_vad.so` | `836b134a0d395c2da72d9ced4418812fa5f986c320c38785380ff9d29f78a432` |
| `android-arm64-v8a/libfsmn_vad.so` | `1b73de944bf06d612d031c71e1eab6dd27dfa44dd34538396e78d5adb466265b` |

> Linux 两个 .so 未 strip，保留符号表，便于定位问题；Android 版已 strip。

## 自己重新编译

```bash
make                 # 本机原生编译
make ARCH=arm64      # x86 上交叉编译 aarch64
```

产物在 `build/libfsmn_vad.so` 或 `build-arm64/libfsmn_vad.so`。
