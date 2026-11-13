# ONNX Runtime 1.16.3（随仓库携带）

本目录存放 **预编译的 ONNX Runtime**，目的是让 `vad-runtime` 在无外网/慢外网环境
（Jetson 设备、内网构建机）里也能直接构建，不需要每次去 GitHub / Maven 下载。

## 目录布局

```
third_party/onnxruntime/
├── VERSION                     # 1.16.3
├── LICENSE                     # MIT
├── ThirdPartyNotices.txt
├── linux-x86_64/
│   ├── include/                # onnxruntime_cxx_api.h 等
│   └── lib/libonnxruntime.so.1.16.3
├── linux-aarch64/
│   ├── include/
│   └── lib/libonnxruntime.so.1.16.3
└── android/
    ├── include/
    └── lib/
        ├── arm64-v8a/libonnxruntime.so
        └── armeabi-v7a/libonnxruntime.so
```

`libonnxruntime.so` 的 SONAME 是 `libonnxruntime.so.1.16.3`，因此仓库中只保存带版本号的
真实文件，CMake 直接按完整文件名链接，不需要无版本软链接（避免 Windows 检出问题）。

## 来源

| 平台 | 来源 | SHA-256 |
|------|------|---------|
| linux-x86_64 | `onnxruntime-linux-x64-1.16.3.tgz` (Microsoft GitHub Release) | — |
| linux-aarch64 | `onnxruntime-linux-aarch64-1.16.3.tgz`（同上） | `784dbef93b40196aa668d29d78294a81c0d21361d36530b817bb24d87e8730e8` |
| android | `onnxruntime-android-1.16.3.aar`（Maven Central） | `75b2927c51955aed63f29644a8c3263a90293e612225a1e209d992565f8f4394` |

- Linux 版：<https://github.com/microsoft/onnxruntime/releases/tag/v1.16.3>
- Android 版：<https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.16.3/>
- 许可证：MIT，见本目录 `LICENSE` 与 `ThirdPartyNotices.txt`

## ABI 兼容性

- `linux-aarch64` 由 Microsoft 官方构建，要求 **glibc ≥ 2.28**（实际依赖到 2.29），
  已在以下设备验证：
  - Ubuntu 20.04 / glibc 2.31 / Jetson (5.10.104-tegra)
  - Ubuntu 22.04 / glibc 2.35 / Jetson Orin (5.15.148-tegra)
- `linux-x86_64` 同样来自官方发布包，要求 glibc ≥ 2.28。
- Android 库是 **bionic** 库，不能在 glibc Linux 上使用；反之亦然。

## 更新依赖版本

1. 下载对应版本的 tgz / aar；
2. 按上面的目录布局覆盖 `include/` 与 `lib/`；
3. 更新本文件与 `VERSION`；
4. 若链接文件名变化（例如升级到 1.17），CMake 会自动识别 `lib/libonnxruntime.so.*`，
   无需改 CMakeLists。

Android 依赖可用脚本重新生成：

```bash
bash tools/setup_onnxruntime_android.sh   # 需要联网，写入 third_party/onnxruntime/android
```
