# 预编译 libfsmn_vad.so

本目录提供已经编译好的 `libfsmn_vad.so`，不想自己编译时可以直接取用。

| 目录 | 架构 | 适用平台 | 编译环境 |
|------|------|----------|----------|
| `linux-x86_64/` | x86_64 | 普通 Linux PC / 服务器 | Ubuntu 20.04, gcc 9.4, glibc 2.31 |
| `linux-aarch64/` | aarch64 | Jetson 系列（Xavier / Orin / Nano）| 交叉编译，gcc 9.4 (aarch64-linux-gnu), glibc 2.31 |

`linux-aarch64` 已在以下设备实测通过（ONNX 后端，xmov.raw 输出一致）：

- 192.168.101.120 — Ubuntu 20.04.6 / glibc 2.31 / 5.10.104-tegra
- 192.168.100.178 — Ubuntu 22.04.5 / glibc 2.35 / 5.15.148-tegra (Orin)

## 运行时依赖

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

## 文件校验

| 文件 | SHA-256 |
|------|---------|
| `linux-x86_64/libfsmn_vad.so` | `b08989767a283e47cd4155dfe5a006a109d9711a37be604af145251435950b81` |
| `linux-aarch64/libfsmn_vad.so` | `836b134a0d395c2da72d9ced4418812fa5f986c320c38785380ff9d29f78a432` |

> 这两个 .so 未 strip，保留符号表，便于定位问题；体积仍只有约 170–190 KB。

## 自己重新编译

```bash
make                 # 本机原生编译
make ARCH=arm64      # x86 上交叉编译 aarch64
```

产物在 `build/libfsmn_vad.so` 或 `build-arm64/libfsmn_vad.so`。
