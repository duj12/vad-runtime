# FSMN VAD 便捷构建入口
#
# 依赖（ONNX Runtime）已随仓库放在 third_party/onnxruntime 下，构建过程无需联网。
#
# 常用命令:
#   make                     # 本机原生编译（Linux x86_64 / aarch64 均可）
#   make ARCH=arm64          # 在 x86 上交叉编译 aarch64（需 g++-aarch64-linux-gnu）
#   make android             # Android arm64-v8a（需 Android NDK）
#   make clean               # 删除 build/
#   make distclean           # 删除所有构建目录
#
# 产物:
#   build/libfsmn_vad.so
#   build/vad_example_onnx   # ONNX 后端示例
#
# 直接使用预编译库见 prebuilt/README.md。

BUILD_DIR ?= build
JOBS      ?= $(shell nproc 2>/dev/null || echo 4)
ARCH      ?= native
CMAKE     ?= cmake

ifeq ($(ARCH),native)
  TOOLCHAIN_FLAGS :=
else ifeq ($(ARCH),arm64)
  TOOLCHAIN_FLAGS := -DCMAKE_TOOLCHAIN_FILE=$(CURDIR)/cmake/toolchain-aarch64-linux.cmake
  BUILD_DIR := $(BUILD_DIR)-arm64
else
  $(error ARCH 只支持 native 或 arm64)
endif

.PHONY: all configure build clean distclean android

all: build

configure:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CMAKE) .. $(TOOLCHAIN_FLAGS) \
		-DENABLE_ONNX=ON -DENABLE_RKNN=OFF -DCMAKE_BUILD_TYPE=Release

build: configure
	$(MAKE) -C $(BUILD_DIR) -j$(JOBS)
	@echo
	@echo "==> 产物:"
	@echo "    $(BUILD_DIR)/libfsmn_vad.so"
	@if [ -f $(BUILD_DIR)/vad_example_onnx ]; then \
		echo "    $(BUILD_DIR)/vad_example_onnx"; \
		echo "    运行: cd $(BUILD_DIR) && ./vad_example_onnx ../model ../examples/xmov.raw"; \
	fi

android:
	bash tools/build_android_onnx.sh

clean:
	rm -rf build

distclean: clean
	rm -rf build-arm64 build-android build-android-rknn
