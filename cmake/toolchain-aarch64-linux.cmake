# Linux aarch64 交叉编译工具链（Ubuntu/Debian 上的 gcc-aarch64-linux-gnu）
#
# 用法:
#   cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-aarch64-linux.cmake
#   或: make ARCH=arm64
#
# 需要预装:
#   sudo apt install g++-aarch64-linux-gnu
#
# 注意: 用 Ubuntu 20.04 的 gcc 9.4 + glibc 2.31 交叉编译出的 .so 可在
#       Jetson (Ubuntu 20.04 / 22.04) 上直接运行；用更新的发行版工具链
#       会引入更高的 GLIBC/GLIBCXX 版本要求。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# 只在 sysroot 内查找库/头文件，程序（如 cmake 自身调用的工具）用宿主机的
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
