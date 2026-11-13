cmake_minimum_required(VERSION 3.16)
project(fsmn_vad C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 编译选项
option(ENABLE_ONNX "Enable ONNX Runtime backend" ON)
option(ENABLE_RKNN "Enable RKNN NPU backend" OFF)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_SHARED_LIBS "Build shared library" ON)

# 编译 flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND NOT ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd -mtune=cortex-a76.cortex-a55")
endif()

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/)

# 第三方依赖：kaldi-native-fbank
file(GLOB KNF_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kaldi-native-fbank/csrc/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kaldi-native-fbank/csrc/*.c)
add_library(knf STATIC ${KNF_SOURCES})
target_include_directories(knf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kaldi-native-fbank)

# 核心源文件
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# 移除不需要的后端文件
if (NOT ENABLE_ONNX)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/fsmn_vad_onnx.cpp)
endif()

if (NOT ENABLE_RKNN)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/fsmn_vad_rknn.cpp)
endif()

# 编译库
add_library(fsmn_vad ${SOURCES})
target_link_libraries(fsmn_vad PUBLIC knf)

# ONNX Runtime 依赖查找
#   - 默认只使用 third_party/onnxruntime 中随仓库携带的预编译包，全程离线
#   - 通过 -DONNXRUNTIME_DIR=<dir> 可指定其它 ONNX Runtime 根目录（含 include/ 与 lib/）
#   - 只有显式传 -DONNXRUNTIME_DOWNLOAD=ON 才会联网下载
option(ONNXRUNTIME_DOWNLOAD "Download ONNX Runtime when no local copy is found" OFF)

if (ENABLE_ONNX)
    target_compile_definitions(fsmn_vad PUBLIC ENABLE_ONNX)

    set(ORT_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)

    if (ANDROID)
        if (NOT DEFINED ONNXRUNTIME_DIR)
            if (EXISTS ${ORT_VENDOR_DIR}/android/include/onnxruntime_cxx_api.h)
                set(ONNXRUNTIME_DIR ${ORT_VENDOR_DIR}/android)
            elseif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android/include/onnxruntime_cxx_api.h)
                # 兼容旧版 tools/setup_onnxruntime_android.sh 的目录布局
                set(ONNXRUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android)
            else()
                message(FATAL_ERROR
                    "ONNX Runtime Android not found.\n"
                    "Expected vendored copy: ${ORT_VENDOR_DIR}/android/{include,lib/${ANDROID_ABI}}\n"
                    "Run: bash tools/setup_onnxruntime_android.sh\n"
                    "See: docs/ANDROID.md")
            endif()
        endif()

        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so)
        if (NOT EXISTS ${ONNXRUNTIME_LIB})
            message(FATAL_ERROR
                "ONNX Runtime Android library not found: ${ONNXRUNTIME_LIB}\n"
                "Vendored dependencies live in third_party/onnxruntime/android/lib/<abi>/.\n"
                "Supported ABIs in this repo: arm64-v8a, armeabi-v7a.\n"
                "Run: bash tools/setup_onnxruntime_android.sh\n"
                "See: docs/ANDROID.md")
        endif()
        if (NOT EXISTS ${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h)
            message(FATAL_ERROR
                "ONNX Runtime headers not found in ${ONNXRUNTIME_DIR}/include\n"
                "Run: bash tools/setup_onnxruntime_android.sh\n"
                "See: docs/ANDROID.md")
        endif()
        message(STATUS "Using ONNX Runtime (Android/${ANDROID_ABI}) from ${ONNXRUNTIME_DIR}")
        add_library(onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRUNTIME_LIB})
        target_include_directories(fsmn_vad PUBLIC ${ONNXRUNTIME_DIR}/include)
        target_link_libraries(fsmn_vad PUBLIC onnxruntime log)
    else()
        # 按目标架构选择 third_party/onnxruntime 下的子目录
        if (NOT DEFINED ONNXRUNTIME_DIR)
            if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
                set(ORT_PLATFORM linux-aarch64)
            elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
                set(ORT_PLATFORM linux-x86_64)
            else()
                set(ORT_PLATFORM "")
            endif()

            if (ORT_PLATFORM AND EXISTS ${ORT_VENDOR_DIR}/${ORT_PLATFORM}/include/onnxruntime_cxx_api.h)
                set(ONNXRUNTIME_DIR ${ORT_VENDOR_DIR}/${ORT_PLATFORM})
            elseif (EXISTS ${ORT_VENDOR_DIR}/include/onnxruntime_cxx_api.h)
                # 兼容旧的扁平布局 third_party/onnxruntime/{include,lib}
                set(ONNXRUNTIME_DIR ${ORT_VENDOR_DIR})
            endif()
        endif()

        if (NOT DEFINED ONNXRUNTIME_DIR)
            if (ONNXRUNTIME_DOWNLOAD)
                if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
                    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz")
                else()
                    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz")
                endif()
                message(STATUS "Downloading ONNX Runtime from ${ONNXRUNTIME_URL}")
                include(FetchContent)
                FetchContent_Declare(
                    onnxruntime
                    URL ${ONNXRUNTIME_URL}
                    DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/downloads
                )
                FetchContent_MakeAvailable(onnxruntime)
                set(ONNXRUNTIME_DIR ${onnxruntime_SOURCE_DIR})
            else()
                message(FATAL_ERROR
                    "ONNX Runtime not found for ${CMAKE_SYSTEM_PROCESSOR}.\n"
                    "Expected vendored copy: ${ORT_VENDOR_DIR}/${ORT_PLATFORM}/{include,lib}\n"
                    "Or pass -DONNXRUNTIME_DIR=<dir>.\n"
                    "To allow a network download instead: -DONNXRUNTIME_DOWNLOAD=ON")
            endif()
        endif()

        # 链接带版本号的 libonnxruntime.so.X.Y.Z（其 SONAME 就是这个文件名），
        # 仓库里无需保存无版本软链接。若用户目录里只有无版本 .so，则回退使用它。
        set(_ort_version_hint "")
        if (EXISTS ${ORT_VENDOR_DIR}/VERSION)
            file(STRINGS ${ORT_VENDOR_DIR}/VERSION _ort_version_hint LIMIT_COUNT 1)
        endif()

        if (_ort_version_hint AND EXISTS ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.${_ort_version_hint})
            set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.${_ort_version_hint})
        elseif (EXISTS ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
            set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
        else()
            file(GLOB _ort_versioned_libs ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.*)
            list(SORT _ort_versioned_libs)
            if (_ort_versioned_libs)
                list(GET _ort_versioned_libs -1 ONNXRUNTIME_LIB)
            else()
                message(FATAL_ERROR
                    "ONNX Runtime shared library not found under ${ONNXRUNTIME_DIR}/lib\n"
                    "Expected libonnxruntime.so.<version> (ORT 1.16.3).")
            endif()
        endif()

        get_filename_component(_ort_lib_name ${ONNXRUNTIME_LIB} NAME)
        message(STATUS "Using ONNX Runtime (${CMAKE_SYSTEM_PROCESSOR}) from ${ONNXRUNTIME_DIR}")

        target_include_directories(fsmn_vad PUBLIC ${ONNXRUNTIME_DIR}/include)

        add_library(onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION ${ONNXRUNTIME_LIB}
            IMPORTED_SONAME ${_ort_lib_name})
        target_link_libraries(fsmn_vad PUBLIC onnxruntime)

        # 运行期查找：构建目录和安装目录都让 libonnxruntime 与 libfsmn_vad.so 同目录
        set_target_properties(fsmn_vad PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "$ORIGIN")
        add_custom_command(TARGET fsmn_vad POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_LIB}" "$<TARGET_FILE_DIR:fsmn_vad>/"
            COMMENT "Copying ONNX Runtime runtime library next to libfsmn_vad.so")
        install(FILES ${ONNXRUNTIME_LIB} DESTINATION lib)
    endif()
endif()

if (ENABLE_RKNN)
    target_compile_definitions(fsmn_vad PUBLIC ENABLE_RKNN)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    include(FindRknn)

    add_library(rknnrt SHARED IMPORTED GLOBAL)
    set_target_properties(rknnrt PROPERTIES IMPORTED_LOCATION "${RKNN_LIBRARY}")
    target_include_directories(fsmn_vad PUBLIC ${RKNN_INCLUDE_DIR})
    if (ANDROID)
        target_link_libraries(fsmn_vad PUBLIC rknnrt log)
    else()
        target_link_libraries(fsmn_vad PUBLIC rknnrt)
    endif()
endif()

# 编译示例程序
if (BUILD_EXAMPLES)
    if (ENABLE_ONNX)
        add_executable(vad_example_onnx examples/demo_onnx.cpp)
        target_link_libraries(vad_example_onnx PRIVATE fsmn_vad)
    endif()

    if (ENABLE_RKNN)
        add_executable(vad_example_rknn examples/demo_rknn.cpp)
        target_link_libraries(vad_example_rknn PRIVATE fsmn_vad)
        if (ANDROID)
            set_target_properties(vad_example_rknn fsmn_vad PROPERTIES
                BUILD_RPATH "\$ORIGIN/lib"
                INSTALL_RPATH "\$ORIGIN/lib")
        endif()
    endif()
endif()

# 安装规则
install(TARGETS fsmn_vad knf
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
install(FILES README.md DESTINATION share/doc/fsmn_vad)