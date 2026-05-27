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

# 添加编译宏
if (ENABLE_ONNX)
    target_compile_definitions(fsmn_vad PUBLIC ENABLE_ONNX)
    if (ANDROID)
        if (NOT DEFINED ONNXRUNTIME_DIR)
            set(ONNXRUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android)
        endif()
        set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so)
        if (NOT EXISTS ${ONNXRUNTIME_LIB})
            message(FATAL_ERROR
                "ONNX Runtime Android library not found: ${ONNXRUNTIME_LIB}\n"
                "Run: bash tools/setup_onnxruntime_android.sh\n"
                "See: docs/ANDROID.md")
        endif()
        if (NOT EXISTS ${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h)
            message(FATAL_ERROR
                "ONNX Runtime headers not found in ${ONNXRUNTIME_DIR}/include\n"
                "Run: bash tools/setup_onnxruntime_android.sh\n"
                "See: docs/ANDROID.md")
        endif()
        message(STATUS "Using ONNX Runtime Android from ${ONNXRUNTIME_DIR} (${ANDROID_ABI})")
        add_library(onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRUNTIME_LIB})
        target_include_directories(fsmn_vad PUBLIC ${ONNXRUNTIME_DIR}/include)
        target_link_libraries(fsmn_vad PUBLIC onnxruntime log)
    else()
        # 查找ONNX Runtime，优先使用项目third_party下的
        if (NOT DEFINED ONNXRUNTIME_DIR)
            set(LOCAL_ONNX_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)
            if (EXISTS ${LOCAL_ONNX_RUNTIME_DIR} AND EXISTS ${LOCAL_ONNX_RUNTIME_DIR}/include/onnxruntime_cxx_api.h)
                set(ONNXRUNTIME_DIR ${LOCAL_ONNX_RUNTIME_DIR})
                message(STATUS "Using local ONNX Runtime from ${ONNXRUNTIME_DIR}")
            else()
                if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
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
            endif()
        endif()
        include_directories(${ONNXRUNTIME_DIR}/include)
        target_link_directories(fsmn_vad PUBLIC ${ONNXRUNTIME_DIR}/lib)
        target_link_libraries(fsmn_vad PUBLIC onnxruntime)
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