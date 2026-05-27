# Resolves RKNN headers and librknnrt.so for Linux and Android.
# Sets: RKNN_INCLUDE_DIR, RKNN_LIBRARY, RKNN_LIB_DIR
#
# Cache / env (priority):
#   1. RKNN_RKNPU2_ROOT
#   2. RKNN_MODEL_ZOO_ROOT -> .../3rdparty/rknpu2
#   3. RKNN_SDK_DIR (legacy layout: include/ + lib/ or lib/aarch64/)

if(NOT DEFINED RKNN_RKNPU2_ROOT AND DEFINED ENV{RKNN_RKNPU2_ROOT})
    set(RKNN_RKNPU2_ROOT "$ENV{RKNN_RKNPU2_ROOT}")
endif()
if(NOT DEFINED RKNN_MODEL_ZOO_ROOT AND DEFINED ENV{RKNN_MODEL_ZOO_ROOT})
    set(RKNN_MODEL_ZOO_ROOT "$ENV{RKNN_MODEL_ZOO_ROOT}")
endif()

set(_rknn_root "")
if(RKNN_RKNPU2_ROOT)
    set(_rknn_root "${RKNN_RKNPU2_ROOT}")
elseif(RKNN_MODEL_ZOO_ROOT)
    set(_rknn_root "${RKNN_MODEL_ZOO_ROOT}/3rdparty/rknpu2")
elseif(RKNN_SDK_DIR)
    set(_rknn_root "${RKNN_SDK_DIR}")
endif()

if(NOT _rknn_root)
    message(FATAL_ERROR
        "RKNN SDK not found. Set one of:\n"
        "  -DRKNN_RKNPU2_ROOT=/path/to/rknpu2\n"
        "  -DRKNN_MODEL_ZOO_ROOT=/path/to/rknn_model_zoo\n"
        "  -DRKNN_SDK_DIR=/path/to/rknn_sdk (legacy)")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    if(ANDROID_ABI)
        set(_rknn_arch "${ANDROID_ABI}")
    else()
        set(_rknn_arch "${CMAKE_ANDROID_ARCH_ABI}")
    endif()
    set(_rknn_os "Android")
else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_rknn_arch "aarch64")
    else()
        set(_rknn_arch "armhf")
    endif()
    set(_rknn_os "Linux")
endif()

# Legacy RKNN_SDK_DIR: headers in include/, lib under lib/ or lib/<arch>/
if(RKNN_SDK_DIR AND NOT RKNN_RKNPU2_ROOT AND NOT RKNN_MODEL_ZOO_ROOT)
    set(RKNN_INCLUDE_DIR "${_rknn_root}/include")
    if(EXISTS "${_rknn_root}/lib/${_rknn_arch}/librknnrt.so")
        set(RKNN_LIBRARY "${_rknn_root}/lib/${_rknn_arch}/librknnrt.so")
    elseif(EXISTS "${_rknn_root}/lib/librknnrt.so")
        set(RKNN_LIBRARY "${_rknn_root}/lib/librknnrt.so")
    elseif(EXISTS "${_rknn_root}/lib/librknn_api.so")
        set(RKNN_LIBRARY "${_rknn_root}/lib/librknn_api.so")
    else()
        set(RKNN_LIBRARY "")
    endif()
else()
    set(RKNN_INCLUDE_DIR "${_rknn_root}/include")
    set(RKNN_LIBRARY "${_rknn_root}/${_rknn_os}/${_rknn_arch}/librknnrt.so")

    if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
        set(_rknn_rt_android "${_rknn_root}/runtime/Android/librknn_api")
        set(_rknn_rt_linux "${_rknn_root}/runtime/Linux/librknn_api")
        if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND EXISTS "${_rknn_rt_android}/include/rknn_api.h")
            set(RKNN_INCLUDE_DIR "${_rknn_rt_android}/include")
            set(RKNN_LIBRARY "${_rknn_rt_android}/${_rknn_arch}/librknnrt.so")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${_rknn_rt_linux}/include/rknn_api.h")
            set(RKNN_INCLUDE_DIR "${_rknn_rt_linux}/include")
            set(RKNN_LIBRARY "${_rknn_rt_linux}/${_rknn_arch}/librknnrt.so")
        endif()
    endif()
endif()

if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
    message(FATAL_ERROR
        "RKNN headers not found under ${RKNN_INCLUDE_DIR}.\n"
        "Set RKNN_RKNPU2_ROOT, RKNN_MODEL_ZOO_ROOT, or RKNN_SDK_DIR.")
endif()

if(NOT RKNN_LIBRARY OR NOT EXISTS "${RKNN_LIBRARY}")
    message(FATAL_ERROR
        "librknnrt.so not found for ${_rknn_os}/${_rknn_arch} under ${_rknn_root}.\n"
        "Expected ${_rknn_root}/${_rknn_os}/${_rknn_arch}/librknnrt.so (rknn_model_zoo) "
        "or runtime/${_rknn_os}/librknn_api/<abi>/librknnrt.so (rknn-toolkit2).")
endif()

get_filename_component(RKNN_LIB_DIR "${RKNN_LIBRARY}" DIRECTORY)
message(STATUS "RKNN include: ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN library: ${RKNN_LIBRARY}")
