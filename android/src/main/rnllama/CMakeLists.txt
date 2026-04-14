cmake_minimum_required(VERSION 3.10)

project(rnllama)

set(CMAKE_CXX_STANDARD 17)
set(RNLLAMA_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cpp)

include_directories(
    ${RNLLAMA_LIB_DIR}
    ${RNLLAMA_LIB_DIR}/common
    ${RNLLAMA_LIB_DIR}/ggml-cpu
    ${RNLLAMA_LIB_DIR}/codec/include
)

file(GLOB GGML_CPU_C_FILES ${RNLLAMA_LIB_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES ${RNLLAMA_LIB_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES ${RNLLAMA_LIB_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB LLAMA_FILES ${RNLLAMA_LIB_DIR}/llama*.cpp)
file(GLOB MODEL_FILES ${RNLLAMA_LIB_DIR}/models/*.cpp)
file(GLOB COMMON_FILES ${RNLLAMA_LIB_DIR}/common/*.cpp)
file(GLOB CODEC_FILES
    ${RNLLAMA_LIB_DIR}/codec/src/*.cpp
    ${RNLLAMA_LIB_DIR}/codec/src/batch/*.cpp
    ${RNLLAMA_LIB_DIR}/codec/src/models/*.cpp
    ${RNLLAMA_LIB_DIR}/codec/src/ops/*.cpp
    ${RNLLAMA_LIB_DIR}/codec/src/runtime/*.cpp
)

set(
    RNLLAMA_SOURCE_FILES
    ${RNLLAMA_LIB_DIR}/ggml.c
    ${RNLLAMA_LIB_DIR}/ggml-alloc.c
    ${RNLLAMA_LIB_DIR}/ggml-backend.cpp
    ${RNLLAMA_LIB_DIR}/ggml-backend-dl.cpp
    ${RNLLAMA_LIB_DIR}/ggml-backend-meta.cpp
    ${RNLLAMA_LIB_DIR}/ggml-backend-reg.cpp
    ${RNLLAMA_LIB_DIR}/ggml-opt.cpp
    ${RNLLAMA_LIB_DIR}/ggml-threading.cpp
    ${RNLLAMA_LIB_DIR}/ggml-quants.c
    ${RNLLAMA_LIB_DIR}/gguf.cpp

    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}

    ${LLAMA_FILES}
    ${MODEL_FILES}
    ${RNLLAMA_LIB_DIR}/unicode-data.cpp
    ${RNLLAMA_LIB_DIR}/unicode.cpp

    ${COMMON_FILES}

    ${CODEC_FILES}

    ${RNLLAMA_LIB_DIR}/rn-llama.cpp
    ${RNLLAMA_LIB_DIR}/rn-completion.cpp
    ${RNLLAMA_LIB_DIR}/rn-tts.cpp
)

find_library(LOG_LIB log)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(RNLLAMA_STANDALONE_BUILD ON)
    message(STATUS "Building rnllama as standalone (for prebuilt distribution)")
else()
    set(RNLLAMA_STANDALONE_BUILD OFF)
    message(STATUS "Building rnllama as subdirectory (will be linked to JNI wrapper)")
endif()

function(build_rnllama_library target_name arch cpu_flags)
    set(ENABLE_OPENCL OFF)
    if (${target_name} MATCHES ".*_opencl$")
        set(ENABLE_OPENCL ON)
    endif ()

    if (NOT ${arch} STREQUAL "generic")
        set(SOURCE_FILES_ARCH
            ${RNLLAMA_LIB_DIR}/ggml-cpu/arch/${arch}/quants.c
            ${RNLLAMA_LIB_DIR}/ggml-cpu/arch/${arch}/repack.cpp
        )
    endif ()

    add_library(
        ${target_name}
        SHARED
        ${RNLLAMA_SOURCE_FILES}
        ${SOURCE_FILES_ARCH}
    )

    target_link_libraries(${target_name} PRIVATE ${LOG_LIB} android)

    if (${arch} STREQUAL "generic")
        target_compile_options(${target_name} PRIVATE -DLM_GGML_CPU_GENERIC)
    endif ()

    target_compile_options(${target_name} PRIVATE -DLM_GGML_USE_CPU -DLM_GGML_USE_CPU_REPACK -pthread ${cpu_flags} -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE)

    if (CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNLLAMA_ANDROID_ENABLE_LOGGING)
    endif ()

    if (ENABLE_OPENCL)
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/OpenCL-Headers)

        set(GGML_OPENCL_KERNELS
            add
            add_id
            argsort
            tri
            fill
            clamp
            cpy
            cvt
            diag_mask_inf
            diag
            div
            gelu
            gemv_noshuffle_general
            gemv_noshuffle
            get_rows
            glu
            group_norm
            solve_tri
            im2col_f32
            im2col_f16
            mean
            mul_mat_Ab_Bi_8x4
            mul_mv_f16_f16
            mul_mv_f16_f32_1row
            mul_mv_f16_f32_l4
            mul_mv_f16_f32
            mul_mv_f32_f32
            mul_mv_q4_0_f32
            mul_mv_q4_0_f32_v
            mul_mv_q4_0_f32_8x_flat
            mul_mv_q4_0_f32_1d_8x_flat
            mul_mv_q4_0_f32_1d_16x_flat
            mul_mv_q4_1_f32
            mul_mv_q4_1_f32_flat
            mul_mv_q4_k_f32
            mul_mv_q4_k_f32_flat
            mul_mv_q5_k_f32
            mul_mv_q5_k_f32_flat
            mul_mv_q6_k_f32
            mul_mv_q6_k_f32_flat
            mul_mv_q8_0_f32
            mul_mv_q8_0_f32_flat
            mul_mv_mxfp4_f32
            mul_mv_mxfp4_f32_flat
            mul_mv_id_q4_0_f32_8x_flat
            mul_mv_id_q8_0_f32
            mul_mv_id_q8_0_f32_flat
            mul_mv_id_mxfp4_f32
            mul_mv_id_mxfp4_f32_flat
            gemm_moe_mxfp4_f32
            gemv_moe_mxfp4_f32
            mul_mm_f32_f32_l4_lm
            mul_mm_f16_f32_l4_lm
            mul_mm_q4_0_f32_l4_lm
            mul_mm_q4_1_f32_l4_lm
            mul_mm_q8_0_f32_l4_lm
            mul_mm_q4_k_f32_l4_lm
            mul_mm_q5_k_f32_l4_lm
            mul_mm_q6_k_f32_l4_lm
            mul_mm_q8_0_f32_8x4
            gemv_noshuffle_q4_1_f32
            gemm_noshuffle_q4_1_f32
            gemv_noshuffle_general_q8_0_f32
            gemv_noshuffle_q4_k_f32
            gemm_noshuffle_q4_k_f32
            gemv_noshuffle_q6_k_f32
            gemv_noshuffle_q6_k_f32
            mul
            neg
            norm
            relu
            l2_norm
            rms_norm
            rope
            scale
            set_rows
            sigmoid
            silu
            softmax_4_f32
            softmax_4_f16
            softmax_f32
            softmax_f16
            sqr
            sqrt
            ssm_conv
            sub
            sum_rows
            cumsum
            transpose
            concat
            tsembd
            upscale
            tanh
            exp
            expm1
            softplus
            pad
            repeat
            mul_mat_f16_f32
            mul_mm_f16_f32_kq_kqv
            conv2d
            conv2d_f16_f32
            flash_attn_f32_f16
            flash_attn_f16
            flash_attn_f32
        )
        set(GGML_OPENCL_KERNEL_DIR ${RNLLAMA_LIB_DIR}/ggml-opencl/kernels)
        set(GGML_OPENCL_KERNEL_HEADERS "")

        foreach(kernel ${GGML_OPENCL_KERNELS})
            set(header_file ${GGML_OPENCL_KERNEL_DIR}/${kernel}.cl.h)
            list(APPEND GGML_OPENCL_KERNEL_HEADERS ${header_file})
        endforeach()

        set(OPENCL_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../bin/${ANDROID_ABI})
        set(OPENCL_STUB     ${OPENCL_STUB_DIR}/libOpenCL.so)

        if (EXISTS ${OPENCL_STUB})
            target_link_directories(${target_name} PRIVATE ${OPENCL_STUB_DIR})
            target_link_libraries(${target_name} PRIVATE OpenCL)
        else()
            message(WARNING
                "OpenCL library not found. Please build with ./scripts/build-opencl.sh")
        endif()

        target_include_directories(${target_name} PRIVATE ${GGML_OPENCL_KERNEL_DIR})
        target_sources(${target_name} PRIVATE
            ${RNLLAMA_LIB_DIR}/ggml-opencl/ggml-opencl.cpp
            ${GGML_OPENCL_KERNEL_HEADERS}
        )
        target_compile_options(${target_name} PRIVATE
            -DLM_GGML_USE_OPENCL
            -DLM_GGML_OPENCL_USE_ADRENO_KERNELS
            -DLM_GGML_OPENCL_EMBED_KERNELS
            -DLM_GGML_OPENCL_SOA_Q
        )
    endif ()

    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${target_name} PRIVATE -flto)
endfunction()

build_rnllama_library("rnllama" "generic" "")

if (ANDROID_ABI AND ANDROID_ABI STREQUAL "arm64-v8a")
    build_rnllama_library("rnllama_v8" "arm" "-march=armv8-a")
    build_rnllama_library("rnllama_v8_2" "arm" "-march=armv8.2-a")
    build_rnllama_library("rnllama_v8_2_dotprod" "arm" "-march=armv8.2-a+dotprod")
    build_rnllama_library("rnllama_v8_2_i8mm" "arm" "-march=armv8.2-a+i8mm")
    build_rnllama_library("rnllama_v8_2_dotprod_i8mm" "arm" "-march=armv8.2-a+dotprod+i8mm")
    build_rnllama_library("rnllama_v8_2_dotprod_i8mm_opencl" "arm" "-march=armv8.2-a+dotprod+i8mm")

elseif (ANDROID_ABI AND ANDROID_ABI STREQUAL "x86_64")
    build_rnllama_library("rnllama_x86_64" "x86" "-march=x86-64;-mtune=generic;-msse4.2;-mpopcnt")

endif ()
