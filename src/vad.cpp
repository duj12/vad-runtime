#include "vad.h"
#include "vad_model.h"
#include "fsmn_vad_onnx.h"
#ifdef ENABLE_RKNN
#include "fsmn_vad_rknn.h"
#endif
#include <cstdlib>
#include <cstring>

using namespace vad;

void* vad_init(const char* model_dir, int sample_rate, VadBackendType backend) {
    if (!model_dir) return nullptr;

    VadModel* model = nullptr;

    // 后端选择逻辑
    if (backend == VAD_BACKEND_ONNX) {
        model = new FsmnVadOnnx();
    }
#ifdef ENABLE_RKNN
    else if (backend == VAD_BACKEND_RKNN) {
        model = new FsmnVadRknn();
    }
#endif
    else if (backend == VAD_BACKEND_AUTO) {
        // 自动选择：优先尝试RKNN，失败则用ONNX
#ifdef ENABLE_RKNN
        model = new FsmnVadRknn();
        try {
            model->Init(model_dir, sample_rate);
            return model;
        } catch (...) {
            delete model;
            model = nullptr;
        }
#endif
        // 回退到ONNX
        model = new FsmnVadOnnx();
    }

    if (!model) return nullptr;

    try {
        model->Init(model_dir, sample_rate);
        return model;
    } catch (const std::exception& e) {
        fprintf(stderr, "VAD init exception: %s\n", e.what());
        delete model;
        return nullptr;
    } catch (...) {
        delete model;
        return nullptr;
    }
}

int vad_process(void* handle, const short* pcm_data, int pcm_len, bool is_final,
                int** out_segments, int* out_segments_len) {
    if (!handle || !pcm_data || pcm_len <= 0 || !out_segments || !out_segments_len) {
        return -1;
    }

    VadModel* model = static_cast<VadModel*>(handle);
    try {
        auto segments = model->Process(pcm_data, pcm_len, is_final);
        *out_segments_len = segments.size();
        if (*out_segments_len == 0) {
            *out_segments = nullptr;
            return 0;
        }

        *out_segments = (int*)malloc(*out_segments_len * 2 * sizeof(int));
        if (!*out_segments) {
            *out_segments_len = 0;
            return -2;
        }

        for (int i = 0; i < *out_segments_len; ++i) {
            (*out_segments)[i * 2] = segments[i].first;
            (*out_segments)[i * 2 + 1] = segments[i].second;
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "VAD process exception: %s\n", e.what());
        return -3;
    } catch (...) {
        return -3;
    }
}

int vad_reset(void* handle) {
    if (!handle) return -1;
    VadModel* model = static_cast<VadModel*>(handle);
    try {
        model->Reset();
        return 0;
    } catch (...) {
        return -2;
    }
}

int vad_destroy(void* handle) {
    if (!handle) return -1;
    VadModel* model = static_cast<VadModel*>(handle);
    delete model;
    return 0;
}

VadBackendType vad_get_backend(void* handle) {
    if (!handle) return VAD_BACKEND_AUTO;
    VadModel* model = static_cast<VadModel*>(handle);
    return model->GetBackendType();
}

void vad_free_segments(int* segments) {
    if (segments) {
        free(segments);
    }
}