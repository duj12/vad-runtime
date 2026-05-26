/**
 * FSMN VAD RKNN后端示例
 * 输入：16kHz采样率，16bit位深，单声道PCM原始音频
 * 输出：检测到的语音段起止时间(毫秒)
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "vad.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_dir> <pcm_file>" << std::endl;
        std::cout << "Example: " << argv[0] << " ./models ./test.pcm" << std::endl;
        return -1;
    }

    const char* model_dir = argv[1];
    const char* pcm_file = argv[2];

    // 初始化VAD，使用RKNN后端
    void* vad_handle = vad_init(model_dir, 16000, VAD_BACKEND_RKNN);
    if (!vad_handle) {
        std::cerr << "Failed to init VAD with RKNN backend" << std::endl;
        return -1;
    }

    std::cout << "VAD initialized, backend: RKNN NPU" << std::endl;

    // 读取PCM文件
    std::ifstream is(pcm_file, std::ios::binary);
    if (!is) {
        std::cerr << "Failed to open pcm file: " << pcm_file << std::endl;
        vad_destroy(vad_handle);
        return -1;
    }

    // 获取文件大小
    is.seekg(0, std::ios::end);
    size_t file_size = is.tellg();
    is.seekg(0, std::ios::beg);

    size_t pcm_len = file_size / sizeof(short);
    std::vector<short> pcm_data(pcm_len);
    is.read(reinterpret_cast<char*>(pcm_data.data()), file_size);
    is.close();

    std::cout << "Read " << pcm_len << " samples, duration: " << (float)pcm_len / 16000 << "s" << std::endl;

    // 流式处理：每次处理300ms的音频
    const int chunk_size = 16000 * 0.3;  // 300ms = 4800 samples
    int pos = 0;
    int total_segments = 0;

    while (pos < pcm_len) {
        int current_chunk = std::min(chunk_size, (int)(pcm_len - pos));
        bool is_final = (pos + current_chunk) >= pcm_len;

        int* segments = nullptr;
        int segments_len = 0;

        int ret = vad_process(vad_handle, pcm_data.data() + pos, current_chunk, is_final, &segments, &segments_len);
        if (ret != 0) {
            std::cerr << "Failed to process audio, error: " << ret << std::endl;
            break;
        }

        if (segments_len > 0) {
            total_segments += segments_len;
            for (int i = 0; i < segments_len; ++i) {
                int start = segments[i * 2];
                int end = segments[i * 2 + 1];
                std::cout << "Segment: " << start << "ms - " << end << "ms, duration: " << (end - start) << "ms" << std::endl;
            }
        }

        vad_free_segments(segments);
        pos += current_chunk;
    }

    std::cout << "Process completed, total " << total_segments << " segments detected" << std::endl;

    // 释放资源
    vad_destroy(vad_handle);
    return 0;
}