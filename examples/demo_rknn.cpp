/**
 * FSMN VAD RKNN后端示例 — 流式每帧状态输出
 *
 * 输入：任意长度 16kHz 16bit 单声道 PCM
 * 输出：状态变化时打印（NONE/BEGIN/CONTINUE/END），跳过连续相同的状态
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include "vad.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_dir> <pcm_file>" << std::endl;
        std::cout << "Example: " << argv[0] << " ./models ./test.pcm" << std::endl;
        return -1;
    }

    const char* model_dir = argv[1];
    const char* pcm_file = argv[2];

    void* vad_handle = vad_init(model_dir, 16000, VAD_BACKEND_RKNN);
    if (!vad_handle) {
        std::cerr << "Failed to init VAD" << std::endl;
        return -1;
    }

    std::cout << "VAD initialized, backend: " << vad_get_backend(vad_handle) << std::endl;

    // 读取PCM文件
    std::ifstream is(pcm_file, std::ios::binary);
    if (!is) {
        std::cerr << "Failed to open pcm file: " << pcm_file << std::endl;
        vad_destroy(vad_handle);
        return -1;
    }

    is.seekg(0, std::ios::end);
    size_t file_size = is.tellg();
    is.seekg(0, std::ios::beg);

    size_t pcm_len = file_size / sizeof(short);
    std::vector<short> pcm_data(pcm_len);
    is.read(reinterpret_cast<char*>(pcm_data.data()), file_size);
    is.close();

    int total_dur_ms = (int)(pcm_len * 1000 / 16000);
    std::cout << "Read " << pcm_len << " samples, duration: " << total_dur_ms << "ms" << std::endl;
    std::cout << std::endl;

    // 首帧必须超过WindowDetector(200ms窗口)，后续60ms步长
    const int chunk_size_first = 5760;  // 360ms (36 LFR帧，确保WindowDetector一次确认)
    const int chunk_size = 960;         // 60ms at 16kHz
    int pos = 0;
    int chunk_ms = 0;
    bool in_speech = false;

    int last_status = -1;
    double total_infer_ms = 0.0;

    while (pos < (int)pcm_len) {
        bool is_first = (pos == 0);
        int first_chunk_limit = is_first ? chunk_size_first : chunk_size;
        int current_chunk = std::min(first_chunk_limit, (int)(pcm_len - pos));
        bool is_final = (pos + current_chunk) >= (int)pcm_len;

        auto start = std::chrono::high_resolution_clock::now();

        int* segments = nullptr;
        int segments_len = 0;
        (void)vad_process(vad_handle, pcm_data.data() + pos, current_chunk,
                          is_final, &segments, &segments_len);

        auto end = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_infer_ms += infer_ms;

        bool got_begin = false, got_end = false;
        int begin_ms = 0, end_ms = 0;
        for (int i = 0; i < segments_len; i++) {
            int s = segments[i * 2], e = segments[i * 2 + 1];
            if (s >= 0 && e < 0) { got_begin = true; begin_ms = s; }
            else if (s < 0 && e >= 0) { got_end = true; end_ms = e; }
            else if (s >= 0 && e >= 0) { got_begin = true; begin_ms = s; got_end = true; end_ms = e; }
        }

        int cur_status;
        const char* status_str = "";
        if (got_begin) {
            cur_status = 1;
            status_str = "BEGIN";
            in_speech = true;
        } else if (got_end) {
            cur_status = 3;
            status_str = "END";
            in_speech = false;
        } else if (in_speech) {
            cur_status = 2;
            status_str = "CONTINUE";
        } else {
            cur_status = 0;
            status_str = "NONE";
        }

        std::string seg_str;
        for (int i = 0; i < segments_len; i++) {
            if (!seg_str.empty()) seg_str += " ";
            seg_str += "[" + std::to_string(segments[i*2]) + "," + std::to_string(segments[i*2+1]) + "]";
        }

        if (cur_status != last_status && cur_status >= 0) {
            printf("%4dms: %s    %.3f", chunk_ms, status_str, total_infer_ms / (chunk_ms + current_chunk * 1000 / 16000));
            if (!seg_str.empty()) printf("    %s", seg_str.c_str());
            printf("\n");
            last_status = cur_status;
        }

        chunk_ms += current_chunk * 1000 / 16000;
        pos += current_chunk;
        vad_free_segments(segments);
    }

    printf("\n=== Summary ===\n");
    printf("RTF: %.3f\n", total_infer_ms / total_dur_ms);

    vad_destroy(vad_handle);
    return 0;
}
