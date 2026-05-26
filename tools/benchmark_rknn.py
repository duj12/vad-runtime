#!/usr/bin/env python3
"""
RKNN PC Simulator 基准测试脚本
测试 FSMN VAD RKNN 模型的推理速度和正确性

用法:
    python3 benchmark_rknn.py                                # 默认: 测试 speed + 对比 ONNX
    python3 benchmark_rknn.py --no-verify                    # 仅 speed，不对比 ONNX
    python3 benchmark_rknn.py --model ../model/model.rknn    # 指定 RKNN 模型路径
    python3 benchmark_rknn.py --iterations 500               # 更多迭代
"""
import os
import sys
import time
import argparse


def load_and_build(onnx_path, rknn_path):
    """
    加载 ONNX → 构建 RKNN → 初始化 PC 模拟器
    注: PC 模拟器不支持 load_rknn 直接加载已编译模型, 必须从 ONNX 走完整流程
    """
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    print(f"Loading ONNX: {onnx_path}")
    print(f"Target RKNN: {rknn_path}")

    # 配置 (与 convert_onnx_to_rknn.py 一致)
    ret = rknn.config(
        mean_values=None,
        std_values=None,
        target_platform='rk3588',
        optimization_level=3,
        output_optimize=True,
        dynamic_input=[
            [[1, 1, 400], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1]],
            [[1, 6, 400], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1]],
            [[1, 36, 400],[1, 128, 19, 1],[1, 128, 19, 1],[1, 128, 19, 1],[1, 128, 19, 1]],
        ]
    )
    if ret != 0:
        print('rknn.config failed!')
        sys.exit(1)

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('rknn.load_onnx failed!')
        sys.exit(1)

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('rknn.build failed!')
        sys.exit(1)

    ret = rknn.init_runtime(target=None)  # PC simulator
    if ret != 0:
        print('rknn.init_runtime(simulator) failed!')
        sys.exit(1)

    return rknn


def get_onnx_model(onnx_path):
    """加载 ONNX 模型用于输出对比"""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        input_names = [i.name for i in session.get_inputs()]
        output_names = [o.name for o in session.get_outputs()]
        print(f"  ONNX inputs: {input_names}")
        print(f"  ONNX outputs: {output_names}")
        return session, input_names, output_names
    except Exception as e:
        print(f"  ONNX Runtime not available: {e}")
        return None, None, None


def make_inputs(n_frames, feat_dim=400, cache_shape=(1, 128, 19, 1)):
    """构造输入数据: speech + 4 caches (NHWC)"""
    import numpy as np
    speech = np.random.randn(1, n_frames, feat_dim).astype(np.float32)
    cache_nchw = np.zeros((1, 128, 19, 1), dtype=np.float32)
    cache_nhwc = np.transpose(cache_nchw, (0, 2, 3, 1))  # [1,19,1,128]
    caches = [cache_nhwc.copy() for _ in range(4)]
    return speech, caches


def make_inputs_from_feats(feats, in_cache_nhwc_list):
    """从特征矩阵构造输入"""
    import numpy as np
    speech = feats.astype(np.float32)  # [1, N, 400]
    if speech.ndim == 2:
        speech = speech[np.newaxis, :, :]
    caches = [c.copy() for c in in_cache_nhwc_list] if in_cache_nhwc_list else []
    return speech, caches


def run_rknn_inference(rknn, speech, caches):
    """执行 RKNN 推理"""
    inputs = [speech] + caches
    outputs = rknn.inference(inputs=inputs)
    return outputs


def run_onnx_inference(session, input_names, speech, caches_nchw):
    """执行 ONNX 推理"""
    # ONNX 需要 NCHW cache [1,128,19,1]
    ort_inputs = {input_names[0]: speech}
    for i, name in enumerate(input_names[1:]):
        ort_inputs[name] = caches_nchw[i]
    outputs = session.run(None, ort_inputs)
    return outputs


def bench_speed(rknn, n_frames_list, iterations=200):
    """测试推理速度"""
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Speed Benchmark ({iterations} iterations each)")
    print(f"{'='*60}")

    results = {}
    for n_frames in n_frames_list:
        speech, caches = make_inputs(n_frames)
        # warmup
        for _ in range(10):
            run_rknn_inference(rknn, speech, caches)

        # timed runs
        start = time.perf_counter()
        for _ in range(iterations):
            run_rknn_inference(rknn, speech, caches)
        elapsed = time.perf_counter() - start
        avg_ms = elapsed / iterations * 1000

        results[n_frames] = avg_ms
        print(f"  {n_frames:2d} frames: {avg_ms:.3f} ms/infer  "
              f"(total {elapsed:.2f}s for {iterations} iters)")
    return results


def bench_streaming(rknn, iterations=200):
    """测试流式场景: 首帧 36f → 后续 6f"""
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Streaming Simulation ({iterations} iterations, 36f→6f→6f→...→6f)")
    print(f"{'='*60}")

    # 构造随机输入
    speech_36, caches_36 = make_inputs(36)
    speech_6, caches_6 = make_inputs(6)

    # 首帧 36f
    for _ in range(10):
        outputs_36 = run_rknn_inference(rknn, speech_36, caches_36)
    start = time.perf_counter()
    for _ in range(iterations):
        outputs_36 = run_rknn_inference(rknn, speech_36, caches_36)
    elapsed_first = time.perf_counter() - start
    avg_first_ms = elapsed_first / iterations * 1000

    # 更新 caches: RKNN 输出 NCHW [1,128,19,1] → 输入需要 NHWC [1,19,1,128]
    caches_updated = [np.transpose(outputs_36[i + 1], (0, 2, 3, 1)) for i in range(4)]

    # 后续 6f
    for _ in range(10):
        outputs_6 = run_rknn_inference(rknn, speech_6, caches_updated)
    start = time.perf_counter()
    for _ in range(iterations):
        outputs_6 = run_rknn_inference(rknn, speech_6, caches_updated)
    elapsed_follow = time.perf_counter() - start
    avg_follow_ms = elapsed_follow / iterations * 1000

    print(f"  First frame (36f): {avg_first_ms:.3f} ms/infer")
    print(f"  Follow   (6f):     {avg_follow_ms:.3f} ms/infer")

    return {"first_36f": avg_first_ms, "follow_6f": avg_follow_ms}


def verify_with_onnx(rknn, onnx_session, input_names):
    """对比 RKNN 和 ONNX 的输出"""
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Correctness Verification (RKNN vs ONNX)")
    print(f"{'='*60}")

    if onnx_session is None:
        print("  SKIP: ONNX Runtime not available")
        return True

    for n_frames in [6, 36]:
        print(f"\n  --- {n_frames} frames ---")
        speech, caches = make_inputs(n_frames)
        caches_nchw = [np.transpose(c, (0, 3, 1, 2)) for c in caches]  # NHWC→NCHW

        # RKNN inference
        rknn_outputs = run_rknn_inference(rknn, speech, caches)

        # ONNX inference
        onnx_outputs = run_onnx_inference(onnx_session, input_names, speech, caches_nchw)

        # 对比 logits (output 0)
        rknn_logits = rknn_outputs[0]
        onnx_logits = onnx_outputs[0]
        max_diff = np.max(np.abs(rknn_logits - onnx_logits))
        mean_diff = np.mean(np.abs(rknn_logits - onnx_logits))
        print(f"    logits max diff:  {max_diff:.6f}")
        print(f"    logits mean diff: {mean_diff:.6f}")

        # 对比 caches (outputs 1-4)
        for i in range(4):
            # RKNN 和 ONNX 的 cache 输出都是 NCHW [1,128,19,1]
            rknn_cache = rknn_outputs[i + 1]
            onnx_cache = onnx_outputs[i + 1]
            max_diff = np.max(np.abs(rknn_cache - onnx_cache))
            mean_diff = np.mean(np.abs(rknn_cache - onnx_cache))
            print(f"    cache_{i} max diff:  {max_diff:.6f}, mean diff: {mean_diff:.6f}")

        # 验证 softmax 结果一致性 (2类: speech/silence)
        rknn_probs = softmax(rknn_logits)
        onnx_probs = softmax(onnx_logits)
        max_prob_diff = np.max(np.abs(rknn_probs - onnx_probs))
        print(f"    softmax probs max diff: {max_prob_diff:.6f}")

    return True


def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    import numpy as np
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def analyze_output(rknn, n_frames=36):
    """分析模型输出结构"""
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Output Structure Analysis ({n_frames} frames)")
    print(f"{'='*60}")

    speech, caches = make_inputs(n_frames)
    outputs = run_rknn_inference(rknn, speech, caches)

    print(f"  Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}, "
              f"min={out.min():.4f}, max={out.max():.4f}, "
              f"mean={out.mean():.4f}, std={out.std():.4f}")

    # logits 分析: 248 维
    logits = outputs[0]
    print(f"\n  Logits [1,{logits.shape[1]},248] stats:")
    for f in range(min(3, logits.shape[1])):
        frame = logits[0, f, :]
        print(f"    frame {f}: min={frame.min():.4f} max={frame.max():.4f} "
              f"mean={frame.mean():.4f} top5_dims={frame.argsort()[-5:][::-1].tolist()}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description='RKNN PC Simulator Benchmark')
    parser.add_argument('--model', default=os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'model', 'model.rknn')),
        help='RKNN model path')
    parser.add_argument('--onnx', default=os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'model', 'model.onnx')),
        help='ONNX model path for verification')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations')
    parser.add_argument('--no-verify', action='store_true', help='Skip ONNX verification')
    args = parser.parse_args()

    # 检查模型文件
    if not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        sys.exit(1)

    # 加载 ONNX → 构建 RKNN → 初始化 PC 模拟器
    rknn = load_and_build(args.onnx, args.model)

    # 加载 ONNX 模型（用于输出对比）
    onnx_session = None
    input_names = None
    if not args.no_verify:
        onnx_session, input_names, _ = get_onnx_model(args.onnx)

    # 1. 输出结构分析
    analyze_output(rknn, n_frames=36)

    # 2. 正确性验证
    if not args.no_verify:
        verify_with_onnx(rknn, onnx_session, input_names)

    # 3. 速度基准测试
    speed_results = bench_speed(rknn, [6, 36], args.iterations)

    # 4. 流式场景模拟
    streaming_results = bench_streaming(rknn, args.iterations)

    # 5. 导出 RKNN 模型
    if not os.path.exists(args.model):
        print(f"\n--> Exporting RKNN model to {args.model}")
        ret = rknn.export_rknn(args.model)
        if ret == 0:
            print('    export done')
        else:
            print(f'    export failed (code {ret})')

    # 6. RTF 估算
    print(f"\n{'='*60}")
    print(f"RTF (Real-Time Factor) Estimation")
    print(f"{'='*60}")
    for n_frames, avg_ms in speed_results.items():
        audio_duration_ms = n_frames * 10  # 10ms shift per frame
        rtf = avg_ms / audio_duration_ms
        print(f"  {n_frames:2d} frames: {avg_ms:.3f}ms infer / {audio_duration_ms}ms audio = "
              f"RTF {rtf:.4f} ({'✅ REAL-TIME' if rtf < 1.0 else '❌ NOT REAL-TIME'})")

    # 流式 RTF
    first_rtf = streaming_results["first_36f"] / 360
    follow_rtf = streaming_results["follow_6f"] / 60
    print(f"  First frame (36f): RTF {first_rtf:.4f} ({'✅' if first_rtf < 1.0 else '❌'})")
    print(f"  Follow   (6f):  RTF {follow_rtf:.4f} ({'✅' if follow_rtf < 1.0 else '❌'})")

    rknn.release()
    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
