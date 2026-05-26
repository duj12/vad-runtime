#!/usr/bin/env python3
"""
FSMN VAD ONNX → RKNN 模型转换脚本 (RK3588)

模型 I/O:
  Inputs:
    speech    [1, feats_length, 400]    — LFR+CMVN 后的特征
    in_cache0 [1, 128, 19, 1]           — FSMN 层 0 缓存
    in_cache1 [1, 128, 19, 1]           — FSMN 层 1 缓存
    in_cache2 [1, 128, 19, 1]           — FSMN 层 2 缓存
    in_cache3 [1, 128, 19, 1]           — FSMN 层 3 缓存
  Outputs:
    logits    [1, N, 248]               — 语音/非语音 logits
    out_cache0-3 [1, 128, 19, 1]        — 更新后的 FSMN 缓存

依赖安装:
  1. 下载 rknn-toolkit2: https://github.com/rockchip-linux/rknn-toolkit2
  2. pip install rknn-toolkit2      # 或从 Rockchip 提供的 whl 安装
  3. pip install onnx numpy

用法:
    python3 convert_onnx_to_rknn.py <onnx_model> <rknn_model> [options]
    python3 convert_onnx_to_rknn.py --verify         # 转换+模拟器验证

选项:
    --platform rk3588      目标平台 (默认: rk3588)
    --quantize             启用INT8量化 (默认: FP32)
    --dataset DATASET      量化校准数据集文件
    --verify               转换后用PC模拟器验证
"""
import os
import argparse
import sys


def inspect_onnx(onnx_path):
    """打印 ONNX 模型输入输出信息"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        print(f"\n=== ONNX Model: {os.path.basename(onnx_path)} ===")
        print(f"IR version: {model.ir_version}")
        print("\nInputs:")
        for i in model.graph.input:
            dims = [d.dim_param if d.dim_param else str(d.dim_value)
                    for d in i.type.tensor_type.shape.dim]
            print(f"  [{i.name}] shape: {dims}")
        print("\nOutputs:")
        for o in model.graph.output:
            dims = [d.dim_param if d.dim_param else str(d.dim_value)
                    for d in o.type.tensor_type.shape.dim]
            print(f"  [{o.name}] shape: {dims}")
    except Exception as e:
        print(f"Warning: cannot inspect ONNX model: {e}")


def verify_simulator(rknn, model_path):
    """用 PC 模拟器验证 RKNN 模型推理"""
    try:
        import numpy as np
    except ImportError:
        print("Warning: numpy not installed, skipping verification")
        return

    # 重新创建 rknn 对象加载模型
    print("\n=== Verifying with PC simulator ===")
    ret = rknn.init_runtime(target=None)
    if ret != 0:
        print(f"Warning: simulator init failed (code {ret})")
        return

    # 从 model_path 解析输入输出个数
    # 构建 dummy 输入并测试多种尺寸
    # 用 6 帧（常规 chunk）和 36 帧（首帧）验证 dynamic_input
    for test_frames in [6, 36]:
        dummy_speech = np.random.randn(1, test_frames, 400).astype(np.float32)
        dummy_cache_nchw = np.zeros((1, 128, 19, 1), dtype=np.float32)
        dummy_cache = np.transpose(dummy_cache_nchw, (0, 2, 3, 1))  # NCHW→NHWC
        inputs = [dummy_speech, dummy_cache, dummy_cache, dummy_cache, dummy_cache]

        outputs = rknn.inference(inputs=inputs)
        print(f"  [{test_frames}frames] {len(outputs)} output(s)")
        for i, out in enumerate(outputs):
            print(f"    Output {i}: shape={out.shape}")


def convert(args):
    onnx_path = args.onnx_model
    rknn_path = args.rknn_model

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found: {onnx_path}")
        sys.exit(1)

    # 检查 ONNX 模型结构
    inspect_onnx(onnx_path)

    # 延迟导入，避免未安装时直接报错
    try:
        from rknn.api import RKNN
    except ImportError:
        print("Error: rknn-toolkit2 not installed. Run: pip install rknn-toolkit2")
        sys.exit(1)

    rknn = RKNN(verbose=True)

    # 配置
    print('\n--> Config model')
    ret = rknn.config(
        mean_values=None,
        std_values=None,
        target_platform=args.platform,
        optimization_level=3,
        output_optimize=True,
        dynamic_input=[
            [[1, 1, 400], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1]],   # min
            [[1, 6, 400], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1]],    # opt
            [[1, 36, 400],[1, 128, 19, 1],[1, 128, 19, 1],[1, 128, 19, 1],[1, 128, 19, 1]],      # max
        ]
    )
    if ret != 0:
        print('rknn.config failed!')
        sys.exit(1)
    print('done')

    # 加载 ONNX（dynamic_input 会处理尺寸）
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('rknn.load_onnx failed!')
        sys.exit(1)
    print('done')

    # 构建
    print('--> Building model')
    dataset = args.dataset if (args.quantize and args.dataset) else ''
    ret = rknn.build(do_quantization=args.quantize, dataset=dataset)
    if ret != 0:
        print('rknn.build failed!')
        sys.exit(1)
    print('done')

    # 导出
    print(f'--> Export RKNN model: {rknn_path}')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('rknn.export_rknn failed!')
        sys.exit(1)
    print('done')

    # 模拟器验证
    if args.verify:
        verify_simulator(rknn, rknn_path)

    rknn.release()
    print('\nConversion done!')


def main():
    parser = argparse.ArgumentParser(description='Convert FSMN VAD ONNX model to RKNN')
    parser.add_argument('onnx_model', nargs='?',
                        default=os.path.normpath(os.path.join(
                            os.path.dirname(__file__), '..', 'model', 'model.onnx')),
                        help='Input ONNX model path')
    parser.add_argument('rknn_model', nargs='?',
                        default=os.path.normpath(os.path.join(
                            os.path.dirname(__file__), '..', 'model', 'model.rknn')),
                        help='Output RKNN model path')
    parser.add_argument('--platform', default='rk3588', help='Target platform (default: rk3588)')
    parser.add_argument('--quantize', action='store_true', help='Enable INT8 quantization')
    parser.add_argument('--dataset', help='Calibration dataset file for quantization')
    parser.add_argument('--verify', action='store_true', help='Verify with PC simulator after conversion')

    args = parser.parse_args()
    convert(args)


if __name__ == '__main__':
    main()
