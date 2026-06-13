"""
benchmark_latency.py
对比 FP32 vs INT8 ONNX 模型的 CPU 推理延迟
"""

import numpy as np
import onnxruntime as ort
import time
import os

FP32_MODEL = "yolov8n_neu_fp32.onnx"
INT8_MODEL = "yolov8n_neu_int8.onnx"

WARMUP = 5
ITERS  = 50

np.random.seed(42)
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)


def benchmark(model_path, label):
    print(f"\n=== {label} ===")
    print(f"模型: {model_path}")
    
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"大小: {size_mb:.2f} MB")
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    sess = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    
    input_name = sess.get_inputs()[0].name
    
    print(f"Warmup ({WARMUP} iters)... ", end="", flush=True)
    for _ in range(WARMUP):
        sess.run(None, {input_name: dummy_input})
    print("done")
    
    print(f"测量 ({ITERS} iters)... ", end="", flush=True)
    latencies = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)
    print("done")
    
    latencies = np.array(latencies)
    mean_ms = latencies.mean()
    median_ms = np.median(latencies)
    p95_ms = np.percentile(latencies, 95)
    
    print(f"  平均:    {mean_ms:6.2f} ms")
    print(f"  中位数:  {median_ms:6.2f} ms")
    print(f"  p95:     {p95_ms:6.2f} ms")
    print(f"  吞吐:    {1000/mean_ms:.1f} FPS")
    
    return {
        'size_mb': size_mb,
        'mean_ms': mean_ms,
        'median_ms': median_ms,
        'p95_ms': p95_ms,
    }


print("=" * 50)
print(" ONNX Runtime CPU 推理延迟 Benchmark")
print(f" Warmup: {WARMUP}  Iters: {ITERS}  Threads: 1")
print(f" Input: (1, 3, 640, 640) float32")
print("=" * 50)

fp32 = benchmark(FP32_MODEL, "FP32 ONNX")
int8 = benchmark(INT8_MODEL, "INT8 ONNX (Dynamic Quant)")

size_ratio = fp32['size_mb'] / int8['size_mb']
mean_ratio = fp32['mean_ms'] / int8['mean_ms']
median_ratio = fp32['median_ms'] / int8['median_ms']
fps_fp32 = 1000 / fp32['mean_ms']
fps_int8 = 1000 / int8['mean_ms']
fps_gain = fps_int8 / fps_fp32 - 1

print("\n" + "=" * 50)
print(" 对比总结")
print("=" * 50)
print(f"{'指标':<15} {'FP32':>12} {'INT8':>12} {'变化':>15}")
print("-" * 55)
print(f"{'模型大小 (MB)':<15} {fp32['size_mb']:>12.2f} {int8['size_mb']:>12.2f} {size_ratio:>10.2f}x 缩")
print(f"{'平均延迟 (ms)':<15} {fp32['mean_ms']:>12.2f} {int8['mean_ms']:>12.2f} {mean_ratio:>10.2f}x 快")
print(f"{'中位数 (ms)':<15}  {fp32['median_ms']:>12.2f} {int8['median_ms']:>12.2f} {median_ratio:>10.2f}x 快")
print(f"{'吞吐 (FPS)':<15}   {fps_fp32:>12.1f} {fps_int8:>12.1f}     +{fps_gain:>6.0%}")
print("=" * 50)
