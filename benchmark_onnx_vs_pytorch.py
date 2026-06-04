"""Day 15 基准测试：PyTorch GPU vs ONNX CPU 100 张 NEU-DET。

Warmup: 前 10 张不进统计。
输出: results/benchmark.csv + 终端汇总。
"""
import csv
import statistics
import time
from pathlib import Path

from src.algo.yolo_detector import YoloDetector
from src.algo.yolo_detector_onnx import YoloDetectorONNX
from src.config.app_config import YOLO_BEST, ONNX_MODEL, IMAGE_DIR, REPORT_DIR

# ── 加载模型 ─────────────────────────────────────────────
print("Loading models...")
pt = YoloDetector(str(YOLO_BEST), device="cuda")
onnx = YoloDetectorONNX(str(ONNX_MODEL))
print(f"  PyTorch: cuda")
print(f"  ONNX:    CPU (onnxruntime)")

# ── 准备图片 ─────────────────────────────────────────────
all_images = sorted(IMAGE_DIR.rglob("*.jpg"))
bench_images = all_images[:100]
print(f"Benchmark images: {len(bench_images)} (first 100)")

WARMUP = 10
num_bench = len(bench_images) - WARMUP

# ── 结果存储 ─────────────────────────────────────────────
results = []


def run_benchmark(label, detector, images, is_pt=True):
    """运行基准测试，返回 per-image 耗时列表（不含 warmup）。"""
    times = []
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")

    for i, img_path in enumerate(images):
        t0 = time.perf_counter()
        _ = detector.detect(img_path, conf=0.05, iou=0.45, per_class_conf=None)
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)

        stage = "WARMUP" if i < WARMUP else "BENCH"
        bar = "█" * (i * 50 // len(images))
        print(f"\r  [{bar:<50s}] {i+1}/{len(images)} {stage} {elapsed:.1f}ms", end="")

    bench_times = times[WARMUP:]
    avg = statistics.mean(bench_times)
    p50 = statistics.median(bench_times)
    p95 = sorted(bench_times)[int(num_bench * 0.95)]
    p99 = sorted(bench_times)[int(num_bench * 0.99)]

    print(f"\n  Avg: {avg:.1f}ms | P50: {p50:.1f}ms | P95: {p95:.1f}ms | P99: {p99:.1f}ms")
    return bench_times


# ── PyTorch GPU ──────────────────────────────────────────
pt_times = run_benchmark("PyTorch GPU", pt, bench_images, is_pt=True)

# ── ONNX CPU ─────────────────────────────────────────────
onnx_times = run_benchmark("ONNX CPU", onnx, bench_images, is_pt=False)

# ── 写入 CSV ─────────────────────────────────────────────
REPORT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = REPORT_DIR / "benchmark.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_filename", "pytorch_gpu_ms", "onnx_cpu_ms"])
    for i in range(num_bench):
        img_name = bench_images[WARMUP + i].name
        writer.writerow([
            img_name,
            f"{pt_times[i]:.1f}",
            f"{onnx_times[i]:.1f}",
        ])

print(f"\nCSV saved: {csv_path} ({num_bench} rows)")

# ── 汇总 ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"BENCHMARK SUMMARY (100 images, first 10 warmup, {num_bench} measured)")
print(f"{'='*60}")

pt_avg = statistics.mean(pt_times)
pt_p50 = statistics.median(pt_times)
pt_p95 = sorted(pt_times)[int(num_bench * 0.95)]
pt_p99 = sorted(pt_times)[int(num_bench * 0.99)]

onnx_avg = statistics.mean(onnx_times)
onnx_p50 = statistics.median(onnx_times)
onnx_p95 = sorted(onnx_times)[int(num_bench * 0.95)]
onnx_p99 = sorted(onnx_times)[int(num_bench * 0.99)]

speedup = pt_avg / onnx_avg  # < 1 = ONNX slower
onnx_fps = 1000.0 / onnx_avg
pt_fps = 1000.0 / pt_avg

print(f"{'':<15s} {'PyTorch GPU':>15s} {'ONNX CPU':>15s} {'Ratio':>10s}")
print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*10}")
print(f"{'Avg (ms)':<15s} {pt_avg:>15.1f} {onnx_avg:>15.1f} {speedup:>10.2f}x")
print(f"{'P50 (ms)':<15s} {pt_p50:>15.1f} {onnx_p50:>15.1f}")
print(f"{'P95 (ms)':<15s} {pt_p95:>15.1f} {onnx_p95:>15.1f}")
print(f"{'P99 (ms)':<15s} {pt_p99:>15.1f} {onnx_p99:>15.1f}")
print(f"{'FPS':<15s} {pt_fps:>15.1f} {onnx_fps:>15.1f}")
print(f"{'Speedup':<15s} {'1.00x (baseline)':>15s} {f'{1/speedup:.1f}x slower':>15s}")

print(f"\n>>> Benchmark complete <<<")
