"""HTML 报表渲染。

属于 Algorithm 层——纯 Python f-string 模板，零 Qt 依赖。
Day 14 新增。浅色打印友好风格，等宽字体数据表格。
"""

from datetime import datetime
from pathlib import Path
from collections import defaultdict

from src.config.app_config import CLASS_NAMES, CLASS_COLORS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_html_report(
    csv_path: str,
    pie_base64: str,
    time_base64: str,
    conf_base64: str,
    output_path: str,
) -> str:
    """读取 CSV → 统计 → 渲染 HTML → 写入文件。

    Args:
        csv_path: 源 CSV 文件路径
        pie_base64: 类别分布饼图 base64 字符串
        time_base64: 耗时分布直方图 base64 字符串
        conf_base64: 置信度分布直方图 base64 字符串
        output_path: HTML 输出路径

    Returns:
        写入的 HTML 文件路径
    """
    # 读 CSV 统计
    rows = _read_csv_rows(csv_path)
    summary = _compute_summary(rows)
    no_defect_rows = [r for r in rows if r.get("has_defect", "").lower() != "true"]
    class_stats = _compute_class_stats(rows)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_filename = Path(csv_path).name

    # 各类检出统计表
    class_rows_html = ""
    for i, cls_name in enumerate(CLASS_NAMES):
        cs = class_stats.get(cls_name, {"count": 0, "total_dets": 0, "avg_conf": 0.0})
        color = CLASS_COLORS[i]
        class_rows_html += f"""
            <tr>
                <td><span class="color-dot" style="background:{color}"></span> {cls_name}</td>
                <td class="num">{cs['count']}</td>
                <td class="num">{cs['total_dets']}</td>
                <td class="num">{cs['avg_conf']:.3f}</td>
            </tr>"""

    # 无检出图像列表
    no_defect_html = ""
    if no_defect_rows:
        ndr = no_defect_rows[:50]  # 最多显示 50 张
        for r in ndr:
            error = r.get("error", "")
            img = r.get("image_filename", "?")
            no_defect_html += (
                f'<tr><td>{img}</td>'
                f'<td class="err">{error if error else "no detections"}</td></tr>\n'
            )
    else:
        no_defect_html = '<tr><td colspan="2" class="ok">所有图像均有检出 ✓</td></tr>'

    total_images = summary["total"]
    defect_count = summary["defect_count"]
    defect_rate = defect_count / total_images if total_images > 0 else 0.0
    avg_time = summary["avg_time_ms"]
    total_time = summary["total_time_ms"]
    error_count = summary["error_count"]
    error_color = "#1a7f37" if error_count == 0 else "#cf222e"  # Day 15 polish

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Metal Defect Detection — 批处理报表</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: Consolas, Menlo, "Courier New", monospace;
        background: #ffffff; color: #1f2328;
        max-width: 1100px; margin: 0 auto; padding: 32px 24px;
        line-height: 1.6;
    }}
    h1 {{
        color: #0969da; font-size: 24px; border-bottom: 3px solid #0969da;
        padding-bottom: 12px; margin-bottom: 8px;
    }}
    h2 {{ color: #1f2328; font-size: 18px; margin: 28px 0 12px; }}
    .subtitle {{ font-size: 0.7em; color: #6e7781; font-weight: 400; }}
    .chart-note {{ font-size: 0.85em; color: #6e7781; font-style: italic; text-align: center;
                   margin: 4px 0 16px; }}
    .meta {{ color: #656d76; font-size: 13px; margin-bottom: 20px; }}
    .stat-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
    .stat-box {{
        background: #f6f8fa; border: 1px solid #d0d7de;
        border-radius: 6px; padding: 12px 20px; min-width: 140px;
        text-align: center;
    }}
    .stat-box .label {{ font-size: 11px; color: #656d76; text-transform: uppercase; }}
    .stat-box .value {{ font-size: 22px; font-weight: 700; }}
    .stat-box .value.defect {{ color: #cf222e; }}
    .stat-box .value.ok {{ color: #1a7f37; }}
    table {{
        width: 100%; border-collapse: collapse; margin: 12px 0 20px;
        font-size: 13px;
    }}
    th {{
        background: #f6f8fa; color: #1f2328; font-weight: 600;
        padding: 8px 12px; text-align: left; border-bottom: 2px solid #d0d7de;
    }}
    td {{ padding: 6px 12px; border-bottom: 1px solid #e1e4e8; }}
    tr:hover td {{ background: #f6f8fa; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .color-dot {{
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px;
    }}
    .err {{ color: #cf222e; }}
    .ok {{ color: #1a7f37; }}
    img.chart {{ max-width: 100%; margin: 12px 0; border: 1px solid #e1e4e8; border-radius: 4px; }}
    footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #e1e4e8;
              color: #656d76; font-size: 12px; }}
</style>
</head>
<body>

<h1>🔍 Metal Defect Detection — 批处理报表</h1>
<p class="meta">
    生成时间: {timestamp}<br>
    CSV 源文件: {csv_filename}
</p>

<div class="stat-row">
    <div class="stat-box">
        <div class="label">总图像数</div>
        <div class="value">{total_images}</div>
    </div>
    <div class="stat-box">
        <div class="label">检出缺陷</div>
        <div class="value defect">{defect_count}</div>
    </div>
    <div class="stat-box">
        <div class="label">检出率</div>
        <div class="value ok">{defect_rate:.1%}</div>
    </div>
    <div class="stat-box">
        <div class="label">总耗时</div>
        <div class="value">{total_time / 1000:.1f}s</div>
    </div>
    <div class="stat-box">
        <div class="label">平均推理</div>
        <div class="value">{avg_time:.1f}ms</div>
    </div>
    <div class="stat-box">
        <div class="label">异常数</div>
        <div class="value" style="color: {error_color}">{error_count}</div>
    </div>
</div>

<h2>1. 类别检出分布 <span class="subtitle">Class Distribution</span></h2>
<img class="chart" src="data:image/png;base64,{pie_base64}" alt="类别分布饼图">
<p class="chart-note">饼图按检测框数统计。每张图可能包含多个同类或不同类检测框。详细图像数请见下方表格。</p>

<h2>2. 每类检出统计 <span class="subtitle">Per-Class Statistics</span></h2>
<table>
    <tr><th>缺陷类别</th><th class="num">检出图像数</th><th class="num">总检测框数</th><th class="num">平均置信度</th></tr>
    {class_rows_html}
</table>

<h2>3. 置信度分布 <span class="subtitle">Confidence Distribution</span></h2>
<img class="chart" src="data:image/png;base64,{conf_base64}" alt="置信度分布直方图">

<h2>4. 推理耗时分布 <span class="subtitle">Inference Time Distribution</span></h2>
<img class="chart" src="data:image/png;base64,{time_base64}" alt="推理耗时分布直方图">

<h2>5. 无检出 / 异常图像 <span class="subtitle">Failed Detections</span></h2>
<table>
    <tr><th>图像文件名</th><th>原因</th></tr>
    {no_defect_html}
</table>

<footer>
    Generated by Metal Defect Detection v1.0 &mdash;
    NEU-DET 6-class (crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches) &mdash;
    YOLOv8n + PyTorch 2.5.1 + CUDA 12.1
</footer>

</body>
</html>"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info(f"HTML report written to {output_path}")
    return str(output_path)


# ── CSV 读取与统计 ────────────────────────────────────────

def _read_csv_rows(csv_path: str) -> list[dict]:
    """读取 CSV 文件，返回每行 dict 列表。"""
    import csv
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _compute_summary(rows: list[dict]) -> dict:
    """计算总统计。"""
    total = len(rows)
    has_defect = sum(1 for r in rows if r.get("has_defect", "").lower() == "true")
    times = [float(r["inference_time_ms"]) for r in rows
             if r.get("inference_time_ms") and not r.get("error")]
    errors = sum(1 for r in rows if r.get("error", "").strip())
    return {
        "total": total,
        "defect_count": has_defect,
        "total_time_ms": round(sum(times), 1),
        "avg_time_ms": round(sum(times) / len(times), 1) if times else 0.0,
        "error_count": errors,
    }


def _compute_class_stats(rows: list[dict]) -> dict:
    """按类别统计检出数、总检测框数、平均置信度。"""
    stats = defaultdict(lambda: {"count": 0, "total_dets": 0, "confs": []})

    for r in rows:
        if r.get("has_defect", "").lower() != "true":
            continue
        classes = r.get("classes_detected", "")
        ndets = int(r.get("num_detections", 0))
        score = float(r.get("top_score", 0))
        if not classes:
            continue
        cls_list = classes.split(";")
        for cls_name in set(cls_list):  # 每张图每类算一次
            if cls_name in CLASS_NAMES:
                stats[cls_name]["count"] += 1
        if cls_list:
            top_cls = cls_list[0]
            if top_cls in CLASS_NAMES:
                stats[top_cls]["confs"].append(score)
        stats[cls_list[0]]["total_dets"] += ndets

    result = {}
    for cls_name in CLASS_NAMES:
        s = stats.get(cls_name, {"count": 0, "total_dets": 0, "confs": []})
        result[cls_name] = {
            "count": s["count"],
            "total_dets": s["total_dets"],
            "avg_conf": round(sum(s["confs"]) / len(s["confs"]), 4) if s["confs"] else 0.0,
        }
    return result
