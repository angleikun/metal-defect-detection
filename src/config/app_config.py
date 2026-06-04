"""应用级配置常量。

属于 Config 层——定义模型路径、阈值、窗口尺寸、6 类名和配色。
所有 UI 和 Manager 模块从此文件 import，禁止硬编码值。
"""

from pathlib import Path

# ── 项目根目录 ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── 模型路径 ──────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "models"
YOLO_BEST = MODEL_DIR / "exp4_crazing_focus" / "weights" / "best.pt"
UNET_BEST = MODEL_DIR / "unet_baseline_best.pt"

# ── 数据路径 ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "raw" / "NEU-DET"

# ── 窗口 ──────────────────────────────────────────────────
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
WINDOW_TITLE = "Metal Defect Detection — SCADA"

# ── 检测阈值（生产模式） ──────────────────────────────────
# Day 12 修复：crazing 纹理缺陷置信度天然低（top~0.23），
# 0.25 会导致该类完全漏检。降至 0.10 覆盖低置信度缺陷。
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45
MAX_DET = 300

# ── NEU-DET 6 类名 ───────────────────────────────────────
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# ── 6 类 BBox 渲染色（十六进制） ─────────────────────────
CLASS_COLORS = [
    "#FF5252",  # crazing        — 红
    "#448AFF",  # inclusion      — 蓝
    "#69F0AE",  # patches        — 绿
    "#FFD740",  # pitted_surface — 琥珀
    "#E040FB",  # rolled-in_scale— 紫
    "#40C4FF",  # scratches      — 青
]

# ── 检测框渲染 ──────────────────────────────────────────────
DETECTION_LINE_WIDTH = 2

# ── per-class 置信度阈值（Day 13） ─────────────────────────
# Day 12 发现 crazing 类 conf 天然偏低（top~0.23），全局阈值 0.25 导致漏检。
# 推理用全局最低值(0.05)拿所有候选，后处理按类别阈值过滤。
PER_CLASS_CONF = {
    "crazing": 0.05,
    "inclusion": 0.20,
    "patches": 0.20,
    "pitted_surface": 0.15,
    "rolled-in_scale": 0.10,
    "scratches": 0.20,
}

# ── 批处理 ─────────────────────────────────────────────────
BATCH_EMIT_EVERY = 10
MAX_BATCH_FILES = 5000  # 单次批处理文件数上限，超限弹警告

# ── 报表 ──────────────────────────────────────────────────
REPORT_DIR = PROJECT_ROOT / "results"
BATCH_OUTPUT_DIR = REPORT_DIR / "batch_runs"
