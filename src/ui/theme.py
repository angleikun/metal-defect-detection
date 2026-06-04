"""SCADA 风格主题常量。

属于 UI 层——所有颜色、字体、间距常量集中定义。
主窗口和子面板从此文件 import，禁止在 widget 中硬编码颜色值。

参考视觉：海康威视 iVMS-4200 / 西门子 WinCC 监控界面。
"""

# ── 画布背景 ──────────────────────────────────────────────
BACKGROUND         = "#0d1117"   # 主背景，深灰黑
SURFACE            = "#161b22"   # 卡片/面板背景
SURFACE_ALT        = "#1c2333"   # 交替面板背景

# ── 边框与分隔线 ──────────────────────────────────────────
BORDER             = "#30363d"   # 面板边框
DIVIDER            = "#21262d"   # 分隔线

# ── 文字 ──────────────────────────────────────────────────
TEXT_PRIMARY       = "#f0f6fc"   # 主要文字
TEXT_SECONDARY     = "#8b949e"   # 次要文字（标签、描述）
TEXT_DIM           = "#484f58"   # 暗淡文字

# ── 语义色（状态编码） ────────────────────────────────────
GREEN_OK           = "#00c853"   # 正常运行 / 检测通过
RED_ALARM          = "#ff1744"   # 报警 / 缺陷检出
YELLOW_WARNING     = "#ffd600"   # 警告 / 低置信度
CYAN_INFO          = "#00e5ff"   # 信息 / 系统消息

# ── 字体 ──────────────────────────────────────────────────
FONT_FAMILY        = "Consolas"          # 等宽字体
FONT_SIZE_STATUS   = 11
FONT_SIZE_LOG      = 11
FONT_SIZE_PANEL    = 12
FONT_WEIGHT_NORMAL = 400
FONT_WEIGHT_BOLD   = 700

# ── 间距 ──────────────────────────────────────────────────
SPACING_XS  = 4
SPACING_SM  = 8
SPACING_MD  = 12
SPACING_LG  = 16
SPACING_XL  = 24

# ── 面板最小尺寸 ──────────────────────────────────────────
LEFT_PANEL_WIDTH   = 220
RIGHT_PANEL_WIDTH  = 280
LOG_PANEL_HEIGHT   = 140
STATUS_BAR_HEIGHT  = 28
MENU_BAR_HEIGHT    = 32
