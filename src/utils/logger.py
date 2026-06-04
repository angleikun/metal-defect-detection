"""日志系统。

属于 Utils 层——同时输出到终端（StreamHandler）和 GUI 事件日志面板。
所有模块用 ```python from src.utils.logger import get_logger ```
替换 print()。
"""

import logging
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


# ── Qt 安全日志信号发射器 ──────────────────────────────────
class _LogSignalEmitter(QObject):
    """将 logging 消息通过 Qt 信号跨线程安全发送到 GUI。

    不直接操作 QWidget；由 log_panel.py 的 LogPanel 连接此信号。
    """

    log_message = pyqtSignal(str)  # 拼接后的日志行


_emitter = _LogSignalEmitter()

# 模块级引用，供 LogPanel 连接
log_signal = _emitter.log_message


# ── 自定义 logging Handler ─────────────────────────────────
class _QtLogHandler(logging.Handler):
    """logging.Handler 子类，将日志记录 emit 到 Qt 信号。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _emitter.log_message.emit(msg)
        except Exception:
            self.handleError(record)


# ── 模块级 logging 配置 ────────────────────────────────────
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "metal_defect") -> logging.Logger:
    """获取项目统一 logger。

    首次调用时配置：
    - 终端输出（StreamHandler），DEBUG 级别，简洁格式
    - Qt 面板输出（_QtLogHandler），INFO 级别，含时间戳

    Args:
        name: logger 名称，默认 "metal_defect"

    Returns:
        配置好的 logging.Logger 实例
    """
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 不重复传播到 root logger

    # 终端 handler：DEBUG 及以上，简洁格式
    console_fmt = logging.Formatter(
        "%(levelname)-8s %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # Qt 面板 handler：INFO 及以上，含时间戳
    qt_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    qt_handler = _QtLogHandler()
    qt_handler.setLevel(logging.INFO)
    qt_handler.setFormatter(qt_fmt)
    logger.addHandler(qt_handler)

    _logger = logger
    return _logger
