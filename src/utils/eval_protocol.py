"""统一评测协议。

所有 mAP 数字都必须用 EVAL_CONFIG，禁止散用默认值。
Day 4 发现 conf 阈值口径不统一会导致 mAP 偏差 0.01-0.02。

区分两个场景：
- EVAL_CONFIG: 学术评测，conf=0.001（COCO 标准，记录所有预测的 PR 曲线）
- PRODUCTION_CONFIG: 桌面应用部署，conf=0.25（过滤弱预测，给用户干净结果）
"""

EVAL_CONFIG = {
    "conf": 0.001,       # COCO 评测标准值，不丢弃低置信预测
    "iou": 0.6,           # NMS IoU 阈值
    "imgsz": 640,
    "max_det": 300,
    "device": 0,
    "verbose": False,
}

PRODUCTION_CONFIG = {
    "conf": 0.25,         # 生产环境过滤弱预测
    "iou": 0.45,          # 生产环境更激进的 NMS
    "imgsz": 640,
    "max_det": 300,
    "device": 0,
    "verbose": False,
}
