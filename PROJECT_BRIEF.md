# PROJECT BRIEF · 金属表面缺陷检测系统

| | |
|---|---|
| **项目名** | metal-surface-defect-detection |
| **作者** | linsanqin |
| **GitHub** | github.com/angleikun（待创建仓库） |
| **创建日期** | 2026-06-02 |
| **预计完成** | 2026-06-30（4 周） |
| **文档版本** | v1.1.1 |
| **配套文档** | PROJECT_PLAN.md（执行细节） |

---

## 一、一段话总结

基于 NEU-DET 公开数据集和 YOLOv8 框架，独立开发面向工业金属表面（钢板、汽车冲压件、铝型材）的 6 类缺陷检测系统，提供 PyQt6 桌面应用形态和 ONNX 推理后端，4 周内完成从算法训练到工程化部署的完整链路。

---

## 二、问题陈述

### 2.1 行业问题

工业生产中金属表面缺陷（裂纹、夹杂、麻点、划痕等）的检测长期依赖人工目检：

- **效率低**：钢厂典型产线带宽几百米/分钟，人眼跟不上
- **一致性差**：不同检验员标准不一致，疲劳后漏检率上升
- **传统机器视觉局限**：基于阈值和形状的算法对裂纹这类不规则缺陷识别率低于 70%

深度学习目标检测是当前行业主流升级方向，海康、大华、思谋、阿丘等公司均在该方向有产品。

### 2.2 项目动机

承接上一个项目「工业机器人视觉引导系统」（HALCON 形状匹配，做"找在哪"），本项目做"判好坏"，**两个项目互补构成工业视觉全栈**：

| 上一个项目 | 本项目 |
|---|---|
| 实时定位 | 批量质检 |
| 传统机器视觉算子（HALCON） | 深度学习（YOLOv8 / U-Net） |
| Qt + C++ | Python + PyQt6 |

---

## 三、项目范围

### 3.1 范围内（必做）

| 功能 | 说明 |
|---|---|
| 6 类缺陷检测 | NEU-DET 定义的 crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches |
| YOLOv8 主算法 | 基于 Ultralytics 框架训练，含完整训练 + 评测 + 调参实验 |
| U-Net 对照实验 | 像素级分割模型 + 30 张人工标注绝对评测，作为算法选型决策依据 |
| PyQt6 桌面应用 | 单图检测 + 批处理 + 参数调节 + 报表导出 |
| ONNX 推理部署 | PyTorch 模型导出 ONNX，验证一致性，做 PyTorch/ONNX 性能对比 |
| 完整文档 | README / DEVLOG / 架构图 / Demo 视频 |

### 3.2 范围外（明确不做）

| 不做的事 | 原因 |
|---|---|
| 实时摄像头采集 | 用静态图测试集即可验证算法，硬件采集和算法是独立问题 |
| PLC / 工业相机集成 | 没有真实工业相机，且偏离算法核心 |
| 缺陷严重程度分级 | 数据集只给类别标注，分级要额外数据，超出 4 周范围 |
| Transformer 系列模型 | 6GB 显存限制，且 YOLOv8 已能达标 |
| 多语言界面 | portfolio 项目仅需中文 |
| 安装包 / 自动更新 | 演示用 `python main.py` 启动即可 |
| 在线学习 / 增量训练 | 离线训练已满足需求 |

### 3.3 验收用户

| 角色 | 关注点 |
|---|---|
| **自己**（学习目标） | 完整跑通深度学习项目工程化全流程 |
| **简历筛选者**（HR/技术经理） | 30 秒能看懂项目价值，1 分钟看出技术含量 |
| **面试官**（机器视觉/AI 岗位） | 能就具体技术决策、bug 修复深入对话 5-10 分钟 |

---

## 四、验收指标

### 4.1 算法指标

| 指标 | 最低 | 目标 | 拉满 |
|---|---|---|---|
| mAP@0.5（整体，test set） | 0.80 | 0.85 | 0.88 |
| mAP@0.5:0.95（整体，test set） | 0.40 | 0.50 | 0.55 |
| 每类 Recall | 0.75 | 0.85 | 0.90 |
| mAP@0.5（small：bbox < 32²） | 0.65 | 0.75 | 0.80 |
| mAP@0.5（medium：32²-96²） | 0.80 | 0.85 | 0.88 |
| mAP@0.5（large：> 96²） | 0.85 | 0.90 | 0.92 |
| U-Net IoU（vs 30 张人工 mask） | 0.50 | 0.60 | 0.68 |

注 1：所有 mAP 报 test set（180 张）数字，val set 仅用于训练期挑权重。

注 2：分尺度按 NEU-DET 原图（200×200）坐标系下的 bbox 面积划分。
- small: area < 32² = 1024 px²
- medium: 32² ≤ area < 96² = 9216 px²
- large: area ≥ 96²

Day 2 EDA 必须输出 bbox 面积分布直方图，确认三档样本数 ≥ 30。
若某档样本数 < 30：DEVLOG 记一行"该档 mAP 高方差，简历仅参考不强报"。

注意：COCO 评测惯例是按原图坐标系，不是按 resize 后的输入分辨率。
即使 YOLOv8 训练 imgsz=640，评测分尺度仍用原图坐标 (200×200)。

注 3：U-Net 评测 GT = 30 张人工标注的真实像素 mask（来自 test set，
每类 5 张，LabelMe polygon 标注）。这是绝对评测，能抗追问。
训练数据仍用 refined mask（1440 张），但评测不用。

数字下调说明：refined mask 训练的 U-Net 在真实 mask 上 IoU 
会比在 refined mask 上低 0.10 左右（拟合了 Otsu 的系统性偏差），
因此目标线 0.60，拉满线 0.68。

人工标注数据 commit 进 git（路径：data/annotations_manual/），
是项目可复现性的关键产物。

### 4.2 性能指标（RTX 3060 Laptop 6GB）

| 指标 | 最低 | 目标 |
|---|---|---|
| 单图推理（PyTorch GPU） | < 50 ms | < 25 ms |
| 单图推理（ONNX GPU） | < 30 ms | < 15 ms |
| 单图推理（ONNX CPU） | < 300 ms | < 200 ms |
| 批处理吞吐 | > 30 img/s | > 60 img/s |
| 内存占用（运行时） | < 2 GB | < 1 GB |
| 应用启动时间 | < 10 s | < 5 s |

### 4.3 工程指标

| 项目 | 验收线 |
|---|---|
| GitHub 仓库 | Public + LICENSE + Topics + README 含 demo |
| DEVLOG 记录 | 每条含现象/根因/修复/教训四段，质量优先；N 为实测条数（不强求 ≥ 10） |
| 代码组织 | src/ 内函数 ≤ 100 行，无硬编码本地路径 |
| 测试覆盖 | 至少对核心数据加载和坐标转换函数写单元测试 |
| Demo 视频 | ≥ 30 秒，分屏显示原图 + 检测结果 |

### 4.4 简历可写产出

完成后简历项目栏应能写出：

```
- 基于 NEU-DET 数据集训练 YOLOv8 模型，test set mAP@0.5 = 0.XX
  （small / medium / large 目标分尺度 mAP 分别为 0.XX / 0.XX / 0.XX）
- 实现 U-Net 像素级分割对照实验，弱监督管线：bbox + Otsu refine 生成
  伪 mask（1440 张训练样本），人工标注 30 张测试样本作绝对评测基准
- 基于人工 mask 评测，refined mask 训练相比 baseline mask 训练 IoU
  从 0.XX 提升到 0.XX（提升 0.XX）
- PyQt6 + qfluentwidgets 桌面应用，批处理吞吐 XX img/s
- ONNX Runtime 部署，相比 PyTorch 推理速度提升 XX%
- 固定 seed + deterministic 模式确保实验可复现，mAP 浮动 < 0.005
- DEVLOG 记录 N 个真实 bug 的诊断和修复过程
```

---

## 五、风险与缓解

### 5.1 技术风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| NEU-DET 数据量小，过拟合严重 | 中 | 高 | ImageNet 预训练 + mosaic/mixup 强增强 + early stopping |
| 6GB 显存不足训练大模型 | 中 | 中 | 仅用 YOLOv8n/s，batch_size ≤ 16，imgsz ≤ 640 |
| ONNX 转换后精度损失 | 中 | 中 | 转完做一致性测试，输出误差 < 1e-4 才算通过 |
| Windows + DataLoader num_workers 卡死 | 中 | 低 | Day 3 先 num_workers=0 跑通；再测 num_workers=2/4；DEVLOG 记两者 epoch 耗时对比 |
| OpenCV 中文路径读图报错 | 高 | 低 | 用 cv2.imdecode + np.fromfile 替代 cv2.imread |
| ultralytics 版本升级 API 变化 | 低 | 中 | requirements.txt 锁定 8.4.60 |

### 5.2 进度风险

| 风险 | 缓解 |
|---|---|
| Week 1 数据集下载/转换占时超预算 | 准备 3 个候选下载源（官方 / Kaggle / GitHub 镜像） |
| Week 2 U-Net 训练效果不理想 | 接受 IoU < 0.50（vs 人工 mask）作为"弱监督上限"的合法结论；DEVLOG 写清楚是标签问题而非模型问题 |
| Week 3 PyQt6 + qfluentwidgets 学习曲线 | 复用上个项目的 Qt 知识，遇到瓶颈可降级用原生 PyQt 控件 |
| Week 4 时间不够 | 砍 PyQt6 → 砍 U-Net → 砍 ONNX，保住 YOLOv8 + GitHub + demo |

### 5.3 学术诚信风险

| 风险 | 缓解 |
|---|---|
| 简历夸大成果（实际 0.82 写成 0.90） | 所有数字来自实测的 CSV 报告，禁止任何调整 |
| 模型权重借用他人 | 完整自训，DEVLOG 记录每次实验过程 |
| 代码大段抄袭 GitHub | 借鉴可以但必须重写理解，引用源放 README |

---

## 六、关键技术决策（含理由）

| 决策 | 选项 | 选择 | 理由 |
|---|---|---|---|
| 检测框架 | YOLOv8 / YOLOv9 / YOLOv10 / Detectron2 | **YOLOv8** | 生态成熟、文档完整、社区资源最多；招聘场景 YOLO 关键字识别度高 |
| 数据集 | NEU-DET / MVTec AD / GC10-DET | **NEU-DET** | 免费、有标注、规模合适、行业认可度高 |
| 训练框架 | 纯 PyTorch / Ultralytics / PyTorch Lightning | **Ultralytics** | YOLOv8 官方实现，避免自造轮子；U-Net 部分用纯 PyTorch |
| 桌面 UI | PyQt6 / Tkinter / Streamlit / Electron | **PyQt6** | 复用上个项目的 Qt 经验；qfluentwidgets 视觉效果加分 |
| 推理后端 | PyTorch / ONNX / TensorRT | **PyTorch + ONNX 双后端** | TensorRT 部署成本高，ONNX 已能体现工程化能力 |
| 实验追踪 | wandb / mlflow / TensorBoard / 手工 CSV | **TensorBoard + CSV** | wandb 需注册，TensorBoard 本地够用 |
| 包管理 | conda / pip / poetry | **mamba 建环境 + pip 装包** | 复用已有 miniforge 环境 |
| 报表格式 | CSV / Excel / PDF | **三种都支持** | 复用上个项目的 ReportManager 思路 |

---

## 七、时间表

| 周 | 主题 | 关键产出 |
|---|---|---|
| **Week 1**（Day 1-5） | 算法跑通 | YOLOv8 训完，mAP > 0.85，初版 README |
| **Week 2**（Day 6-10） | 对照实验 | U-Net 伪 mask refine（Otsu+形态学）+ 训练，30 张人工 mask 评测，三模型对比表，消融实验 |
| **Week 3**（Day 11-15） | 桌面应用 | PyQt6 应用可用，能录 demo |
| **Week 4**（Day 16-20） | 工程化 | ONNX 部署，DEVLOG 完整，GitHub 上线 |

详细每日任务见 `PROJECT_PLAN.md`。

---

## 八、成功标准（三档）

### 最低线（必须达到）

- [x] YOLOv8 训练完成，mAP@0.5 ≥ 0.80
- [x] GitHub 仓库公开，README + LICENSE + 5 条以上 DEVLOG
- [x] 终端命令可演示检测（`python detect.py xxx.jpg`）

**未达此线 = 项目失败，需要复盘**。

### 目标线（争取达到）

- [ ] mAP@0.5 ≥ 0.85
- [ ] U-Net 对照实验完成
- [ ] PyQt6 桌面应用可用
- [ ] 30 秒以上 demo 视频
- [ ] ONNX 推理验证通过

**达此线 = 简历可以正常写**。

### 拉满线（理想结果）

- [ ] mAP@0.5 ≥ 0.90
- [ ] 三种推理后端性能基准对比
- [ ] 单元测试覆盖核心模块
- [ ] DEVLOG 含 ≥ 15 条真实 bug 复盘
- [ ] 同步发表知乎/CSDN 技术博客 ≥ 1 篇

**达此线 = 简历加分项明显**。

---

## 九、参考资料

### 数据集与论文

- He, Y., Song, K., et al. *An end-to-end steel surface defect detection approach via fusing multiple hierarchical features*. IEEE TIM, 2020.
- NEU-DET dataset: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm

### 框架文档

- Ultralytics YOLOv8 Docs: https://docs.ultralytics.com
- segmentation_models_pytorch: https://github.com/qubvel/segmentation_models.pytorch
- ONNX Runtime: https://onnxruntime.ai/docs/
- PyQt6-Fluent-Widgets: https://github.com/zhiyiYo/PyQt-Fluent-Widgets

### 个人知识库

- `D:\dev_notes\AI_CODING_TRAPS.md` —— 上一个项目沉淀的 AI 编程踩坑
- 上一个项目 GitHub：github.com/angleikun/industrial-robot-vision

---

## 十、变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-06-02 | v1.0 | 初版完成 |
| 2026-06-02 | v1.1 | 实验可信度三件套（split/复现性/test 终值）+ U-Net refine mask + 分尺度 mAP + 工程细节微调 |
| 2026-06-02 | v1.1.1 | audit 第二轮：U-Net 评测改 vs 30 张人工 mask 绝对基准 + 数字/口径内部一致性同步 + LabelMe 规范 + 类名笔误修正（Crack → crazing） |

---

**Brief 结束**。详细执行计划见 `PROJECT_PLAN.md`。
