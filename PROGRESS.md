# PROGRESS

> 每天 session 启动时，Claude Code 必读的第一份文件。
> 每天 session 结束时，最后一件事就是更新本文件。

---

## 当前位置

- **Day**: Week 3 Day 15 入口
- **阶段**: 报表导出完成，待 ONNX 导出 + Polish + Demo 录制
- **上次 session 结束于**:
  - Day 14 HTML 报表 + matplotlib 可视化，1800 张 CSV 生成 0.7s，105.5KB
  - 端到端批处理→报表→浏览器打开，8/8 验收通过
  - 发现 inclusion 类 149% 假阳性 bias（模型三类混淆）
  - DEVLOG Insight #10 (HTML 报表工程决策) 记录
- **今天目标**: Day 15 ONNX 导出 + 性能基准 + Polish + Demo 视频录制

---

## 已完成

- [x] Week 1 (Day 1-5) 完成
- [x] Day 5.5 人工标注 30 张 polygon mask（LabelMe 手动完成）
- [x] Week 2 U-Net: baseline+both refined, best = baseline IoU=0.413, Insight #4 已写
- [x] Day 11 环境迁移完成
- [x] Day 12: 单图检测 + 多线程 Worker 实战完成
- [x] Day 13: 批处理 Worker + per-class threshold + 1800 张全验完成
- [x] Day 13 fix: cancel-CSV 写入 + MAX_BATCH_FILES 保护 + 11/11 补充验证
- [x] Day 14: HTML 报表 + matplotlib 可视化（4 章节 + 5 异常列表）
- [x] 1800 张端到端验证（生成 0.7s，文件 105.5KB）
- [x] DEVLOG Insight #10 (HTML 报表工程决策) 记录
- [x] Day 15 块 2: HTML 报表 Polish (4.5/5 分)
- [x] Day 15 块 1: ONNX 导出 + 一致性验证 + CPU benchmark (25 FPS)
- [x] Day 15 块 4: 简历素材清单 + 简历段落 + 面试 Q&A
- [x] DEVLOG Insight #11 (ONNX CPU 工业意义), #12 (一致性验证) 记录
- [x] Day 16: README 重写 + 7 截图 + 菜单 fix + 1800 批修复 + bat 修复
- [x] Day 16: docs/UI_DEV_PROMPTS.md 通用 GUI 方法论
- [x] GPU 推理稳态 21.5ms，crazing/inclusion 端到端验证通过
- [x] DEVLOG Insight #6 (CUDA 不自动绑定), #7 (crazing 置信度低) 记录
- [x] PyQt6 6.6.1 + Qt6 6.6.3 + Fluent 1.5.7 装齐 pytorch env
- [x] base 环境清理（-220 包）
- [x] 关闭 base 自动激活
- [x] DEVLOG Insight #5
- [x] requirements.txt + CONSTRAINTS.md 同步

---

## 当前状态

- [x] Day 1-16 全部完成 (Week 1-3)
- [x] GitHub Public publish: https://github.com/angleikun/metal-defect-detection
- [ ] Week 4 待办: 简历精修 + 求职准备 + LinkedIn (按用户节奏)
- [ ] Demo 视频: 投出简历看反馈后定夺
- [ ] GitHub Contributors 显示 "claude" → 方案 B: README 加 AI 协作致谢段

---

## 阻塞

- 无

---

## 关键决策日志

（决策点出现时追加到这里。例如：）
- Day 2: small 档 bbox 数 = 447（10.7%），但 4 类不足 30 个 → 分尺度 mAP 仅参考，不强报简历
- Day 2.5: train.txt sha256 = `26dac896fb2e21bfd89550ebf419e5ca9388e1cdc455df9a51cdc2ab895c88fa`（split 固定后填）
- Day 11: 选择产品定位 B（工业 SCADA 风）
- Day 11: 选择"迁移到 pytorch env"而非"DLL 补丁"
- Day 11: spyder 不再使用，统一改 PyCharm

---

## 今日交接信息

> 每天结束前，用 1 段话告诉明天的自己/Claude：
> 1. 今天到哪里停下的？
> 2. 明天第一件事做什么？
> 3. 有没有需要先决定的事？

Day 15 三块全部完成。HTML Polish 4.5/5 + ONNX 导出 25 FPS + 简历素材双版本。
12 条 DEVLOG Insight 汇总到 docs/RESUME_MATERIAL.md (192 行)。
简历段落 5/3 bullet 双版本 + 8 个面试 Q&A 保存到 docs/RESUME_SECTION_NEU_DET.md。
与 #1 RobotVisionSystem 错位定位（经典视觉+实时 / 深度学习+离线）。
下一步：Day 16 完整验收 + GitHub README + push。

---

## 时间记录

| Day | 计划时长 | 实际时长 | 偏差原因 |
|---|---|---|---|
| 1 | 2-3h | - | - |
| 2 | 3-4h | - | - |
| 2.5 | 1h | - | - |
| 3 | 4-5h | - | - |
| 4 | 4-5h | - | - |
| 5 | 3-4h | - | - |
| 5.5 | 3-4h | - | - |
| 6-7 | 8-10h | - | - |
| 8 | 4-5h | - | - |
| 9 | 3-4h | - | - |
| 10 | 2-3h | - | - |
| 11-15 | 17-22h | - | - |
| 16-20 | 16-21h | - | - |

**累计预算**：约 70-90 小时分散到 20 天，每天 3-5 小时。

---

## 技术债 / 未来打磨清单

以下项不影响 Day 11 验收，但需在 Day 12+ 或 Week 4 重构时处理：

1. [Day 12-13] 左侧图像列表按类别分组（当前按字母排序，crazing_* 全堆前面）
2. [Day 12-13] 状态栏增加 GPU 使用率 / 当前文件 / 推理耗时显示
3. [Day 15] 右侧 CONTROL PANEL 下方空白填入系统监控小面板（SCADA 风必备）
4. [Week 4 重构] control_panel.py 259 行，拆出 _qss.py 单独管理 QSS 样式
5. [Optional] activate_env.bat → activate_env.ps1（PowerShell 版本，可选）
6. [Day 13] 实现 per-class conf threshold（crazing 0.05 / 其他 0.20）
7. [Day 13+] 添加 GPU/显存监控显示到状态栏
8. [Day 14] 报表 HTML/PDF 生成 from CSV
9. [Day 14+] CSV → matplotlib 类别分布可视化
10. [Week 4 重构] main_window.py 430 行超出 200 行限制（与 control_panel.py 406 行一起拆）
11. [Day 15] 报表异常数颜色编码：= 0 绿，> 0 红
12. [Day 15] ✅ 报表章节标题加英文副标题
13. [Day 15] ✅ 饼图标签精简（去掉检测框数，引导看表）
14. [Week 4 重构] 把 ONNX 集成进 GUI（main_window 加 PyTorch / ONNX 后端切换）
15. [Week 4 优化] ONNX INT8 量化（可能再降 40-50% CPU 耗时）
16. [Week 4 重构] main_window.py 529 行拆分（菜单 → menu_builder.py / 回调 → menu_actions.py）

---

## 防漂移提醒

每天 session 启动时确认：

- [ ] 我现在做的事，是否在 PROJECT_PLAN.md 当前 Day 的任务清单里？
- [ ] 如果不是，我有没有先更新 PROGRESS.md 说明为什么偏离？
- [ ] 如果是临时返工/补救（比如某个 bug 拖到第二天），是否已记 DEVLOG？
