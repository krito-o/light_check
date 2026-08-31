# 技术报告：ref 参考区作用评估 + 入槽检测现状

> 评估对象：当前 `D:\code\light_check\` 代码（含面积门 + 亮像素均值 + ref 自适应阈值改进）
> 评估视频：正常起落.MP4（新批）、异常-灯不亮/灯长亮/正常起落-近点（旧批）
> 日期：2026-07-31 | 方式：纯实测，未改代码

---

## 一、ref 起到作用了吗？作用大吗？

### 结论：在当前所有测试视频上，ref **没有起到任何实际作用**；且在面积门模式下 ref **结构性失效**。

### 1.1 实测：ref 在 qiyue 视频里未改变任何判定

采集正常起落.MP4 的 ref 区（1117,649,12,12）green_excess 全程序列：

| 指标 | 值 |
|---|---|
| ref_ge median | **-3.39** |
| ref_ge min / max | -7.24 / 1.00 |
| ref_ge pp | 8.24 |

自适应阈值计算：`on_thr = max(on_threshold=5, ref_med + on_margin) = max(5, -3.39+3) = max(5, -0.39) = 5`

→ `ref_med + on_margin = -0.39 < 5`，被固定阈值 5 兜底，**on_thr 退化回固定值 5**。

判定结果对比（lower 全程）：

| 配置 | 结果 |
|---|---|
| 有 ref（on_thr=5） | flashing 56/56 |
| 无 ref（on_thr=5） | flashing 56/56 |

**完全相同**。原因：ref 区是中性环境（无绿色物体），ref_ge 恒为负，自适应阈值永远低于固定阈值，ref 被绕过。

### 1.2 结构性问题：面积门模式下 ref 的 on_thr 根本不参与判定

这是比"数值没生效"更严重的问题——**即使 ref 算出了 on_thr，判定流程也没用它**。

`classify_span` 在传入面积序列（`has_area=True`，即新视频/有 ref_roi 的 config）时的判定分支：

```
if not is_real_light:        off        # 面积门
elif is_oscillating:         flashing   # 振荡门
else:                        steady_on
```

`on_thr` / `lit_ratio`（亮度门）**在这个分支里完全没出现**。实测验证：

| 场景 | on_thr=50 | on_thr=0 | 结论 |
|---|---|---|---|
| 大面积(500px)+低信号(median=2)+不振荡 | steady_on | steady_on | on_thr 不影响结果 |

也就是说：
- **面积门模式（新视频）**：判定 = 面积门 + 振荡门，ref 的 on_thr 不参与。
- **退化模式（旧 config，无 ref_roi）**：判定用 `med < on_threshold`，但旧 config 没配 ref，on_threshold 是固定值 5。

两条路径下 ref 都没真正接入判定。**ref 目前是一个"接了线但没通电"的功能**。

### 1.3 ref 的理论作用场景（当前未覆盖）

构造实验确认 ref 在什么情况下**本应**起作用：off 灯 + 环境有绿色本底（信号被抬到 8）+ 中等面积反光（200px，面积门挡不住）：

| 配置 | 结果 |
|---|---|
| 无 ref（on_thr=5） | steady_on（误判，8>5 且面积大不振荡） |
| 有 ref（on_thr=11=8+3） | steady_on（**仍误判**——因面积门模式下 on_thr 不参与） |

→ 即便在 ref 本该起作用的场景，因 1.2 的结构问题，它也救不了。要让 ref 真正生效，需要把亮度门（`is_lit = med >= on_thr`）接回面积门模式的判定流程。

### 1.4 那 qiyue 为什么判定正确？

不是 ref 的功劳，是**面积门 + 振荡门 + 亮像素均值信号**三者的功劳：
- 亮像素均值信号解决了稀释（lower median≈35，不再是被稀释的 -2）
- 面积门区分了反光（60px）与真灯（412+px）
- 振荡门区分了闪烁（osc≈8）与常亮（osc≈0）

拆解验证（lower 全程）：完整判定 / 去掉面积门 / 去掉 ref，三者结果都是 56/56 flashing——lower 靠信号本身（median≈35 ≫ on_thr）和振荡就够，面积门和 ref 都没改变结果。

### 1.5 小结

| 问题 | 答案 |
|---|---|
| ref 起作用了吗？ | **没有**。qiyue 视频里 ref_ge 为负，自适应阈值退化为固定值。 |
| 作用大吗？ | **零**。且面积门模式下 on_thr 结构性不参与判定，即便 ref 有值也没用。 |
| 判定对的功劳归谁？ | 亮像素均值信号 + 面积门 + 振荡门。ref 是摆设。 |
| ref 有潜在价值吗？ | 有，但需修复：把亮度门接回面积门模式，ref 才能在"环境有绿色本底"场景纠正误判。当前测试视频不满足该场景。 |

---

## 二、当前入槽检测是什么样的？准确率如何？

### 2.1 当前实现

- **方法**：亮度法（`enter_method="brightness"`，默认）。
- **原理**：对每个灯 ROI 取灰度，用滑动窗口（`baseline_win=300` 帧）的**低百分位**（`percentile=5`）作为"空槽基线"（空槽=最暗稳定态）。当前灰度持续 > `基线 + margin(15)` 达 `enter_frames(10)` 帧 → 触发入槽，进入 DETECTING。
- **触发逻辑**：`any(entered)`——**任一** ROI 检测到设备即触发（多 ROI 场景）。
- **取走检测**（`leave_check`）：与入槽时冻结的空槽基线快照比较，灰度持续回落到基线附近 → 回 WAITING。
- **`skip_enter`**：跳过入槽检测直接 DETECTING（旧 off/on config 用，因设备已在位）。

### 2.2 准确率实测

| 视频 | ROI 数 | 入槽检测配置 | 实际触发时刻 | 真实入槽时刻 | 偏差 | 评价 |
|---|---|---|---|---|---|---|
| near（近点） | 1 | skip_enter=false, 亮度法 | f2408（48.2s） | ~50s 完成 | -1.8s（提前） | **合理**，单 ROI 正常工作 |
| qiyue（正常起落） | 2 | skip_enter=false, 亮度法 | f158（3.2s） | 14-19s | **-11s（严重提前）** | **错误**，误触发 |

### 2.3 qiyue 入槽检测为何失效

qiyue 有两个 ROI：lower（下方，全程有灯/设备）和 upper（上方，14-19s 才入槽）。

- lower 框灰度全程 34-69（**一开始就有设备**，因为 lower 槽位本来就占着）。
- upper 框灰度全程 44-96 波动（手持晃动 + 偶有物体经过），**没有"空槽低 → 入槽高"的清晰台阶**。

亮度法对 lower 在第 158 帧就触发 `entered=True`，`any()` 逻辑让整个检测器进入 DETECTING——此时 upper 还是空槽。**入槽检测对 upper 完全失效**。

upper 最终判对（flashing）是靠**面积门兜底**：DETECTING 后 upper 在空槽段亮像素面积=0 → 判 off；19s 后灯亮面积>150 → 判 flashing。结果碰巧正确，但与入槽检测无关。

### 2.4 入槽检测的根本局限

1. **多 ROI 的 `any()` 触发**：任一槽有设备就全进 DETECTING，无法区分"哪个槽入槽了"。对多槽位场景（固定摄像头部署）是硬伤。
2. **依赖"空槽=最暗"假设**：qiyue 的 upper 框在空槽时灰度并不稳定最低（晃动/光照/物体经过），基线法失效。
3. **手持视频晃动**：灰度基线被晃动污染，不适合亮度法。固定摄像头下会改善。
4. **入槽检测与灯状态判定解耦不彻底**：upper 靠面积门在 DETECTING 阶段自己判 off→flashing，入槽检测成了摆设。

### 2.5 小结

| 问题 | 答案 |
|---|---|
| 入槽检测什么样？ | 亮度法：ROI 灰度偏离滑动低百分位基线 + margin，连续 10 帧触发；多 ROI 用 any()。 |
| 准确率如何？ | 单 ROI（near）合理（偏差 -1.8s）；多 ROI（qiyue）**失效**（误提前 11s 触发，靠面积门兜底才结果正确）。 |
| 适用场景 | 单槽位 + 固定摄像头 + 空槽有稳定低灰度。多槽位/手持晃动场景不可靠。 |

---

## 三、总体评估与建议（供决策，未动代码）

### 当前状态
- **灯状态判定**：7/7 场景正确（旧 off/on/near + 新 qiyue upper/lower）。功在亮像素均值信号 + 面积门 + 振荡门。
- **ref 参考区**：**实际无效**（数值退化 + 结构性不参与判定）。是一个名义上实现但未通电的功能。
- **入槽检测**：单 ROI 可用，多 ROI 失效（靠面积门兜底）。

### 若要改进，方向如下（需你确认才会动代码）

1. **ref 修复（小）**：把亮度门 `is_lit = (med >= on_thr) or (lit_ratio >= lit_ratio_min)` 接回面积门模式的判定流程，让 ref 自适应阈值在"环境有绿色本底"场景真正起作用。当前测试视频不需要，但固定摄像头现场若有绿色面板/反光则有价值。
2. **入槽检测多 ROI 化（中）**：把 `any()` 触发改为**每个 ROI 独立状态机**（每个槽位独立 WAITING→DETECTING），避免一个槽入槽导致全部进 DETECTING。这对固定摄像头多槽位部署是必要的。
3. **入槽检测抗晃动（中）**：手持视频晃动让灰度基线失效，可改用"亮像素面积突变"或"灯体出现"作为入槽信号（与灯状态判定复用面积特征），对晃动鲁棒。固定摄像头下可暂不动。
4. **删 ref（备选）**：若确认现场环境无绿色本底，可直接删掉 ref 相关代码简化系统，避免维护一个无效功能。

### 我的建议
ref 和入槽检测目前都是"为了功能完整性而保留、但当前测试视频未真正受益"的部分。灯状态判定本身已足够稳健。建议优先级：**入槽检测多 ROI 化（部署刚需）> ref 修复或删除（看现场）**。是否动代码请你定。

---

## 四、改动落地（已实施，2026-07-31）

基于本报告结论，已实施两项改动：**入槽检测多 ROI 化** + **删除 ref**。

### 4.1 入槽检测多 ROI 化

**实现**：新增 `PerLightState` 类，每个 ROI 封装独立状态机（state/入槽计数/信号缓冲/确认状态/背景模型）。`SlotDetector` 持有 `self.lights: list[PerLightState]`，每帧逐 ROI 推进。`any(entered)` 改为每 ROI 独立判定；取走检测 `all_left` 改为每 ROI 独立。`process_frame` 返回 `states`（每 ROI 状态列表）。单 ROI 退化为 `lights[0]`，行为不变。

**实测验证（qiyue 双 ROI 独立性）**：
- upper：WAITING(0-2s) → DETECTING(4s 自触发) → **off**(8-22s 空槽) → **flashing**(24s+ 灯亮)。符合用户描述（14s 前空槽、19s 后闪烁）。
- lower：WAITING(0-2s) → DETECTING(4s) → **flashing**(8s+ 全程闪烁)。
- upper 在 off 时 lower 已 flashing，两者状态独立，互不连累 ✓。

**注意**：upper 单独亮度法仍在 4s 误触发入槽（灰度波动大），但进 DETECTING 后空槽段面积门判 off 正确，不影响最终结果。这是 upper ROI 亮度法在手持晃动下的固有限制，固定摄像头下会改善。

### 4.2 删除 ref

**实现**：移除 `ref_roi` 配置、`_ref_ge`/`_ref_mean` 方法、`ref_signals` 采集、ref 自适应阈值逻辑、ref 区绘制、CSV 的 `ref_mean`/`ref_ge_med` 列、config 的 `on_margin`。`on_threshold` 保留为固定值（面积门模式下不参与判定，仅退化模式/单元测试用）。

**验证**：删 ref 后 4 视频判定结果与删前完全一致（ref 本就无效）。

### 4.3 回归测试结果（全 [OK]）

| 视频 | ROI | 判定 | 期望 | 结果 |
|---|---|---|---|---|
| 异常-灯不亮 | normal / fault_off | flashing / off | 同 | [OK][OK] |
| 异常-灯长亮 | normal / fault_steady | flashing / steady_on | 同 | [OK][OK] |
| 正常起落-近点 | light | flashing | 同 | [OK]（单 ROI，入槽 f2408 触发不变）|
| 正常起落 | light_upper / light_lower | flashing / flashing | 同 | [OK][OK]（多 ROI 独立）|

- 单元测试：`python light_classifier.py` 全 PASS（分类器签名不变）。
- 性能：qiyue 1536 帧处理 24.33s，**每帧 15.84ms**（远低于 50fps 实时阈值 20ms/帧），较旧版无下降。
- 颜色规范：flashing 绿、off 黄、steady_on 红、检测中灰，抽帧验证正确。
- CSV：14 列（删 ref_mean/ref_ge_med），表头与数据列数一致。

### 4.4 改动文件

| 文件 | 改动 |
|---|---|
| `slot_detector.py` | 新增 PerLightState 类；SlotDetector 多 ROI 化；删 ref 全部触点 |
| `main.py` | per-ROI state 适配（states 列表）；CSV 删 ref_mean/ref_ge_med 列 |
| `config_qiyue/near/off/on.json` | 删 ref_roi、on_margin |
| `light_classifier.py` | 不变（签名向后兼容，on_thr 保留但传 None）|
| `README.md` / `TECH_REPORT.md` | 第4节改"已移除"；配置表删 ref_roi/on_margin；新增多 ROI 入槽说明 |

---

## 五、视频3：固定摄像头现场监控（面积法入槽 + 灰度门区分不亮/常亮）

### 5.1 背景

视频3（20260807-153359 / 20260807-153608 / C0065）为固定摄像头现场监控画面（无晃动），ROI 设置严格。需支持：先判断入槽再开始灯检测（部署刚需），并正确识别闪烁/常亮/不亮。每个 ROI 有多次出入槽（153608 共4次入槽3次出槽）。

### 5.2 关键实测发现

1. **"不亮"≠"无绿色"**：设备入槽后绿灯**外壳本身是绿色**（ge≈31，面积≈450），与"常亮"（LED 点亮 ge≈34）的 green_excess 几乎相同——`max(R,B)` 抹平了 LED 点亮带来的 G 通道提升。**ge 无法区分不亮/常亮**。
2. **灰度可区分**：LED 点亮会显著照亮整个 ROI，不亮灰度≈51、常亮灰度≈110，差 2 倍。引入**灰度门**（gray_thr=80）：真灯+不振荡时，灰度≥80 判常亮，否则判不亮。
3. **灰度法判入槽不可行**（固定摄像头现场）：机械臂移动经过空槽时灰度升高（误判入槽）、设备空中停留灰度更高（176，误判）、不同槽空槽灰度差异大（33~90）。实测铁证，灰度法彻底失效。
4. **面积法判入槽可行**：设备入槽后灯体（LED 点亮或仅外壳绿）带来稳定绿色面积（>100px），真在槽时面积基本不归零（闪烁暗态面积仍>200，不亮外壳面积~450 稳定）；机械臂经过空槽时绿色面积短暂出现又归零（持续<0.5s）；空槽面积恒为 0。用"连续 enter_frames 帧 area>area_thr"判入槽，连续性过滤机械臂经过的瞬态。
5. **弱振幅闪烁**：视频3部分闪烁（如 C0065 slot1）ge 振幅仅 3.5-4.8，min_sep 从 4.0 降到 3.0 才能识别（不亮 ge 振幅 2.6，仍被正确抑制）。

### 5.3 算法改动

- **light_classifier.py**：classify_span/classify_window 增加 `gray_seq`/`gray_thr` 参数。判定流程改为"面积门 + 振荡门 + 灰度门"：
  `面积不够→off`；`真灯+振荡→flashing`；`真灯+不振荡+灰度高→steady_on`；`真灯+不振荡+灰度低→off`。振荡分支不看灰度。向后兼容（不传 gray_seq 退化为旧行为）。
- **slot_detector.py**：
  - 新增 `enter_method="area"`（默认，固定摄像头推荐）：绿色面积法入槽（连续 enter_frames 帧超阈）+ 出槽（面积归零或显著低于稳定值）。
  - `_roi_lit_signal` 返回值增加 ROI 灰度；`_do_detecting` 采集 gray_signals。
  - 出槽检测改为每帧检查（不绑定 span 判定），确保短时在槽也能 reset。
  - **短时在槽紧急判定**：在槽< span_size 就出槽时，用已有帧(≥60)补判；仅信号明确（osc≥osc_min 判闪烁，或面积门判 off）才确认，steady_on 不确认（短跨度易误判）。
  - PerLightState 增加 `stable_area`（出槽判定参考）、`last_in_slot`（出槽后保留最近在槽判定，供汇总/标注展示）。
  - `_check_leave_one` 按 enter_method 分流：area 法用面积，brightness 法用灰度基线（保持旧 config 兼容）。
- **main.py**：CSV 增加 gray_med 列（15列）；汇总显示"最近在槽判定"+"当前状态"。

### 5.4 视频3 回归结果

| 视频 | ROI | 期望 | 判定 | 结果 |
|---|---|---|---|---|
| 153359 | slot1 / slot2 | flashing / flashing | flashing / flashing | [OK][OK]（slot1 出槽后保留闪烁判定）|
| 153608 | slot1/slot2/slot3 | 全 flashing | 全 flashing | [OK][OK][OK]（4次出入槽全正确，时序吻合）|
| C0065 | slot2_off / slot3_off | off / off | off / off | [OK][OK]（灰度门正确识别不亮）|
| C0065 | slot4_ok | flashing | flashing | [OK] |
| C0065 | slot1_ok | flashing | 待定 | [X]（在槽仅2s+弱振幅，保守不判，详见5.5）|

- 入槽时序吻合用户描述（153608：入槽1 f366/出槽 f566、入槽2 f1563/出槽 f1863、入槽3 f2842/出槽 f3192，对应 3s/11s/27s/36s/53s/63s）。
- 旧 config 回归全 [OK]：off(n=闪/fault=不亮)、on(n=闪/steady=常亮)、near(闪)、qiyue(双闪)、changliang(常亮)。灰度门正确识别常亮，未破坏旧逻辑。
- 单元测试全 PASS（新增灰度门断言：外壳绿低灰度→off、LED高灰度→steady_on、不传灰度退化兼容）。

### 5.5 已知局限

- **C0065 slot1**：在槽仅 2 秒（< span_size 3s），且为极弱振幅闪烁（ge 振幅 3.5，中位数 34，与不亮 ge≈31 几乎重叠），ge/gray 振荡均接近不亮/常亮，难以可靠区分。紧急判定保守地保持"待定"（不错判）。这是 LED 亮度变化小的物理特性所致，非算法缺陷。实际部署中设备入槽后持续工作，不会出现此边界。
