# feat/py-binding 分支状态汇总(暂存工作文档)

> 日期:2026-08-05 · 分支 `feat/py-binding` · 提交 `2a6048f`(基于 main @ `dfb3c79`)
> 本文档为阶段性汇总,不在 mkdocs 导航中;正式用户文档见 [python-scripting.md](python-scripting.md)。

## 一、修改情况

### 1. 新增 Python 包 `python/vkgs/`(约 7600 行,含测试与示例)

纯「文件 + 子进程」架构:Python 写 `.vkgs`(工程,JSON v7)+ 生成 `.cfg`(SEQUENCE 基准脚本)→ `--headless 1 --benchmark 1` 子进程渲染 → `--saveImage` 导出 → numpy 读回。无 pybind11/IPC。

| 模块 | 内容 |
|---|---|
| `constants.py` | 全部枚举,int 值逐一对照 shaderio.h / shading.h / parameters.h |
| `project.py` | `Scene`/`Material`/`RendererSettings` 等 dataclass 树,`.vkgs` v7 读写往返;必需字段、相对路径、Euler 度数约定均按 C++ reader/writer 源码固化 |
| `camera.py` | `Camera`(eye/ctr/up/fov/DoF);view matrix、COLMAP/INRIA(修复了 C++ 导入器丢 fov 的问题)、kaolin 转换;两侧世界系转换 `M = diag(1,-1,-1)`;GLM Euler 约定 `R = Rz@Ry@Rx`(测试中用 GLM 四元数公式对拍验证) |
| `sequence.py` | `RenderScript` → `.cfg`;强制「渲染序列 + 保存序列」成对模式(`--saveImage` 回调在序列开始即执行,捕获上一序列的收敛帧——源码验证) |
| `runner.py` | 可执行文件查找(参数 → `$VKGS_BIN` → `_bin/` → `build*/`)、子进程驱动、日志解析(移植 benchmark.py 正则)、错误检测 |
| `images.py` | PNG/HDR/RAW → numpy;按文件名后缀(`_main`/`_depth`/`_normal`…)选 buffer(数字索引会漂移,统一 `--saveImageBuffer -1`);RAW 带 16 字节头 |
| `facade.py` | `render_scene(scene, cameras, spp, buffers, …) → RenderResult`,一次进程摊销 N 相机 |
| `materials.py` | 3dgrut 同名材质因子预设(逐值抄自 engine.py:424-561);checkboard 等纹理型预设标注不支持 |
| `geometry.py` | 程序化 Quad/Sphere OBJ(对照 obj_loader.cpp 要求) |
| `video.py` | Catmull-Rom/线性/循环相机插值 → N 预设单次运行 → imageio-ffmpeg 出 MP4;兼容 3dgrut VideoRecorder 轨迹文件 |
| `compat/` | `EngineVKGS`:镜像 `Engine3DGRUT` 表面(lazy 状态 + flush-on-render);`PrimitivesVKGS`(资产注册表、autoscale 复刻)、灯光/材质/变换转换、tonemap/gamma 用 3dgrut 原公式在 Python 侧后处理实现像素级对齐;不可支持项抛带 workaround 的 `CompatWarning` |

示例:`examples/01`(建场景+多相机渲染)、`02`(环绕视频)、`03`(3dgrut checkpoint → PLY → VKGS)、`04`(playground 脚本移植演示)。

### 2. C++ 改动(两处,均为用户确认过的小改动)

1. **`--saveProject <path.vkgs>` 基准参数**(`src/gaussian_splatting_ui.cpp:123-135` + `gaussian_splatting_ui.h`):复用已有 `saveProject()`;用于 Python↔C++ round-trip 测试与 CLI 场景导出。与 saveImage 同时序(序列开始执行,捕获上一序列末状态,help 中已注明)。
2. **RTX alpha 补丁**(`shaders/threedgrt_raytrace.rgen.slang`):最终写入从 `float4(color, 1.0)` 改为写 `1 - maxComponent(transmittance)`(累积覆盖率),temporal 累积路径同步累积 alpha,fisheye 圈外像素 alpha=0。已核查全部 alpha 消费点(hybrid 回读的是 raster 输出、FTB blend 本就按 coverage 语义、tonemapper 写独立 LDR buffer),与 raster 管线语义一致,未加 define 门控。使 pipeline 2/3/5 的 `--saveImage` PNG/RAW 携带可用 opacity。

### 3. 文档与工程

- `docs/python-scripting.md` + mkdocs 导航条目;`python/README.md`;`.gitignore` 增补 venv/egg-info。
- 测试:**112 个单元测试通过**(往返/golden/相机数学/cfg 快照/runner/兼容层/视频),另有 `-m gpu` 冒烟测试(无可执行文件时跳过)。

### 4. 验证状态(如实记录)

| 项 | 状态 |
|---|---|
| `.vkgs` 往返、cfg 生成、相机/Euler 数学、compat 转换 | ✅ 单元测试 + 对照 C++ 源码验证 |
| 三个官方 sample 工程 golden 往返 | ✅ |
| C++ 两处改动编译 | ⚠️ **未验证**——本机无 cmake/Vulkan SDK,nvpro_core2 子模块未检出;已做语法/风格一致性核查 |
| GPU 冒烟(真实渲染、saveImage 成对时序经验性确证、非黑图断言) | ⚠️ **未验证**——本机无 GPU;构建后运行 `python -m pytest python/tests -m gpu` |
| 相机对齐(同 .ply 两侧渲染同位姿对拍) | ⚠️ 数学已测,**像素级对拍待 GPU 环境** |

## 二、缺失能力汇总(相对 3dgrut playground 脚本)

### A 类 — 不做重大渲染器改造无法支持

| # | 功能 | 在 3dgrut 中的角色 | 缺失的影响 | Workaround |
|---|---|---|---|---|
| A1 | `raygen()` / `RayPack` / 自定义逐像素光线 | 管线第一级:把相机转成显式光线张量再喂 tracer;用户可自造任意光线(全景/正交/自定义畸变、稀疏查询、把追踪当几何查询 API) | 「非标准投影」和「追踪当查询用」的脚本无法移植。注意:DoF/AA 效果本身不受影响(VKGS shader 内置 DoF,AA 用 temporal 累积) | 全景:cube face 六面渲染 + Python 重投影;深度查询:读 `_depth` buffer |
| A2 | `render_pass()` 渐进渲染 | 把 SPP/DoF 采样摊到多次 pass:先粗图后逐 pass 累积精化,支撑交互视口与可打断收敛 | 无增量预览、无中途打断、无逐 pass 收敛观察;每次 render = 进程启动+场景加载(秒级)。**离线批量渲染基本无影响**(质量旋钮 = `spp`→`sequenceframes`) | shim 中 `render_pass` = 一次全质量渲染+缓存;交互场景直接用 VKGS GUI |
| A3 | Shadow catcher(`SHADOW_CATCHER`/`shadow_min`) | 地面只收阴影不遮内容 | 合成类场景(物体+接地阴影叠在实拍/GS 地面上)做不了同款效果 | 漫反射地板(观感不同)或双渲染 Python 合成 |
| A4 | 矩形面光源(AREA + tangent_u/v) | 软盒光 | VKGS 仅 directional/point/spot | 发光 quad mesh(path-traced 管线,较噪)或 point+radius |
| A5 | `scene_mog` 张量级场景访问 | 高斯模型本体(positions/rotation/scale/density/SH 的 GPU torch 张量):裁剪、删点、改色、拼接、逐高斯动画、autograd 微调;改后 `rebuild_bvh()` 毫秒级生效 | 逐高斯实时编辑没有(任何编辑=改写 PLY+整场景重载,秒级一轮);梯度工作流完全不可行。**缓解**:刚体操作走实例 transform 原生支持;多模型组合反而是 VKGS 超集(3dgrut 单 mog) | numpy 改写 PLY + 重载;刚体/复制用多实例 |
| A6 | 额外 AOV(`hits_count`、逐光线距离) | 调试/分析输出 | saveImage 仅 main/aux1/normal/depth/ldr/dlss | `renderer.visualize` 模式把 AOV 渲进 `_main`(单 AOV、伪彩) |
| A7 | OptiX denoiser(含 `rgb_buffer` 降噪前后对) | AI 降噪 | 仅 DLSS-RR(需 USE_DLSS 构建 + RTX 硬件),算法不同 | DLSS 可用时映射,否则加大累积帧数 |
| A8 | `shadow_spp` / `shadow_min` | 全局阴影采样数/下限 | 无对等参数 | light `radius`、`shadowMode`、`particleShadow*` 近似 |

**一句话判断**:脚本若只是「加载 → 摆场景 → 设相机/灯光/材质 → 批量出图/出视频」,A 类不构成影响;一旦把渲染器当**可微分/可查询/可实时迭代的库**用(自造光线、收敛循环、张量编辑),就落在 A 类,只能走降级路径。

### B 类 — 已桥接或文档化解决

| 功能 | 处理 |
|---|---|
| `.pt`/`.ingp` checkpoint | 按决策不做自动桥;`examples/03` 提供 `model.export_ply()` 导出脚本(API 已对照 3dgrut 源码验证) |
| `VideoRecorder` 视频轨迹 | `vkgs.video` 纯 Python 重实现,轨迹文件互通 |
| 程序化几何(Quad/Sphere) | `vkgs.geometry` 生成 OBJ |
| splat 编辑 | PLY numpy 改写 + 重载(慢路径) |
| 指标(PSNR/SSIM/LPIPS) | Python 侧对输出图计算(SSIM/LPIPS 需额外依赖) |
| 逐帧物体动画 | 实例 transform 不可 cfg 设置 → 每帧一次工程+运行;相机动画高效(预设) |

### C 类 — 可用但语义/质量不同(均有 CompatWarning)

GLASS/MIRROR BSDF 模型差异(折射色调/TIR/焦散可见差异)· 软阴影(角半径 vs 世界半径,`tan(θ)·scale` 启发式)· AA(temporal 累积/超采样替代 MSAA 子像素抖动,收敛结果可比、逐样本不可复现)· tonemap/gamma(**已用 HDR 读回 + 3dgrut 原公式做到像素级对齐**)· DLSS-RR vs OptiX denoiser · fisheye 投影公式待 GPU 环境对拍 · 点光衰减模型差异(待一次性标定)· `envmap_offset` 极角分量仅近似 · DoF 仅 pipeline 2/4/5 生效 · 随机管线 RNG 不同,仅统计对齐 · 单进程单分辨率(无逐相机分辨率)。

## 三、后续待办

1. 用户侧构建(`nvpro_core2` 子模块 + cmake + Vulkan SDK)→ 验证两处 C++ 改动编译。
2. `python -m pytest python/tests -m gpu`:真实渲染冒烟 + saveImage 成对时序经验性确证。
3. 同一 `.ply` + 同一转换位姿在 3dgrut / VKGS 两侧渲染对拍(相机对齐是 compat 层最高风险项)。
4. C++ round-trip 测试补全:Python 写 `.vkgs` → app `--saveProject` 回写 → diff。
5. 视情况:点光强度标定常数、fisheye 投影对拍、DLSS 映射验证。
