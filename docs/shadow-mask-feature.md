# GS Shadow Mask:gs-shadow 灯光对 3DGS emissive 的阴影遮罩

> 分支 `feat/shadow-mask`(commit `b8fdd8c`,基于 `feat/py-binding`)· 2026-08-05
> 状态:代码完成,117 个 Python 单测通过;**C++/shader 编译与 GPU 冒烟待构建环境验证**(本机无 cmake/Vulkan SDK/GPU)。

## 1. 背景与动机

VKGS 原有着色架构中,**阴影只乘在 NEE 光照项上,从不作用于 emissive 项**(`threedgrt_raytrace.rgen.slang` 注释明言 "the emissive term has no other occlusion")。而默认 splat 材质是纯发射(emissive=(1,1,1))——即纯发射辐射场**收不到任何投影阴影**。想给 GS 场景叠阴影只能走"物理重打光"(baseColor 抬白 + emissive 降),但那会改变烘焙外观。

本功能补齐这一缺口:**保留 GS 烘焙原貌,把 mesh 遮挡物的阴影直接"遮罩"到 emissive 输出上**——即 3dgrut playground shadow catcher 的"辐射场版":无需代理几何,阴影落在真实 GS 几何(重建 surfel)上。

## 2. 机制

一条链路(全部复用现有基建):

1. **强制 surfel 重建**:`gsShadowMask` 或 `forceSurfel` 使 `needSurfaceInfo()` 为真 → 每条主光线经 iso-opacity 深度拾取得到接收点 `isoSurfPos` + 积分法线 `integratedNormal`。
2. **gs-shadow 灯**:灯资产标记 `shadowOnly=1` 后,被 NEE(`sampleOneLightNEE`,按 `enabled==0` 先例,无偏)与全部直接光循环(deferred/mesh raster)跳过——**只投影,不照明**。
3. **阴影遮罩**:`computeGsShadowMask`(以 `computeSplatEmissiveAO` 为骨架)在 needShading 早退**之前**执行:对每盏 gs-shadow 灯 `computeLightToSurfaceVector`(SOFT 模式 + radius>0 自动做 disk 抖动软影)→ shadow ray 求可见度 → 按 `intensity×luminance` **加权平均**(重叠阴影不叠乘)→ `shadowMin + (1-shadowMin)·mask` 乘进 `emissiveContribution`。
4. **遮挡源**:默认 **mesh-only**(`traceShadowRayMesh`,透射材质投彩色影);`gsShadowMaskFromParticles` 开启后粒子也遮挡(接收 set 会自遮挡,慎用)。
5. **编译耦合**:开启 mask 经 `effectiveLightingMode()` 隐含 `LIGHTING_MODE=1`(注入点所在函数被其门控);纯发射 set 视觉不变的保证 = needShading 早退 + tonemapper 联动抵消 linear 化(与手动开灯行为一致)。副作用:mesh 转 PBR 着色、tonemapper 激活。

**语义要点**:mask 专职"烘焙发射项"的遮挡(与 emissive AO 同一作用面);needShading=1 的 set 同样生效(其 shading 项阴影仍由 NEE 负责,gs-shadow 灯被 NEE 跳过故无重复计算);不乘 N·L(烘焙 radiance 已含明暗)。

## 3. 参数参考

### 灯光标志

| 入口 | 名称 | 说明 |
|---|---|---|
| GUI 灯光面板 | **GS shadow only**(Enabled 之后) | 勾选即成 gs-shadow 灯 |
| .vkgs | `lights.assets[].shadowOnly`(int 0/1,可选键,v7 不升版本) | 缺省 0 |
| Python | `scene.add_light(..., shadow_only=True)` / `LightAsset.shadow_only` | |
| shader | `LightSource.shadowOnly`(尾部追加,offset 兼容) | |

**建议 gs-shadow 灯 `radius=0`**:恒为零噪声硬阴影;radius>0 + `shadowsMode=SOFT` 才做软影(每帧单样本,需 temporal 累积收敛)。

### 渲染选项(GUI「Lighting and Temporal」组 / CLI / .vkgs renderer 块 / Python RendererSettings)

| 参数 | 默认 | 说明 |
|---|---|---|
| `gsShadowMask` / `gs_shadow_mask` | false | 总开关(RTX 管线外置灰);改动触发 shader 重编译 |
| `gsShadowMaskMin` / `gs_shadow_mask_min` | 0.2 | 阴影下限:0=纯黑,1=无影(运行时 uniform,免重编译) |
| `gsShadowMaskFromParticles` / `gs_shadow_mask_from_particles` | false | 粒子也遮挡(自遮挡风险,tooltip 已警告) |
| `forceSurfel` / `force_surfel` | false | 独立强制 surfel 重建(深度/法线导出用,不开 mask 也可用) |

## 4. 使用方法

### GUI

1. 加载 GS 场景(ply)+ 放置遮挡 mesh;切 **RTX(2)/ HYBRID(3)/ HYBRID_3DGUT(5)** 管线。
2. 加一盏平行光/点光,灯属性勾 **GS shadow only**(radius 保持 0)。
3. Renderer > Lighting and Temporal 勾 **GS shadow mask**,调 **Shadow mask min**。
4. 应看到:GS 烘焙原貌不变 + 遮挡体投影落在 GS 上。

### CLI / .cfg

```
vk_gaussian_splatting --inputProject scene.vkgs --pipeline 3 --gsShadowMask 1 --gsShadowMaskMin 0.25
```

### Python(vkgs 包)

```python
from vkgs import Scene, Camera, LightType, Pipeline, render_scene

scene = Scene()
scene.renderer.pipeline = Pipeline.HYBRID
scene.renderer.gs_shadow_mask = True
scene.renderer.gs_shadow_mask_min = 0.25
scene.add_splats("garden.ply")
scene.add_mesh("occluder.obj", position=(0, 1.0, 0))
scene.add_light(LightType.DIRECTIONAL, rotation=(-60, 0, 0), radius=0.0, shadow_only=True)
idx = scene.add_camera_preset(Camera(eye=(3, 1.5, 2), ctr=(0, 0.5, 0)))
result = render_scene(scene, cameras=[idx], out_dir="out")
```

### 3dgrut compat shim(SHADOW_CATCHER 自动映射)

playground 脚本中的 `OptixPrimitiveTypes.SHADOW_CATCHER` 不再抛异常:发出 `CompatWarning` 并自动映射——catcher mesh 不进场景、`gs_shadow_mask=True`、为每盏普通灯克隆一份 `shadow_only=True` 副本(radius=0;原灯继续照亮 mesh)。catcher 与 GS 几何重合的地面场景下效果等价甚至更好(阴影贴合真实几何);悬空 catcher 无影(近似的边界)。差距清单 **A3 就此关闭**。

## 5. 适用范围与限制

| 项 | 说明 |
|---|---|
| 管线 | 仅 RT 管线 **2/3/5**(光栅 deferred 无 ray query 基建);RTX AS 未就绪静默 fallback 时 mask 无提示消失(GUI 置灰缓解) |
| 软影 | SOFT + radius>0 依赖 temporal 累积;拖动相机时半影有噪,静止收敛;radius=0 恒零噪 |
| 自遮挡 | `gsShadowMaskFromParticles` 开启时接收 set 自身粒子云会挡光(烘焙已含自阴影 → 双重变暗);默认 mesh-only 规避 |
| 副作用 | 开 mask 隐含 LIGHTING_MODE=1:mesh 转 PBR 着色、tonemapper 激活、FTB/surfel 重建开销(等同手动开灯) |
| stochastic 模式 | surfel 逐帧抖动 → 运动中影缘沙化,静止收敛;DLSS 可压 |
| 多灯语义 | 加权平均(遮一盏只暗一份);要"死黑"用单灯 + `gsShadowMaskMin=0` |

## 6. 验证状态与构建后检查单

已验证:117 个 Python 单测(round-trip 新键、compat 映射、灯副本几何一致性等)全绿;shader/C++ 改动逐行对照既有先例(AO 函数、enabled 跳过、宏注入模式)。

构建后待跑:
1. **编译检查点**:`parameters.cpp` 中 `gsShadowMaskMin` 的 float 参数注册——仓库无 float CLI 先例、nvpro_core2 未检出无法核签名(已保守去掉 min/max;若 registry 无 float 重载,删该行注册即可,GUI/.vkgs 路径不受影响)。
2. `pytest python/tests -m gpu`(含 `test_gs_shadow_mask_smoke`:mask on/off 对比、遮挡区更暗、非阴影区一致)。
3. 纯发射场景 mask on/off 视觉一致性(tonemapper 抵消,预期高 PSNR)。
4. GUI 手工冒烟(见 §4)。

## 7. 关键源码位置

| 内容 | 位置 |
|---|---|
| mask 函数 + 注入点 | `shaders/threedgrt_raytrace.rgen.slang`(`computeGsShadowMask`,AO 函数之后;注入在 needShading 早退前) |
| NEE 跳过 | `shaders/shading.h.slang`(`sampleOneLightNEE`) |
| 直接光循环跳过 | `shaders/deferred_shading.comp.slang` / `shaders/threedmesh_raster.frag.slang` |
| 灯光结构/上传 | `shaders/shading.h`(`LightSource.shadowOnly`)、`src/light_manager_vk.h/.cpp` |
| 渲染选项/宏 | `src/parameters.h`(prmRender + `effectiveLightingMode()`)、`src/gaussian_splatting.cpp`(GS_SHADOW_MASK 宏注入、temporal、帧同步)、`src/gaussian_splatting.h`(`needSurfaceInfo`) |
| GUI | `src/gaussian_splatting_ui.cpp`(灯光面板 checkbox;Lighting and Temporal 组) |
| .vkgs 序列化 | `src/vkgs_project_writer.cpp` / `vkgs_project_reader.cpp`(可选键) |
| Python | `python/vkgs/project.py`、`python/vkgs/compat/{convert,engine,primitives}.py` |
| FrameInfo uniform | `shaders/shaderio.h`(`gsShadowMaskMin`) |
