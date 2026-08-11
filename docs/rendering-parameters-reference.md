# VKGS 渲染参数完整参考

本文档穷尽罗列 VKGS(vk_gaussian_splatting)的**所有可选渲染参数**,涵盖命令行(CLI)、GUI 面板、benchmark/`.cfg` 脚本动作,以及可持久化到 `.vkgs` 工程文件的字段。所有默认值、取值范围、枚举定义均直接引用自源码,并标注 `file:line`。

标识符、枚举名、flag、路径均保留英文;说明文字用中文。

---

## 0. 总览:管线 × 功能适用性

VKGS 有 6 条渲染管线(`shaders/shaderio.h:103-108`),通过 `--pipeline` 或 GUI「Pipeline」选择:

| 值 | 宏 | GUI 标签 | 类型 | 说明 |
|----|----|---------|------|------|
| 0 | `PIPELINE_VERT` | Raster 3DGS vertex shader | 纯光栅 | 3DGS,顶点着色器 |
| 1 | `PIPELINE_MESH` | Raster 3DGS mesh shader | 纯光栅(默认) | 3DGS,mesh 着色器,需 `VK_EXT_mesh_shader` |
| 2 | `PIPELINE_RTX` | Ray tracing 3DGRT | 纯光追 | 3DGRT,需 RT 三件套 |
| 3 | `PIPELINE_HYBRID` | Hybrid 3DGS+3DGRT | 混合 | 主光线光栅(3DGS)、次光线光追 |
| 4 | `PIPELINE_MESH_3DGUT` | Raster 3DGUT mesh shader | 纯光栅 | 3DGUT(无迹变换),mesh 着色器 |
| 5 | `PIPELINE_HYBRID_3DGUT` | Hybrid 3DGUT+3DGRT | 混合 | 主光线光栅(3DGUT)、次光线光追 |

**判定辅助函数(源码内部)**:
- **光栅管线 (raster)**:0、1、3、4、5(即除纯 RTX=2 外均有光栅路径;`isRasterPipelineActive()`)。
- **光追管线 (RTX)**:2、3、5(`isRtxPipelineActive()`,即「RTX 或 hybrid」)。
- **DLSS 支持管线**:与 RTX 管线一致(2、3、5)。

功能适用性速览:

| 功能 | 适用管线 | 说明 |
|------|---------|------|
| 排序(sorting)、剔除(culling)、FTB、Splat scale | 0/1/3/4/5(光栅) | 见第 5 节 |
| Kernel degree / Projection method | 4、5(3DGUT);光追时 2/3/5 | 3DGS 恒用 Eigen 投影 |
| 路径追踪、光追阴影、AO、particle depth | 2、3、5(光追) | 见第 6 节 |
| 光照(lighting) | 全部(光栅=直接光,光追=路径追踪) | 见第 7 节 |
| 阴影(shadows) | 仅 2/3/5(光追) | `SHADOWS_MODE` |
| 景深(DoF) | 仅 2(`PIPELINE_RTX`)、4(`PIPELINE_MESH_3DGUT`)、5(`PIPELINE_HYBRID_3DGUT`) | 见 `gaussian_splatting_ui.cpp:4516-4517` |
| Auto focus(DoF) | 仅 2、5(需光追提供表面距离,`supportsAutoFocus()`={PIPELINE_RTX, PIPELINE_HYBRID_3DGUT}) | `gaussian_splatting.h:212-215` |
| 时间累积(temporal sampling) | 光追(DoF/Monte-Carlo 自动触发) | 见第 8 节 |
| DLSS-RR | 2、3、5 + `USE_DLSS` 构建 + NVIDIA 硬件 | 见第 12 节 |
| Visualize 模式(Depth/Normal/Clay 等) | 仅光追/hybrid(`gaussian_splatting_ui.cpp:3437`) | 见第 4 节 |
| 色调映射 tonemapping / 环境 environment | 全部(后处理/共享) | 见第 10、11 节 |

---

## 1. 应用级 / 场景加载(CLI)

这些参数在 `src/main.cpp` 与 `src/parameters.cpp`(`registerCommandLineParameters`)、`src/gaussian_splatting_ui.cpp`(`registerParameters`)中注册。应用创建期参数(窗口、headless 等)一经启动不可再由 benchmark 脚本改变。

| 参数 (C++ 变量) | CLI flag | 默认值 | 取值范围 | 作用与调优建议 | 适用 |
|------|----------|--------|---------|---------------|------|
| `appInfo.windowSize` | `--size W H` | 平台默认 | 两个整数 | 创建窗口尺寸 (`main.cpp:56`) | 全局 |
| `appInfo.vSync` | `--vsync` | 平台默认 | bool | 垂直同步。benchmark 模式强制关闭 (`main.cpp:57,275`) | 全局 |
| `appInfo.headless` | `--headless` | false | bool | 无窗口运行(无 swapchain、无显示),用于离屏导出 (`main.cpp:58`) | 全局 |
| `appInfo.headlessFrameCount` | `--headlessFrameCount` | — | int | headless 渲染帧数;开启 `--benchmark` 时被忽略(改由 sequencer 结束循环)(`main.cpp:59-60,229-232`) | headless |
| `vkSetup.verbose` | `--verbose` | false | bool | Vulkan 上下文详细输出 (`main.cpp:61`) | 全局 |
| `vkSetup.enableValidationLayers` | `--validation` | false | bool | 开启验证层 (`main.cpp:62`) | 全局 |
| `benchmarkMode` | `--benchmark` | false | bool | 开启基准测试:禁止异步加载、关闭 vsync、隐藏菜单 (`main.cpp:63,228,272-276`) | 全局 |
| `vkSetup.forceGPU` | `--forcegpu` | — | int(GPU ID) | 强制使用指定 GPU (`main.cpp:64`) | 全局 |
| `prmScene`(队列回调) | `--inputFile <path>` | — | `.ply`/`.spz`/`.splat`,**可重复** | 加载模型文件。第一个文件重置场景,后续文件叠加 (`parameters.cpp:94-114`) | 全局 |
| `prmScene.projectToLoadFilename` | `--inputProject <path>` | — | `.vkgs` | 加载 vkgs 工程文件 (`parameters.cpp:120-123`) | 全局 |
| `prmScene.enableDefaultScene` | `--loadDefaultScene <0/1>` | true | 0=禁用 | 无 ply 输入时是否加载内置默认场景;仅在 `WITH_DEFAULT_SCENE_FEATURE` 构建下存在 (`parameters.cpp:116-117`,`parameters.h:56`) | 全局 |

注:`--inputFile` 通过后缀(`.ply`/`.spz`/`.splat`)自动触发,也可直接把文件作为位置参数传入。

---

## 2. 渲染管线选择

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmSelectedPipeline` | Pipeline / Global Settings(菜单栏也有) | `--pipeline` | `PIPELINE_MESH`=1 (`parameters.cpp:31`) | 0-5(见第 0 节) | 选择渲染方法。缺少 mesh/RT 扩展的项在 GUI 中置灰 (`parameters.cpp:137`,`gaussian_splatting_ui.cpp:3356,264-269`) | 全局 |

---

## 3. 数据格式与存储(含 RTX 加速结构)

`VramDataParameters prmData` 与 `RtxVramDataParameters prmRtxData`(`src/parameters.h:86-127`)。改动会触发数据重生成/BLAS 重建。持久化于 `.vkgs` 的 `splatsGlobals` 段(`vkgs_project_writer.cpp:288-301`)。

### 3.1 数据格式(所有管线共享)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmData.shFormat` | SH format | `--shformat` | `FORMAT_UINT8`=2 (`parameters.h:88`) | `FORMAT_FLOAT32`=0 / `FLOAT16`=1 / `UINT8`=2 (`shaderio.h:40-42`) | SH 系数存储格式,精度 vs 显存 (`parameters.cpp:126`) | 全局 |
| `prmData.rgbaFormat` | RGBA format | `--rgbaformat` | `FORMAT_UINT8`=2 (`parameters.h:89`) | 同上;fp32=16B/splat,fp16=8B,uint8=4B | RGBA 颜色+alpha 存储格式 (`parameters.cpp:127`,`gaussian_splatting_ui.cpp:4226-4231`) | 全局 |
| `splatSet->dataStorage`(每 splat set) | Storage(单个 splat set 属性) | 无 CLI | `STORAGE_BUFFERS`(资产默认) | `STORAGE_BUFFERS`=0 / `STORAGE_TEXTURES`=1 (`shaderio.h:30-31`) | 位置/颜色/协方差/SH 存 buffer 还是 texture(`gaussian_splatting_ui.cpp:4403-4407`)。此为 per-splat-set,非全局 | 光栅为主 |

### 3.2 RTX 加速结构与粒子几何

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRtxData.compressBlas` | BLAS Compaction | `--compressBlas` | true (`parameters.h:113`) | bool | 压缩 BLAS,省显存 (`parameters.cpp:134`,`gaussian_splatting_ui.cpp:4279`) | 光追 |
| `prmRtxData.useAABBs` | Use AABBs / Particle format=AABB+parametric | `--useAABBs` | false (`parameters.h:116`) | bool | 用 AABB + 参数化相交着色器代替 icosahedron 网格 + 硬件三角求交。**不可与 `useTlasInstances=0` 组合**;会强制 `useTlasInstances=1` (`parameters.cpp:128-129`,`gaussian_splatting_ui.cpp:4253-4262`) | 光追 |
| `prmRtxData.useSpheres` | Particle format=Sphere (NV) | `--useSpheres` | false (`parameters.h:119`) | bool | 用球体图元(`VK_NV_ray_tracing_linear_swept_spheres`)。与 `useAABBs` 互斥,强制 `useTlasInstances=1`;设备不支持时置灰 (`parameters.cpp:130-131`,`gaussian_splatting_ui.cpp:333`) | 光追 |
| `prmRtxData.useTlasInstances` | Use TLAS instances | `--useTlasInstances` | true (`parameters.h:124`) | bool | true=每 splat 一个 TLAS 实例 + 单位小 BLAS(共享 TLAS 模式);false=单 TLAS 项 + 巨型 BLAS。`useAABBs`/`useSpheres` 开启时强制 true (`parameters.cpp:132-133`,`gaussian_splatting_ui.cpp:4265-4276`) | 光追 |
| `prmRtxData.billboardBoundingMode` | Billboard bounding mode | `--rtxBillboardBounding` | `eBillboardBoundingFitted`=0 (`parameters.h:126`) | 0=Fitted / 1=Uniform / 2=Uniform3/4 / 3=Uniform2/3 / 4=Uniform1/2 / 5=Uniform1/3 / 6=Uniform1/4 / 7=Optimal (`parameters.h:96-106`,`gaussian_splatting_ui.cpp:426-433`) | billboard 模式下 TLAS 实例包围盒的缩放方式:Fitted 最快但可能漏各向异性 billboard;Optimal=逐轴 (max+s_i)/2,推荐匹配光栅质量 (`parameters.cpp:165-166`,`gaussian_splatting_ui.cpp:3981-3993`) | 光追 billboard |
| `parametric`(派生自 useAABBs/useSpheres) | Particle format | 无(派生) | Icosahedron | `PARTICLE_FORMAT_ICOSAHEDRON`=0 / `PARAMETRIC`=1 / `SPHERE`=2 (`shaderio.h:153-155`) | GUI 下拉,内部映射到 useAABBs/useSpheres (`gaussian_splatting_ui.cpp:3916-3943`) | 光追 |

**Particle format 硬件说明**:UI 标签 `PARTICLE_FORMAT_*` 未直接进 shader,shader 用 `RTX_USE_AABBS`/`RTX_USE_SPHERES` 编译宏(`shaderio.h:150-155`)。

---

## 4. 通用渲染(SH / alpha / scale / visualize)

`RenderParameters prmRender`(`src/parameters.h:162-193`)与 `FrameInfo prmFrame`(`shaders/shaderio.h:321-424`)中的通用字段。持久化于 `.vkgs` 的 `renderer` 段。

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmFrame.shDegree` | Maximum SH degree | `--maxShDegree` | 3 (`shaderio.h:345`) | 0-3 (`parameters.cpp:139-140`,`gaussian_splatting_ui.cpp:3511`) | 视相关效果的最高球谐阶数。降低可提速/省带宽,牺牲视角相关高光 | 全局 |
| `prmRender.showShOnly` | Show SH deg > 0 only | 无 | false (`parameters.h:191`) | bool | 去掉 SH deg0 基色,仅显示高阶 SH 贡献(叠加到中性灰)。用于分析视相关分量 (`gaussian_splatting_ui.cpp:3516`) | 全局 |
| `prmRender.opacityGaussianDisabled` | Disable opacity gaussian | 无 | false (`parameters.h:192`) | bool | 禁用高斯 alpha 分量,使完整范围可见。配合 Splat scale 分析分布 (`gaussian_splatting_ui.cpp:3521`) | 全局 |
| `prmFrame.splatScale` | Splat scale | 无 | 1.0 (`shaderio.h:344`) | 0.1–2.0(点云模式 0.1–10.0)(`gaussian_splatting_ui.cpp:3673-3674`) | 缩放 splat 尺寸(可视化用)。3DGUT 管线下置灰 | 光栅 |
| `prmFrame.alphaCullThreshold` | Alpha culling threshold | 无 | 1/255 ≈ 0.0039 (`shaderio.h:348`) | GUI 显示 0–255(内部 /255)(`gaussian_splatting_ui.cpp:3505-3508`) | 丢弃低透明度(低贡献)splat | 全局 |
| `prmFrame.alphaClamp` | Alpha clamp | 无 | 0.99 (`shaderio.h:354`) | GUI 0.0–3.0(`gaussian_splatting_ui.cpp:4173`) | 单次粒子命中的最大 alpha 上限,防单 splat 完全不透明。源自 3DGS 论文,防数值不稳定 | 光追 |
| `prmFrame.minTransmittance` | Minimum transmittance | 无 | 0.01 (`shaderio.h:355`) | GUI 0.0–1.0(`gaussian_splatting_ui.cpp:4068`) | 透射率降到此值以下时停止粒子行进 | 光追 |
| `prmFrame.maxPasses` | Maximum pass count | 无 | 200 (`shaderio.h:352`) | GUI clamp 1–1000(`gaussian_splatting_ui.cpp:4049-4054`) | 每像素最大 ray marching 遍数,每遍处理最多「particle samples per pass」个命中。越多可渲染越密场景,越慢 | 光追 |
| `prmRender.visualize` | Visualize Mode | 无 | `VISUALIZE_FINAL`=0 (`parameters.h:166`) | 见下表 | 可视化模式。**仅光追/hybrid 可用**(`gaussian_splatting_ui.cpp:3437,3458`) | 光追/hybrid |
| `prmRender.clayColor` | Clay Color | 无 | (0.423, 0.337, 0.251) (`parameters.h:186`) | RGB | `visualize==VISUALIZE_CLAY` 时的粘土色 (`gaussian_splatting_ui.cpp:3465`) | 光追 |
| `prmRender.wireframe` | Wireframe | 无 | false (`parameters.h:190`) | bool | 线框显示包围体。需 `VK_KHR_fragment_shader_barycentric` + `shaderFloat64`,否则置灰 (`gaussian_splatting_ui.cpp:3498`) | 全局 |
| `prmRender.colorFormat` | Color Format | `--colorBufferFormat`(仅 .cfg,见第 13 节) | `VK_FORMAT_R32G32B32A32_SFLOAT` (`parameters.h:183`) | R8G8B8A8_UNORM / R16G16B16A16_SFLOAT / R32G32B32A32_SFLOAT (`gaussian_splatting_ui.cpp:395-397`) | 颜色缓冲格式,精度越高时间累积质量越好但更耗显存 | 全局 |

> 注意默认值差异:C++ 结构体 `prmRender.colorFormat` 默认为 `R32G32B32A32_SFLOAT`(`parameters.h:183`),而 GUI tooltip 与 `.cfg` 的 `--colorBufferFormat` 帮助文本把 **R16G16B16A16(64-bit)称为「default」**(`gaussian_splatting_ui.cpp:3367,169`)。以源码结构体默认为准:实际初始默认是 128-bit 的 R32F。

### 4.1 Visualize 模式全枚举(`shaders/shaderio.h:111-127`)

| 值 | 宏 | GUI 标签 | 附加控件 |
|----|----|---------|---------|
| 0 | `VISUALIZE_FINAL` | Final render | — |
| 1 | `VISUALIZE_CLOCK` | Clock cycles | Min/max(默认 0.0–0.5)、Shift(`clockVisu*`,`parameters.h:169-170`,`gaussian_splatting_ui.cpp:3470-3471`) |
| 2 | `VISUALIZE_RAYHITS` | Ray Hit Count | Min/max(默认 0–100)、Shift(`hitsVisu*`,`parameters.h:168`,`gaussian_splatting_ui.cpp:3486-3487`) |
| 3 | `VISUALIZE_DEPTH` | Depth (Closest hit) | Min/max(默认 0.0–20.0)、Shift(`depthVisu*`,`parameters.h:172`) |
| 4 | `VISUALIZE_DEPTH_INTEGRATED` | Depth (iso thres) | 同上 depth 控件 |
| 5 | `VISUALIZE_DEPTH_FOR_DLSS` | Depth (for DLSS) | 同上 depth 控件 |
| 6 | `VISUALIZE_NORMAL` | Normal (closest hit) | — |
| 7 | `VISUALIZE_NORMAL_INTEGRATED` | Normal (Integrated) | — |
| 8 | `VISUALIZE_NORMAL_FOR_DLSS` | Normal (For DLSS) | — |
| 9 | `VISUALIZE_DLSS_INPUT` | DLSS Input | 仅 DLSS 开启时可选 |
| 10 | `VISUALIZE_DLSS_ALBEDO` | DLSS Guide: Albedo | 仅 DLSS 开启 |
| 11 | `VISUALIZE_DLSS_SPECULAR` | DLSS Guide: Specular | 仅 DLSS 开启 |
| 12 | `VISUALIZE_DLSS_NORMAL` | DLSS Guide: Normal | 仅 DLSS 开启 |
| 13 | `VISUALIZE_DLSS_MOTION` | DLSS Guide: Motion | 仅 DLSS 开启 |
| 14 | `VISUALIZE_DLSS_DEPTH` | DLSS Guide: Depth | 仅 DLSS 开启 |
| 15 | `VISUALIZE_SPLAT_ID` | Splat ID (Harlequin) | 每 splat 唯一伪彩色 |
| 16 | `VISUALIZE_CLAY` | Clay mode | 见 Clay Color |

GUI 用两套菜单:`GUI_VISUALIZE`(DLSS 关,DLSS 项置灰)与 `GUI_VISUALIZE_DLSS_ON`(DLSS 开,DLSS 项可选)(`gaussian_splatting_ui.cpp:274-308,3433-3435`)。

### 4.2 Visualize 微调子参数

| 参数 (C++ 变量) | GUI 标签 | 默认值 | 范围 | 适用模式 |
|------|---------|--------|------|---------|
| `prmRender.hitsVisuShift` / `hitsVisuMinMax` | Shift / Min/max | 0.0 / (0,100) | Shift -1..1 | Ray Hit Count |
| `prmRender.clockVisuShift` / `clockVisuMinMax` | Shift / Min/max | 0.0 / (0.0,0.5) | Shift -1..1 | Clock cycles |
| `prmRender.depthVisuShift` / `depthVisuMinMax` | Shift / Min/max | 0.0 / (0.0,20.0) | Shift -1..1 | 三种 Depth |

(`parameters.h:167-172`)

---

## 5. 光栅化专属(3DGS / 3DGUT primary)

`RasterParameters prmRaster`(`src/parameters.h:199-224`)。仅光栅管线(0/1/3/4/5)生效。面板 `guiDrawRasterizationProperties`(`gaussian_splatting_ui.cpp:3622`)。持久化于 `.vkgs` 的 `renderer` 段。

### 5.1 形状 / 投影

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRtx.kernelDegree` | Kernel degree | `--kernelDegree` | `KERNEL_DEGREE_QUADRATIC`=2 (`parameters.h:241`) | 0(Linear)/1(Laplacian)/2(Quadratic)/3(Cubic)/4(Tesseractic)/5(Quintic)(`shaderio.h:158-163`,`gaussian_splatting_ui.cpp:342-347`) | 高斯求值核阶数,须与训练一致;改动触发 BLAS 重建。**仅 3DGUT / Hybrid 3DGUT(管线 4/5)可用**(`gaussian_splatting_ui.cpp:3637-3650`) | 3DGUT & 光追共享 |
| `prmRaster.extentProjection` | Projection Method | `--extentProjection` | `EXTENT_CONIC`=1 (`parameters.h:209`) | `EXTENT_EIGEN`=0 / `EXTENT_CONIC`=1 (`shaderio.h:139-140`) | 3D 协方差 → 2D 范围投影法。Eigen=基对齐矩形(更快);Conic=轴对齐矩形(同 3DGS/3DGUT 论文)。**仅 3DGUT 可选;3DGS(0/1/3)恒用 Eigen**(`parameters.cpp:151-152`,`gaussian_splatting_ui.cpp:3653-3666`) | 3DGUT |
| `prmRaster.pointCloudModeEnabled` | Disable splatting | 无 | false (`parameters.h:208`) | bool | 点云模式,只显示 splat 中心。Splat scale 仍生效。3DGUT 下置灰 (`gaussian_splatting_ui.cpp:3678`) | 光栅 |

### 5.2 排序(sorting)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRaster.sortingMethod` | Sorting method(菜单栏也有) | `--sortStrategy` | `SORTING_GPU_SYNC_RADIX`=0 (`parameters.h:201`) | 0=GPU radix / 1=CPU async mono / 2=CPU async multi / 3=Stochastic splat (`shaderio.h:24-27`) | GPU radix 最快;CPU async 慢~超慢;Stochastic splat 无需排序、超快但有噪 (`parameters.cpp:149-150`,`gaussian_splatting_ui.cpp:1664-1716`) | 光栅 |
| `prmRaster.cpuLazySort` | Lazy CPU sorting | 无 | true (`parameters.h:202`) | bool | 仅视点变化时才排序。仅 CPU async 模式下可用(GPU radix / stochastic 时置灰)(`gaussian_splatting_ui.cpp:3698-3701`) | 光栅(CPU 排序) |

> GUI 排序下拉 `GUI_SORTING` 只列出 3 项:GPU radix(0)、CPU async std multi(2)、Stochastic splat(3)——**未暴露 CPU async mono(1)**;但 CLI `--sortStrategy` 接受 0-3 全部 4 个值(`gaussian_splatting_ui.cpp:310-312` vs `parameters.cpp:150`)。

### 5.3 剔除(culling)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRaster.frustumCulling` | Frustum culling | 无 | `FRUSTUM_CULLING_AT_DIST`=1 (`parameters.h:203`) | 0=Disabled / 1=At distance stage / 2=At raster stage (`shaderio.h:130-132`) | Disabled 用于性能对比;At distance 在距离计算 shader(仅 GPU radix / stochastic);At raster 在 vertex/mesh shader (`gaussian_splatting_ui.cpp:3720-3734`) | 光栅 |
| `prmFrame.frustumDilation` | Frustum dilation | 无 | 0.2 (`shaderio.h:347`) | GUI 0.0–1.0(`gaussian_splatting_ui.cpp:3736`) | 扩张视锥边界补偿「仅按中心测可见性」;正值按百分比外扩,减少边界处误剔 | 光栅 |
| `prmRaster.sizeCulling` | Screen size culling | 无 | `SIZE_CULLING_DISABLED`=0 (`parameters.h:204`) | 0=Disabled / 1=Enabled (`shaderio.h:135-136`) | 剔除投影包围球小于阈值像素的 splat。仅用距离计算 shader 时(GPU radix / stochastic)可用 (`gaussian_splatting_ui.cpp:3746`) | 光栅 |
| `prmFrame.sizeCullingMinPixels` | Min pixel coverage | 无 | 1.0 (`shaderio.h:349`) | GUI 0.1–20.0(`gaussian_splatting_ui.cpp:3759`) | splat 可见的最小投影像素覆盖(包围球直径),小于则剔 | 光栅 |

### 5.4 着色(Shading,需 lighting 开)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRaster.quantizeNormals` | Quantize Normals | 无 | true (`parameters.h:218`) | bool | 法线八面体编码(Meyer 2010),mesh→fragment 带宽从 96 bit 降到 32 bit。需 lighting 开 (`gaussian_splatting_ui.cpp:3780`) | 光栅 |
| `prmRaster.ftbSyncMode` | FTB Sync Mode | 无 | `FTB_SYNC_DISABLED`=0 (`parameters.h:221`) | `FTB_SYNC_DISABLED`=0 / `FTB_SYNC_INTERLOCK`=1 (`shaderio.h:147-148`) | 深度缓冲存储图访问的同步。Interlock 正确但慢;Disabled 快但偶有瑕疵。需 lighting 开且非 stochastic 排序 (`gaussian_splatting_ui.cpp:3787-3796`) | 光栅 |
| `prmRaster.depthIsoThreshold` | Depth Iso Threshold | 无 | 0.7 (`parameters.h:223`,`shaderio.h:401`) | GUI 0.0–1.0(`gaussian_splatting_ui.cpp:3798`) | 深度拾取的透射率阈值,透射率降到此值时记录深度。越低越晚拾取(积累更多不透明度)。需 lighting 开 | 光栅 |

### 5.5 高级(Advanced)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRaster.distShaderWorkgroupSize` | Dist WG size | 无 | 256(ADA6000 实验最优)(`parameters.h:205`) | 512/256/128/64/32/16(`gaussian_splatting_ui.cpp:358-363`) | 距离计算 compute shader 的 workgroup 大小,影响占用率/性能,依 GPU 架构 | 光栅 |
| `prmRaster.meshShaderWorkgroupSize` | Mesh WG size | 无 | 32(ADA6000 实验最优)(`parameters.h:206`) | 128/64/32/16/8(`gaussian_splatting_ui.cpp:365-369`) | mesh shader 的 workgroup 大小,影响占用率/性能 | 光栅(mesh) |
| `prmRaster.fragmentBarycentric` | Fragment shader barycentric | 无 | false (`parameters.h:207`) | bool | 用重心坐标减少 vertex/mesh shader 输出。需 `VK_KHR_fragment_shader_barycentric`,3DGUT 下置灰 (`gaussian_splatting_ui.cpp:3840`) | 光栅(非 3DGUT) |

### 5.6 低通滤波 / mip-splatting(位于「Particle Filtering」面板,通用渲染面板内)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRaster.covarianceDilation` | Low pass Kernel Size | 无 | 0.3(3DGS/3DGUT/Stochastic 默认;MipSplatting 用 0.1)(`parameters.h:216`) | 下拉仅四档:0.0 / 0.1 / 0.2 / 0.3 (`gaussian_splatting_ui.cpp:435-438,3539`) | 2D 协方差低通核尺寸(0=不滤波)。越大越平滑但可能损失锐度 | 光栅 |
| `prmRaster.msAntialiasing` | Mip splatting antialiasing | 无 | false (`parameters.h:213`) | bool | 指示模型是否用 mip-splatting 抗锯齿训练/渲染;按低通核尺寸补偿粒子不透明度。**仅当 `covarianceDilation > 0.0` 时可用**(否则置灰)(`gaussian_splatting_ui.cpp:3562-3566`) | 光栅 |

---

## 6. 光线追踪专属(3DGRT secondary)

`RtxParameters prmRtx`(`src/parameters.h:236-272`)与 `FrameInfo` 的 RT 字段。仅光追管线(2/3/5)生效。面板 `guiDrawRaytracingProperties`(`gaussian_splatting_ui.cpp:3856`)。持久化于 `renderer` 段。

### 6.1 路径追踪(Path Tracing)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmFrame.rtxMaxBounces` | Max bounces | `--rtxMaxBounces` | 3 (`shaderio.h:356`) | 0–16(`parameters.cpp:157-158`,`gaussian_splatting_ui.cpp:3873`) | 路径追踪最大光线反弹数,越高越真实(全局光照)越慢;0=仅直接光。`PIPELINE_MESH_3DGUT`(4)下置灰 | 光追 |
| `prmFrame.rtxSecondaryRayOffset` | Secondary ray offset | 无 | 0.001 (`shaderio.h:357`) | GUI 0.0–1.0(`gaussian_splatting_ui.cpp:3877`) | 所有次级光线(反弹 + mesh 阴影)的 TMin 偏移,防自交。主光线用相机近裁面为 TMin;粒子阴影光线另用「Particle shadow offset」 | 光追 |
| `prmRtx.fireflyClampThreshold` | Firefly clamp | 无 | 13.0(结构体默认)(`parameters.h:256`);但 `FrameInfo.rtxFireflyClampThreshold` 着色默认 0.0(`shaderio.h:409`) | GUI 0.0–100.0(`gaussian_splatting_ui.cpp:3884`) | 抑制随机亮点(firefly)的亮度阈值,超过则按比例缩放。减少快速移动时的时间条纹(尤其 DLSS)。0=禁用 | 光追 |

> `fireflyClampThreshold` 默认值细节:主机端结构体默认 13.0(`parameters.h:256`),而 shader UBO 字段 `rtxFireflyClampThreshold` 初值 0.0(`shaderio.h:409`);运行时由主机端 13.0 覆盖写入 UBO。持久化字段名 `fireflyClampThreshold`(`vkgs_project_writer.cpp:148`)。

### 6.2 粒子形状(Particle Shape)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRtx.kernelDegree` | Kernel degree | `--kernelDegree` | 2 (`parameters.h:241`) | 0-5(见 5.1) | 同 5.1,须匹配训练;改动触发 BLAS 重建 (`gaussian_splatting_ui.cpp:3902-3912`) | 光追 |
| `prmRtx.kernelMinResponse` | 无 GUI | 无 | 0.0113(论文常量)(`parameters.h:242`) | float | 核最小响应阈值(用于 BLAS 构建的包围;论文常量)。持久化 `kernelMinResponse`(`vkgs_project_writer.cpp:143`) | 光追 |
| `prmRtx.kernelAdaptiveClamping` | Adaptive clamp | 无 | true (`parameters.h:243`) | bool | 按粒子不透明度自适应包围体:低不透明度粒子更紧,提速。关闭则统一缩放。触发 BLAS 重建 (`gaussian_splatting_ui.cpp:3945`) | 光追 |
| `prmRtx.particleDepth` | Particle depth | `--rtxParticleDepth` | `PARTICLE_DEPTH_ELLIPSOID`=1 (`parameters.h:262`) | `BILLBOARD`=0 / `ELLIPSOID`=1 / `MAX_DENSITY_PLANE`=2(GUI 仅暴露 0/1)(`shaderio.h:184-186`,`gaussian_splatting_ui.cpp:422-424`) | 粒子命中深度计算:Billboard=光线到 billboard 平面交点(需 AABB 或 stochastic any-hit);Ellipsoid=依几何(AABB→最大密度点,icosa/sphere→椭球面命中)(`parameters.cpp:161-162`,`gaussian_splatting_ui.cpp:3959-3968`) | 光追 |
| `prmRtxData.billboardBoundingMode` | Billboard bounding mode | `--rtxBillboardBounding` | Fitted=0 | 0-7(见 3.2) | billboard 模式下的 TLAS 包围盒缩放;仅 billboard depth + (AABB 或 stochastic any-hit) 时可用 (`gaussian_splatting_ui.cpp:3977-3993`) | 光追 billboard |
| `prmRtx.billboardFrustumCulling` | Billboard frustum culling | 无 | true (`parameters.h:265`) | bool | any-hit 中剔除中心在视锥外的粒子,减少「光追可见但光栅被剔」的边界瑕疵。仅 billboard 模式相关 (`gaussian_splatting_ui.cpp:3994`) | 光追 billboard |
| `prmRtx.shortenRay` | Shorten ray | `--rtxShortenRay` | true (`parameters.h:268`) | bool | 用 payload 最远命中距离在 stochastic any-hit 中提前终止遍历。仅 billboard depth + stochastic any-hit 时相关 (`parameters.cpp:163-164`,`gaussian_splatting_ui.cpp:4006`) | 光追 billboard+stochastic |

### 6.3 粒子追踪(Particle Tracing)

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRtx.rtxTraceStrategy` | Trace strategy(菜单栏也有) | `--rtxSortStrategy` | `RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT`=2(结构体默认)(`parameters.h:245`) | 0=All pass(FULL_ANYHIT)/ 1=Stochastic pass(PASS_STOCHASTIC)/ 2=Stochastic any-hit(STOCHASTIC_ANYHIT)(`shaderio.h:166-168`,`gaussian_splatting_ui.cpp:383-385`) | All pass=沿每条光线处理全部高斯;Stochastic pass=逐遍随机透明;Stochastic any-hit=逐命中随机透明 (`parameters.cpp:155-156`,`gaussian_splatting_ui.cpp:1784-1817`) | 光追 |
| `prmRtx.particleSamplesPerPass` | Particle samples per pass | 无 | 18(ADA6000 实验最优)(`parameters.h:244`) | 下拉:1/2/4/8/12/16/18/20/32/64/128(`gaussian_splatting_ui.cpp:371-381`) | 每遍存储的粒子命中数(PARTICLES_SPP)。stochastic any-hit 时强制显示为 1 并置灰 (`gaussian_splatting_ui.cpp:4034-4047`) | 光追 |
| `prmFrame.maxPasses` | Maximum pass count | 无 | 200 | 1–1000 | 见第 4 节。stochastic any-hit 下总 any-hit/像素 = 1 × maxPasses | 光追 |
| `prmFrame.minTransmittance` | Minimum transmittance | 无 | 0.01 | 0.0–1.0 | 见第 4 节 | 光追 |

「Maximum anyhit/pixel」为只读派生值 = effectiveSpp × maxPasses(`gaussian_splatting_ui.cpp:4058-4064`)。

### 6.4 RT 着色 / 阴影 / AO / 合成 / 高级

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRtx.depthIsoThresholdRTX` | Depth Iso Threshold(RT Shading) | 无 | 0.7 (`parameters.h:260`,`shaderio.h:402`) | GUI 0.0–1.0(`gaussian_splatting_ui.cpp:4083`) | 光追深度拾取的透射率阈值,越低越晚拾取 | 光追 |
| `prmRtx.particleShadowOffset` | Particle shadow offset | 无 | 0.2 (`parameters.h:248`,`shaderio.h:391`) | GUI 0.0–1.0(`gaussian_splatting_ui.cpp:4100`) | 粒子阴影光线原点偏移(体积特性),越大越防自阴影瑕疵 | 光追阴影 |
| `prmRtx.particleShadowTransmittanceThreshold` | Particle shadow threshold | 无 | 0.8 (`parameters.h:249`,`shaderio.h:392`) | GUI 0.0–0.99(`gaussian_splatting_ui.cpp:4104`) | 粒子阴影终止的透射率阈值,越高越早终止=越硬阴影 | 光追阴影 |
| `prmRtx.particleShadowColorStrength` | Colored shadow strength | 无 | 0.0 (`parameters.h:250`,`shaderio.h:393`) | GUI 0.0–5.0(`gaussian_splatting_ui.cpp:4109`) | 阴影逐通道染色(彩色玻璃效果),0=单色,越大颜色渗透越强 | 光追阴影 |
| `prmRtx.particleEmissiveAoEnabled` | Particle emissive AO | 无 | false (`parameters.h:252`) | bool | 对发光 splat set 启用环境光遮蔽,仅对 mesh 追踪(不对其他 splat set)。编译期宏门控 (`gaussian_splatting_ui.cpp:4127`) | 光追 |
| `prmRtx.particleEmissiveAoRadius` | Particle emissive AO radius | 无 | 0.05 (`parameters.h:253`,`shaderio.h:396`) | GUI 0.001–FLT_MAX(`gaussian_splatting_ui.cpp:4136`) | AO 半球采样半径,控制 AO 光线追踪距离 | 光追 |
| `prmRtx.particleEmissiveAoStrength` | Particle emissive AO strength | 无 | 1.0 (`parameters.h:254`,`shaderio.h:397`) | GUI 0.0–5.0(`gaussian_splatting_ui.cpp:4139`) | AO 变暗强度:0=无,1=完全,>1=夸张 | 光追 |
| `prmFrame.minSplatSetCompositeTransmittance` | Splat set composite threshold | 无 | 0.1 (`shaderio.h:371`) | GUI 0.0–1.0(`gaussian_splatting_ui.cpp:4156`) | mesh/环境在 splat 后合成所需的最小透射率;低于则 splat 完全遮挡。用 stochastic pass 时通常需调高 | 光追 |
| `prmFrame.alphaClamp` | Alpha clamp | 无 | 0.99 | GUI 0.0–3.0 | 见第 4 节 | 光追 |
| `prmRtx.quantizeMeshPayload` | Quantize mesh payload | 无 | true (`parameters.h:271`) | bool | mesh 命中数据用八面体法线 + fp16 UV/切线打包,payload 15→9 float 槽,降低寄存器压力/local memory 溢出 (`gaussian_splatting_ui.cpp:4181`) | 光追 |
| `prmRtx.traceProfile` | 无 GUI(shader feedback) | 无 | false (`parameters.h:246`) | bool | 收集逐命中 trace profile 供 shader feedback,由编译宏 `TRACE_PROFILE` 门控,关闭时零开销 | 光追 |
| `prmRtx.temporalSampling` | (由 temporalSamplingMode 间接控制) | 无 | false (`parameters.h:239`) | bool | 是否累积多帧(DoF 等),不直接暴露,见第 8 节 | 光追 |

---

## 7. 光照与阴影

`prmRender.lightingEnabled`、`prmRender.shadowsMode`、`prmRender.normalMethod`、`prmRender.thinParticleThreshold`(`src/parameters.h:175-180`)。持久化于 `renderer` 段。逐 splat-set / mesh 材质见第 9 节末的材质表。

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRender.lightingEnabled` | Lighting(菜单栏也有) | `--lightingEnabled` | `LIGHTING_DISABLED`=0 (`parameters.h:179`) | `LIGHTING_DISABLED`=0 / `LIGHTING_ENABLED`=1 (`shaderio.h:175-176`,`gaussian_splatting_ui.cpp:408-409`) | 全模型光照:光栅=直接光,光追=完整路径追踪(NEE + BSDF,逐材质 maxBounces)。开启会自动激活 tonemapping(`gaussian_splatting_ui.cpp:1744`) (`parameters.cpp:143-144`) | 全部 |
| `prmRender.shadowsMode` | Shadows mode(菜单栏也有) | `--shadowMode` | `eShadowsDisabled`=0 (`parameters.h:180`) | `SHADOWS_DISABLED`=0 / `HARD`=1 / `SOFT`=2 (`shaderio.h:189-191`,`gaussian_splatting_ui.cpp:411-413`) | Off=不追阴影光线;Hard=点采样硬阴影;Soft=光源盘采样随机软阴影。**需 lighting 开且为光追管线**,否则置灰 (`parameters.cpp:145-146`,`gaussian_splatting_ui.cpp:1748-1781`) | 光追 |
| `prmRender.normalMethod` | Normal vectors | 无 | `eMaxDensityPlane`=0 (`parameters.h:175`) | `NORMAL_METHOD_MAX_DENSITY_PLANE`=0(GUI「Max density plane」)/ `ISO_SURFACE`=1(GUI「Kernel ellipsoid」)(`shaderio.h:171-172`,`gaussian_splatting_ui.cpp:405-406`) | 法线计算法:Max density plane=中心切平面近似(StochasticSplats,快且好);Iso-surface ellipsoid=正则空间椭球面求交(更几何精确)(`gaussian_splatting_ui.cpp:3584-3591`) | 全部(光照时) |
| `prmRender.thinParticleThreshold` | Thin particle threshold | 无 | 1e-6 (`parameters.h:176`,`shaderio.h:406`) | GUI 0.0–1.0(`gaussian_splatting_ui.cpp:3594`) | 粒子某轴退化的尺度阈值,细于此则当作平盘(法线沿细轴)而非完整椭球计算 | 全部 |

### 光源(LightSource,`shaders/shading.h:116-129`)

光源是场景资产,通过 GUI「Light」属性面板编辑(`guiDrawLightProperties`,`gaussian_splatting_ui.cpp:4682`),持久化于 `.vkgs` 的 `lights` 段(`vkgs_project_writer.cpp:234-283`)。无 CLI flag。

| 参数 (C++ 变量) | GUI 标签 | 默认值 | 取值范围/枚举 | 说明 |
|------|---------|--------|--------------|------|
| `type` | Type | `ePointLight`=1 (`shading.h:118`) | `eDirectionalLight`=0 / `ePointLight`=1 / `eSpotLight`=2 (`shading.h:108-113`,`gaussian_splatting_ui.cpp:349-351`) | 光源类型 |
| `enabled` | Enabled | 1 (`shading.h:128`) | 0/1 | 启用 |
| `color` | Color | (1,1,1) (`shading.h:119`) | RGB | 颜色 |
| `intensity` | Intensity | 1.0 (`shading.h:120`) | 0–1e7(`gaussian_splatting_ui.cpp:4747`) | 强度 |
| `range` | Range | 10.0 (`shading.h:122`) | 0.1–1e7,point/spot 有效(`gaussian_splatting_ui.cpp:4753`) | 有效范围,到此距离平滑衰减为 0 |
| `attenuationMode` | Attenuation | 2 (`shading.h:126`) | 0=None / 1=Linear / 2=Quadratic / 3=Physical (`gaussian_splatting_ui.cpp:353-356`) | 衰减方式;directional 强制 None |
| `innerConeAngle` | Inner Cone Angle | 30.0 (`shading.h:124`) | 0–90°,仅 spot(`gaussian_splatting_ui.cpp:4793`) | spot 内锥角(全强度) |
| `outerConeAngle` | Outer Cone Angle | 45.0 (`shading.h:125`) | inner–90°,仅 spot(`gaussian_splatting_ui.cpp:4797`) | spot 外锥角(渐隐到 0) |
| `radius` | Radius | 1.0 (`shading.h:127`) | 0.01–100(`gaussian_splatting_ui.cpp:4802`) | 光源半径(软阴影 + proxy 可视化) |
| `position` / `direction` | Translation / Rotation(实例) | — | — | 由实例 translation/rotation 计算 |

---

## 8. 时间累积与抗锯齿

`prmRtx.temporalSamplingMode`、`prmRtx.temporalSampling`、`prmFrame.frameSampleMax`。持久化于 `renderer` 段。

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `prmRtx.temporalSamplingMode` | Temporal sampling | 无 | `TEMPORAL_SAMPLING_AUTO`=0 (`parameters.h:240`) | `AUTO`=0(Automatic)/ `ENABLED`=1(Force enabled)/ `DISABLED`=2(Force disabled)(`shaderio.h:263-265`,`gaussian_splatting_ui.cpp:338-340`) | 多帧累积控制。Auto 会在有 DoF 或 Monte-Carlo(pass)trace 策略时自动启用。vsync 关闭时收敛更快 (`gaussian_splatting_ui.cpp:3389-3397`) | 光追 |
| `prmFrame.frameSampleMax` | Temporal samples count | 无 | 1000(结构体默认)(`shaderio.h:360`) | GUI InputInt,clamp 1–100000(`gaussian_splatting_ui.cpp:3409-3413`) | 累积多少帧后停止;持久化名 `temporalSamplesCount`(`vkgs_project_writer.cpp:140`) | 光追 |
| `prmRtx.temporalSampling` | (间接) | 无 | false (`parameters.h:239`) | bool | 由 mode 派生的实际开关,不直接暴露 UI;持久化名 `temporalSampling` | 光追 |
| `prmFrame.frameSampleId` | (运行时) | 无 | 0 (`shaderio.h:359`) | int | 当前累积帧索引(内部状态,非用户设置) | — |

抗锯齿相关的 `msAntialiasing`(mip-splatting)见 5.6;DLSS-RR 见第 12 节。

---

## 9. 相机与景深(DoF)+ Splat Set 材质

相机是场景资产(非全局参数),GUI「Camera」属性面板(`guiDrawCameraProperties`,`gaussian_splatting_ui.cpp:4476`),持久化于 `.vkgs` 的 `camera` 与 `cameras`(预设)段(`vkgs_project_writer.cpp:190-229`)。无直接 CLI flag(但可用 `--loadCameraPresets`/`--activateCameraPreset`,见第 13 节)。

| 参数 (C++ 变量) | GUI 标签 | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|--------|--------------|---------------|------|
| `camera.model` | Camera type | `CAMERA_PINHOLE`=0 | `CAMERA_PINHOLE`=0 / `CAMERA_FISHEYE`=1 (`shaderio.h:143-144`,`gaussian_splatting_ui.cpp:335-336`) | 相机模型;Fisheye 未必所有管线支持;当前不逐相机存储 (`gaussian_splatting_ui.cpp:4499-4502`) | 全部 |
| `camera.fov` | FOV | — | 1–179°,对数滑块(`gaussian_splatting_ui.cpp:4511`) | 视场角(度) | 全部 |
| `camera.clip` | Clip planes | — | 两个 float(`gaussian_splatting_ui.cpp:4508`) | 近/远裁面 | 全部 |
| `camera.dofMode` | Depth of Field | `DOF_DISABLED`=0 | `DOF_DISABLED`=0 / `FIXED_FOCUS`=1 / `AUTO_FOCUS`=2 (`shaderio.h:179-181`,`gaussian_splatting_ui.cpp:415-420`) | 景深模式。**仅 3DGRT(2)/3DGUT(4)/hybrid 3DGUT(5)** 生效;Auto focus 用光标处表面距离,**需光追(仅 2 或 5,`supportsAutoFocus()`)**;设为 Auto 会触发 temporal sampling (`gaussian_splatting_ui.cpp:4516-4534`,`gaussian_splatting.h:212-215`) | 2/4/5 |
| `camera.focusDist` | Focus distance | 1.3(`shaderio.h:362`) | GUI 0.1–15.0(`gaussian_splatting_ui.cpp:4548`) | 对焦距离;Auto focus 下只读 | 2/4/5 DoF |
| `camera.aperture` | Aperture | 0.001(`shaderio.h:363`) | GUI 0.0–0.01(`gaussian_splatting_ui.cpp:4554`) | 光圈;0 无 DoF 效果 | 2/4/5 DoF |
| `camera.eye`/`ctr`/`up` | Eye / Center / Up | — | vec3 | 外参(位置/目标/上向量)(`gaussian_splatting_ui.cpp:4578-4582`) | 全部 |

> DoF 管线限制的源码依据:`gaussian_splatting_ui.cpp:4516-4517` 仅当 `prmSelectedPipeline` ∈ {`PIPELINE_RTX`, `PIPELINE_HYBRID_3DGUT`, `PIPELINE_MESH_3DGUT`} 才启用 DoF 控件。注意 `PIPELINE_HYBRID`(3)**不在**其列。

### Splat Set 材质(PBR metallic-roughness,`shaders/shading.h:29-88`)

逐 splat-set 实例材质,GUI「Material」面板(`guiDrawSplatSetProperties`,`gaussian_splatting_ui.cpp:4365-4392`),持久化于 `.vkgs` 的 `splats[].material`(`vkgs_project_writer.cpp:348-364`)。仅在 lighting 开时影响着色。无 CLI flag。

| 参数 (C++ 变量) | GUI 标签 | 默认值 | 范围 | 说明 |
|------|---------|--------|------|------|
| `baseColor` | Base Color | (0.7,0.7,0.7) (`shading.h:31`) | RGB | 基色 |
| `metallic` | Metallic | 0.0 (`shading.h:32`) | 0–1 | 金属度 |
| `roughness` | Roughness | 0.5 (`shading.h:33`) | 0–1 | 粗糙度 |
| `ior` | IOR | 1.5 (`shading.h:34`) | 1–3(GUI)| 折射率 |
| `transmission` | Transmission | 0.0 (`shading.h:35`) | 0–1 | 透射(0=不透明,1=全透射) |
| `opacity` | Opacity | 1.0 (`shading.h:36`) | 0–1 | 不透明度 |
| `emissive` | Emissive | (0,0,0) (`shading.h:41`) | RGB | 自发光 |
| `emissiveStrength` | Emissive Strength | 1.0 (`shading.h:42`) | 0–FLT_MAX(GUI)| 自发光倍率(KHR_materials_emissive_strength) |
| `maxBounces` | Max Bounces | 3 (`shading.h:43`) | 0–16(GUI)| 逐材质路径追踪最大反弹(0=仅直接光) |
| `specularFactor` | (仅工程持久化) | 1.0 (`shading.h:37`) | — | KHR_materials_specular 电介质高光权重 |
| `specularColorFactor` | (仅工程持久化) | (1,1,1) (`shading.h:38`) | RGB | KHR_materials_specular 高光颜色 |
| `clearcoatFactor` | (仅工程持久化) | 0.0 (`shading.h:39`) | — | KHR_materials_clearcoat 清漆强度 |
| `clearcoatRoughness` | (仅工程持久化) | 0.0 (`shading.h:40`) | — | 清漆粗糙度 |

> 上表中 `specularFactor`/`specularColorFactor`/`clearcoat*` 会写入 `.vkgs`(`vkgs_project_writer.cpp:359-364`)但**未在 splat set 的 GUI Material 面板暴露滑块**(GUI 仅暴露 baseColor/metallic/roughness/emissive/emissiveStrength/ior/transmission/opacity/maxBounces,见 `gaussian_splatting_ui.cpp:4373-4382`)。此外 `Material` 还含大量贴图槽(`baseColorTexture` 等,-1=无)、KHR_texture_transform UV 变换、以及 legacy `pbrSpecularGlossiness` 字段(`shading.h:45-87`),这些主要由 glTF mesh 导入填充,不作为可调渲染参数。Mesh 材质字段类似,持久化于 `meshInstances`(`vkgs_project_writer.cpp:437-449`)。

---

## 10. 色调映射(Tonemapping)

`m_tonemapperData`(`shaderio::TonemapperData`),GUI「Tonemapping」面板(`guiDrawTonemappingProperties`,`gaussian_splatting_ui.cpp:2367`),持久化于 `.vkgs` 的 `tonemapping` 段(`vkgs_project_writer.cpp:560-595`)。全局后处理,无 CLI flag。默认值取自 `shaderio::TonemapperData()`。

| 参数 (C++ 变量) | GUI 标签 | 默认 | 范围 | 说明 |
|------|---------|------|------|------|
| `isActive` | Enable | 见注 | bool | 启用色调映射;lighting 开时自动置 1(`gaussian_splatting_ui.cpp:1744`) |
| `method` | Method | Filmic(0) | Filmic / Uncharted 2 / Clip / ACES / AgX / Khronos PBR(`gaussian_splatting_ui.cpp:2373`) | 算法(HDR→SDR) |
| `exposure` | Exposure | — | 0.1–200,对数(`gaussian_splatting_ui.cpp:2385`) | 曝光倍率(0.1 很暗,1 中性,200 很亮) |
| `contrast` | Contrast | — | 0–2(`:2387`) | 对比度(1 中性) |
| `brightness` | Brightness | — | 0–2(`:2389`) | 亮度 gamma 曲线(1 中性) |
| `saturation` | Saturation | — | 0–2(`:2391`) | 饱和度(0 灰度,1 中性) |
| `vignette` | Vignette | — | -1–1(`:2393`) | 暗角(-1 很亮,0 无,1 很暗) |
| `autoExposure` | Auto Exposure/Enable | — | bool | 自动曝光(`:2407`) |
| `averageMode` | Average Mode | — | Mean / Median(`:2410`) | 场景亮度计算法 |
| `autoExposureSpeed` | Adaptation Speed | — | 0–100(`:2412`) | 自动曝光适应速度 |
| `evMinValue` | Min (EV100) | — | -24–24(`:2415`) | 直方图最小亮度(log stops) |
| `evMaxValue` | Max (EV100) | — | -24–24(`:2417`) | 直方图最大亮度 |
| `enableCenterMetering` | Center Weighted Metering | — | bool | 中央测光(`:2419`) |
| `centerMeteringSize` | Center Metering Size | — | 0.01–1.0(`:2422`) | 中央测光区域大小 |
| `vibrance` | Vibrance | — | -1–1(`:2439`) | 自然饱和度(仅提升低饱和色) |
| `shadowBias` | Shadow Bias | — | -1–1(`:2441`) | 阴影色调偏移 |
| `midtoneBias` | Midtone Bias | — | -1–1(`:2443`) | 中间调偏移 |
| `highlightBias` | Highlight Bias | — | -1–1(`:2445`) | 高光偏移 |
| `coolColor` | Cool Shadows | 白(无染) | RGB(`:2450`) | 阴影染色(分离色调) |
| `warmColor` | Warm Highlights | 白(无染) | RGB(`:2452`) | 高光染色 |
| `splitBalance` | Split Balance | — | -0.5–0.5(`:2454`) | 冷暖平衡 |
| `temperature` | Temperature | 6506K(D65) | 2000–15000 K(`:2478`) | 白平衡色温 |
| `tint` | Tint | 见注 | -0.03–0.03(`:2495`) | 绿/品红色调(Duv 单位) |
| `dither` | Enable(Dithering) | — | bool | 抖动减少色带(`:2520`) |

> `isActive`/`temperature`/`tint` 等默认值来自 `shaderio::TonemapperData` 结构体(定义在 nvpro_core2 依赖中,repo 内未含),此处以 GUI 显示语义为准。「reset」按钮把整个 `m_tonemapperData = {}` 还原(`:2536`)。

---

## 11. 环境 / 背景(Environment)

`m_sky`(SkyPhysicalParameters + IBL),GUI「Sky」属性面板(`guiDrawSkyProperties`,`gaussian_splatting_ui.cpp:2730`),持久化于 `.vkgs` 的 `environment` 段(`vkgs_project_writer.cpp:460-501`)。无 CLI flag。

### 11.1 通用

| 参数 (C++) | GUI 标签 | 默认 | 取值范围/枚举 | 说明 |
|------|---------|------|--------------|------|
| `m_sky.mode()` | Mode | 见注 | `ENV_MODE_NONE`=0(None)/ `SKY`=1(Sky)/ `HDR`=2(HDR)(`shaderio.h:258-260`,`gaussian_splatting_ui.cpp:2740`) | 环境模式;`prmFrame.envMode` 镜像(`shaderio.h:414`) |
| `m_sky.isEnabled()` | Lighting | — | bool | 环境是否贡献光照(`prmFrame.envEnabled`,`shaderio.h:415`);仅 mode≠None 时显示(`gaussian_splatting_ui.cpp:2749`) |
| `m_sky.resolution()` | Resolution | — | ivec2;HDR 模式只读(`gaussian_splatting_ui.cpp:2759`) | 环境贴图分辨率 |

### 11.2 Sky & Sun(仅 `eSky` 模式;`skyParams`,`prmFrame.skyParams`)

字段来自 `SkyPhysicalParameters`(nvpro_core2 依赖);默认值由 `shaderio::SkyPhysicalParameters()` 构造(repo 内未含头文件,「reset」按钮还原,`gaussian_splatting_ui.cpp:2778-2781`)。持久化于 `environment.skyAndSun`(`vkgs_project_writer.cpp:472-482`)。

| 参数 (C++) | GUI 标签 | 范围 | 说明 |
|------|---------|------|------|
| `sunDirection` | (方位/仰角滑块) | azimuth/elevation(`gaussian_splatting_ui.cpp:2783`) | 太阳方向 |
| `sunDiskScale` | Sun Disk Scale | 0–10(`:2784`) | 日盘大小 |
| `sunDiskIntensity` | Sun Disk Intensity | 0–5(`:2785`) | 日盘强度 |
| `sunGlowIntensity` | Sun Glow Intensity | 0–5(`:2786`) | 日晕强度 |
| `haze` | Haze | 0–15(`:2792`) | 雾霾 |
| `redblueshift` | Red Blue Shift | -1–1(`:2793`) | 红蓝偏移 |
| `saturation` | Saturation | 0–1(`:2794`) | 饱和度 |
| `horizonHeight` | Horizon Height | -1–1(`:2795`) | 地平线高度 |
| `groundColor` | Ground Color | RGB(`:2796`) | 地面色 |
| `horizonBlur` | Horizon Blur | 0–5(`:2797`) | 地平线模糊 |
| `nightColor` | Night Color | RGB(`:2798`) | 夜空色 |

### 11.3 IBL / HDR(仅 `eHDR` 模式)

持久化于 `environment.ibl`(`vkgs_project_writer.cpp:486-498`)。

| 参数 (C++) | GUI 标签 | 默认 | 范围 | 说明 |
|------|---------|------|------|------|
| `m_sky.iblFilePath()` | File | 空 | `.hdr` | HDR 环境贴图路径(GUI「Load...」加载)(`gaussian_splatting_ui.cpp:2808`) |
| `m_sky.iblIntensity()` | Intensity | 见注 | 0–10(`:2827`) | HDR 强度倍率(`prmFrame.envIntensity`,`shaderio.h:420`) |
| `m_sky.iblRotation()` | Rotation | (0,0,0) | vec3(度)(`:2834`) | 环境旋转(欧拉度;`prmFrame.envRotation`,`shaderio.h:419`) |

---

## 12. DLSS(仅 `USE_DLSS` 构建)

`DlssDenoiser`(`src/dlss_denoiser.hpp`),GUI「DLSS-RR」面板(`guiDrawDenoisingProperties`,`gaussian_splatting_ui.cpp:2272`)。需 `USE_DLSS` 编译、NVIDIA 硬件/驱动、且为光追管线(2/3/5)。持久化于 `renderer` 段(`vkgs_project_writer.cpp:149-153`,`#if USE_DLSS`)。

| 参数 (C++ 变量) | GUI 标签 | CLI flag | 默认值 | 取值范围/枚举 | 作用与调优建议 | 适用 |
|------|---------|----------|--------|--------------|---------------|------|
| `m_dlss` `m_settings.enable` | Enable | `--dlssEnable` | false (`dlss_denoiser.hpp:56`) | bool | 启用 DLSS Ray Reconstruction 去噪(`dlss_denoiser.cpp:38`,`gaussian_splatting_ui.cpp:2295`);持久化名 `dlssEnabled` | 光追 + NVIDIA |
| `m_dlss` `m_settings.sizeMode` | Size Mode | 无 CLI | `SizeMode::eOptimal`=1 (`dlss_denoiser.hpp:57`) | `eMin`=0(Min)/ `eOptimal`=1(Optimal)/ `eMax`=2(Max)(`dlss_denoiser.hpp:47-52`,`gaussian_splatting_ui.cpp:2307`) | 内部渲染分辨率:Min 最小最快最低质;Optimal 均衡上采样(推荐);Max 最大内部分辨率(仅去噪+抗锯齿)。持久化名 `dlssSizeMode` | 光追 |
| `prmRtx.dlssMinRadianceThreshold` | Minimum Radiance | 无 | 0.0 (`parameters.h:258`,`shaderio.h:412`) | GUI 0.0–0.1(`gaussian_splatting_ui.cpp:2348`) | DLSS 输入最小辐亮度地板值,防负值(如放大 AO)造成噪点;持久化名 `dlssMinRadianceThreshold` | 光追 |

> `--dlssEnable` 通过 `m_dlss.registerParameters` 注册进全局 registry(`gaussian_splatting.cpp:58`)。Size Mode 无 CLI flag,只能 GUI/工程文件设置。GUI 另有只读「Current Resolution」显示(DLSS 内部分辨率 + 原生分辨率)。

---

## 13. benchmark / `.cfg` 动作参数

这些参数主要供 benchmark 脚本(`.cfg`,`--benchmark`)使用,多为触发式回调,在 sequence 起始处被应用。注册于 `src/gaussian_splatting_ui.cpp`(`registerParameters`,`:73-190`)与 sequencer(nvutils)。

### 13.1 UI 级动作回调(`registerParameters`)

| 参数 (C++ 变量) | flag | 参数/范围 | 作用 | 备注 |
|------|------|----------|------|------|
| `m_updateDataTrigger` | `--updateData` | 1 触发 | 参数改动后触发数据 buffer/texture 重生成 + shader 更新 | 仅 benchmark 脚本用 (`:78-85`) |
| `m_screenshotFilename` | `--screenshot <path>` | `.png` | 从 swapchain 截屏 | **headless 模式不可用**,改用 `--saveImage` (`:87-96`) |
| `m_saveImageBufferIndex` | `--saveImageBuffer <n>` | -1..20,默认 -1 | 选择 `--saveImage` 导出的缓冲索引 | -1=全部,0=main,1=aux1,2=comparison,3=normal,4=depth,5=ldr,6-11=DLSS (`:98-102`) |
| `m_saveImageFilename` | `--saveImage <path>` | `.png`/`.jpg`/`.hdr` | 保存内部渲染缓冲到文件 | 配合 `--saveImageBuffer` 选缓冲;**headless 可用** (`:104-121`) |
| `m_saveProjectFilename` | `--saveProject <path>` | `.vkgs` | 保存当前场景为 vkgs 工程 | 在 sequence 起始应用,捕获上个 sequence 结束时状态 (`:123-134`) |
| `m_cameraPresetsFilename` | `--loadCameraPresets <path>` | `.json`(INRIA 格式)| 加载相机预设(追加到现有列表) | (`:136-145`) |
| `m_activateCameraPresetIndex` | `--activateCameraPreset <i>` | 0-based 索引 | 立即激活某相机预设 | 需先 `--loadCameraPresets`;必要时触发 shader 重建 (`:147-166`) |
| `m_colorBufferFormatIndex` | `--colorBufferFormat <0/1/2>` | 0-2 | 设置颜色缓冲格式 | 0=R8G8B8A8_UNORM(32-bit),1=R16G16B16A16_SFLOAT(64-bit,帮助文本称默认),2=R32G32B32A32_SFLOAT(128-bit)。仅 .cfg 用(需已初始化 app)(`:168-189`) |

### 13.2 Sequencer 序列参数(nvutils,`--benchmark`)

由 `sequencerInfo.registerScriptParameters`(`main.cpp:75`)注册,控制每个 benchmark 段的采样。用法见仓库根 `benchmark_*.cfg`。

| flag | 作用 | 典型值 |
|------|------|--------|
| `--sequenceframes <n>` | 本段渲染帧数 | 16 / 200 / 256 等(见 `benchmark_billboards.cfg`) |
| `--sequenceaverages <n>` | 计时平均样本数 | 16 / 50 / 128 |
| `--sequenceresetframes <0/1>` | 是否在段起始重置帧计数器 | 0 |

> 这三个 flag 定义在 nvutils 的 ParameterSequencer(repo 内 `benchmark_*.cfg` 广泛使用,如 `benchmark_billboards.cfg:2-4`;`benchmark_3dgrt.cfg` 演示 `--pipeline`/`--kernelDegree`/`--screenshot`/`--updateData` 的组合)。

---

## 14. 常见目标 → 该调哪些参数(速查)

**减少边缘/薄结构锯齿(mip-splatting)**
- 光栅:`Low pass Kernel Size`(`covarianceDilation`)设 0.1–0.3 + 勾选 `Mip splatting antialiasing`(需 dilation>0)。
- 光追/hybrid:开 DLSS-RR(第 12 节),或 `Firefly clamp` 减快速移动噪点;`Temporal samples count` 调高。

**提高画质(收敛/去噪)**
- `Color Format` 用 R16F 或 R32F;`Maximum SH degree`=3;`Temporal sampling`=Force enabled + 加大 `Temporal samples count`(vsync 关更快收敛)。
- 光追:`Trace strategy`=All pass(最准,最慢);`Max bounces` 调高;`Firefly clamp` 抑噪;`kernelDegree` 匹配训练。

**加速(降质换帧率)**
- 光栅:`Sorting`=GPU radix 或 Stochastic splat;开 `Frustum culling`(At distance)+ `Screen size culling`;降 `Maximum SH degree`;调 `Dist/Mesh WG size`。
- 光追:`Trace strategy`=Stochastic any-hit;`kernelAdaptiveClamping` 开;`useTlasInstances`=1 + `useAABBs`;降 `Max bounces`/`maxPasses`;`quantizeMeshPayload`/`quantizeNormals` 开。
- 全局:开 DLSS(Size Mode=Min/Optimal)。

**开启重光照 + 阴影(需光追管线 2/3/5)**
- `Pipeline`=3DGRT/Hybrid;`Lighting`=Lighting on;`Shadows mode`=Hard/Soft;加光源(第 7 节);设 splat set/mesh 材质(第 9 节);调 `Particle shadow offset/threshold/colored strength`;可选 `Particle emissive AO`。
- 环境光:`Environment` mode=Sky/HDR 并勾 `Lighting`。

**景深(DoF)**
- `Pipeline`=3DGRT(2)/3DGUT(4)/Hybrid 3DGUT(5);相机 `Depth of Field`=Fixed/Auto(Auto 需 3DGRT);调 `Focus distance`/`Aperture`;会自动触发 temporal sampling(收敛需时间)。

**导出无头(headless)图**
- `--headless --benchmark`(或 `--headlessFrameCount N`);用 `.cfg` 里 `--saveImage <path>` + `--saveImageBuffer <n>`(**不要用 `--screenshot`,headless 无 swapchain**);可 `--colorBufferFormat` 选精度;`--size W H` 定分辨率;`--inputFile`/`--inputProject` 载场景;`--saveProject` 存工程。

**分析 splat 分布 / 调试**
- `Disable opacity gaussian` + 调 `Splat scale`;`Disable splatting`(点云);`Visualize Mode`=Splat ID / Clock cycles / Ray Hit Count / Depth / Normal(仅光追/hybrid);`Show SH deg > 0 only`;`Wireframe`。

---

## 附:源文件核实清单

- `src/parameters.h` — RenderParameters / RasterParameters / RtxParameters / VramDataParameters / RtxVramDataParameters、DofMode / ShadowsMode / NormalMethod / BillboardBoundingMode 枚举、默认值。
- `src/parameters.cpp` — `registerCommandLineParameters`:全部 CLI flag 与绑定。
- `shaders/shaderio.h` — FrameInfo 默认值、全部 `#define` 枚举(PIPELINE/FORMAT/SORTING/EXTENT/CAMERA/DOF/SHADOWS/LIGHTING/KERNEL_DEGREE/RTX_TRACE_STRATEGY/PARTICLE_DEPTH/TEMPORAL_SAMPLING/NORMAL_METHOD/VISUALIZE/ENV_MODE/FTB_SYNC/PARTICLE_FORMAT)。
- `shaders/shading.h` — Material 与 LightSource 结构及默认值。
- `src/gaussian_splatting_ui.cpp` — `registerParameters`(benchmark 回调)、全部 GUI 面板的 label/tooltip/range、`m_ui.enumAdd` 全部枚举标签。
- `src/main.cpp` — 应用级 CLI。
- `src/vkgs_project_writer.cpp` — `.vkgs` 持久化字段全集。
- `src/dlss_denoiser.hpp` / `dlss_denoiser.cpp` — DLSS SizeMode 枚举、`--dlssEnable`。
