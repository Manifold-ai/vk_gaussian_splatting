# VKGS 的 surfel 表面重建与 shadow(阴影)工作机制

本文是 VKGS(Vulkan Gaussian Splatting 渲染器,仓库根 `/home/lxruan/dev/vk_gaussian_splatting`)中「surfel 表面重建 + 阴影」子系统的参考手册。所有关键论断都标注了 `file:line`,以便直接回源码核对。正文用中文,代码标识符 / 参数名 / 文件路径保持英文。

> 权威背景文档:`docs/deep-dives/lighting_and_shadows.md`(下文简称「lighting 深潜文档」)。本文在其基础上,聚焦「surfel 是什么、着色如何消费它、阴影如何计算」这三条主线,并补齐源码级细节。

---

## 1. 概述:表面式着色 + 体积式阴影,不是逆渲染

VKGS 对高斯泼溅(3D Gaussian Splatting)的光照/阴影处理,可以用两句话概括:

1. **不是 re-lighting / 逆渲染。** 系统不为每个粒子反解材质、剥离原始光照再重打光。它把训练好的模型连同**已烘焙(baked)的光照**一起「ingest」进来——烘焙光照存在于 base color、更高阶 SH,或两者(lighting 深潜文档开篇,`docs/deep-dives/lighting_and_shadows.md:7`)。默认每个高斯 splat set 被当作一个**自发光辐射场**(radiance field),原样显示训练辐射,不需要任何光源(`docs/deep-dives/lighting_and_shadows.md:9`)。
2. **合成光是叠加上去的。** 用户可以给某个 splat set 降低 emissive、抬高 diffuse/specular,让它开始接收场景中的合成光源(点光/聚光/方向光)贡献,这部分光照(含阴影)**叠加**在烘焙辐射之上(`docs/deep-dives/lighting_and_shadows.md:9-11`)。

在这套框架里,「surfel」是把体积化的多层高斯**沿单条光线塌缩成的一个表面接收点**,着色和阴影都在这个点上进行,而**不是**把粒子集当成全体积介质逐粒子求光照再积分(那样代价过高,见 `docs/deep-dives/lighting_and_shadows.md:25`)。阴影则相反,是**体积式**的:阴影光线穿过真实几何(mesh 三角形)和高斯代理体,对彩色透过率做积分。

---

## 2. surfel 重建:逐光线重建的一个点,不是逐高斯的包围面

### 2.1 两道门控:是否重建、是否着色

surfel 的存在有两层开关。

**门控 1 —— 全局是否重建表面信息(`NEED_SURFACE_INFO`)。**
只有当以下任一条件成立时,才会重建 surface 信息(深度/法线/splat-id):`lightingEnabled` 打开、DoF 模式 ≠ disabled、DLSS 启用、或者当前是 depth/normal/splat-id 类可视化。见 `needSurfaceInfo()`(`src/gaussian_splatting.h:219-229`):

```cpp
bool need = prmRender.lightingEnabled || (m_assets.cameras.getCamera().dofMode != DOF_DISABLED);
#if defined(USE_DLSS)
  need = need || m_dlss.isEnabled();
#endif
const int v = prmRender.visualize;
need = need || (v >= VISUALIZE_DEPTH && v <= VISUALIZE_DEPTH_FOR_DLSS)
       || (v >= VISUALIZE_NORMAL && v <= VISUALIZE_NORMAL_FOR_DLSS) || (v == VISUALIZE_SPLAT_ID);
```

该 CPU 判定必须与着色器宏 `NEED_SURFACE_INFO` 一致,后者在 `src/gaussian_splatting.cpp:2326` 由 `needSurfaceInfo()` 求值后写入 Slang 宏。**默认模型 lighting 关闭 → `NEED_SURFACE_INFO=0`,不重建任何 surfel,退化为纯体积 3DGS 显示。**

**门控 2 —— 逐 splat set 是否参与着色(`needShading`)。**
即便全局重建了 surfel,某个 splat set 是否真正走 PBR 着色,取决于它的材质是否与光照交互。这个标志由 CPU 端 `updateMaterialNeedsShading()` 推导(`shaders/shading.h:91-104`):

```cpp
bool interactsWithLight = glm::length(mat.baseColor) > 0.001f
                          || mat.metallic > 0.001f
                          || mat.specularFactor > 0.001f
                          || mat.clearcoatFactor > 0.001f
                          || mat.transmission > 0.001f;
mat.needShading = interactsWithLight ? 1 : 0;
```

只有 baseColor 模长 > 0.001,或 metallic / specularFactor / clearcoatFactor / transmission 任一 > 0.001,`needShading` 才为 1(另有一条捷径:`usePbrSpecularGlossiness != 0` 时直接置 1,`shaders/shading.h:93-97`)。

**默认 splat 材质是纯发射**,故 `needShading = 0`。默认值在 `SplatSetManagerVk::createSplatSet` 里设置(`src/splat_set_manager_vk.cpp:253-261`):

| 字段 | 默认值 | 含义 |
|---|---|---|
| `baseColor` | `(0,0,0)` | 无 albedo → `needShading` 判定不满足 |
| `emissive` | `(1,1,1)` | 完全自发光,原样显示训练辐射 |
| `metallic` | `0.0` | |
| `roughness` | `0.5` | |
| `specularFactor` | `0.0` | |
| `specularColorFactor`| `(0,0,0)` | |
| `transmission` | `0.0` | |
| `maxBounces` | `0` | 无二次弹射 |

注意 `Material` 结构体本身的默认(`shaders/shading.h:31-43`)是 `baseColor=(0.7,0.7,0.7)`、`emissive=(0)`、`maxBounces=3`,那是给 mesh 材质用的;splat set 在创建时被显式改写成上面这张「纯发射」表。

### 2.2 sorted-blending 模式下的 iso-opacity 重建

在 sorted-blending(排序混合)模式下,VKGS **不**把粒子集当全体积介质,而是认为「即便体积,它主要描述表面」,于是从多层粒子的体积信息里**推导出一个表面**,把积分辐射与积分法线绑定到这个表面上做光照(`docs/deep-dives/lighting_and_shadows.md:25`)。

核心是 **iso-opacity(等积分不透明度)面**:沿光线做 **front-to-back** 遍历,透过率从 1.0 开始,每命中一个 splat 就下降;取**首个使透过率跌破阈值** `depthIsoThreshold`(默认 0.7,即积分不透明度 0.3)的那个粒子的距离作为表面距离(`docs/deep-dives/lighting_and_shadows.md:29`, `docs/deep-dives/lighting_and_shadows.md:33`)。若阈值未被触及,该像素贡献被丢弃(仅当需要深度时才丢;不需要时全部保留)。

**为何不能直接积分深度?** lighting 深潜文档专门给出反例(`docs/deep-dives/lighting_and_shadows.md:27`):按 alpha 加权粒子距离得到的加权距离没有物理意义;两组分离粒子的距离平均会落在任何地方,唯独不在最近物体的前表面;floater(漂浮噪点)也会污染积分距离。iso-opacity 反而能缓解 floater——调低阈值可把 iso 面推到下一层高斯(`docs/deep-dives/lighting_and_shadows.md:37`)。

**光线追踪路径的实现**(`shaders/threedgrt_raytrace.rgen.slang`):

- 法线沿途按权累积:`pixel.integratedNormal = pixel.integratedNormal + normalWorld * weight`(`shaders/threedgrt_raytrace.rgen.slang:887`,仅在 `NEED_SURFACE_INFO` 下编译)。
- 深度拾取的触发条件与写入(`shaders/threedgrt_raytrace.rgen.slang:891-903`):

```cpp
bool shouldUpdateDepth = (pixel.isoSurfDepth == 0.0 && pixel.transmittance.x < double(frameInfo.depthIsoThresholdRTX));
if(shouldUpdateDepth)
{
  pixel.isoSurfSplatSetId = descIdx;
  pixel.isoSurfSplatId    = splatId;
  pixel.isoSurfDepth      = dist;
  pixel.isoSurfPos        = rayOrigin + dist * rayDirection;   // ← surfel 的世界坐标
  ...
}
```

`isoSurfPos`(`:897`)就是重建出的 **surfel 世界坐标**,`integratedNormal`(`:887`)是它的着色法线,`isoSurfSplatSetId`(`:894`)用于回查该 splat set 的材质。RTX 路径用的阈值是独立参数 `depthIsoThresholdRTX`(默认 0.7,`src/parameters.h:260`),与光栅路径的 `depthIsoThreshold` 是两个值(因为两种方法的逐粒子命中距离不同,`docs/deep-dives/lighting_and_shadows.md:29`)。

### 2.3 stochastic 模式:偏体积、面不稳定

在 Stochastic Splat(光栅)/ Stochastic Any Hit(光追)模式下,直接用被随机透明度**选中的那个粒子**的距离,不做 iso 拾取(`docs/deep-dives/lighting_and_shadows.md:41`)。由于没有 MSAA,每帧只有一个样本、选一个粒子,光照逐样本计算、再靠 temporal 累积平均。因此表面在时间上不稳定,得到的更像**体积光照**而非表面光照(`docs/deep-dives/lighting_and_shadows.md:41`)。

Stochastic Pass(光追)介于两者之间:丢弃部分 pass,只保留被选中 pass,在该 pass 内部用 iso-opacity 拾取;若阈值未达到,退回用该 pass 最远命中粒子的距离作为 fallback(`docs/deep-dives/lighting_and_shadows.md:45`)。对应代码见 `shaders/threedgrt_raytrace.rgen.slang:931-963`(`RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_PASS_STOCHASTIC` 分支,fallback 到最远命中在 `:938-951`)。

### 2.4 单粒子命中距离(surfel 落点的粒度)

同一根「重建光线」上,每个粒子贡献一个候选距离,其定义取决于加速结构几何模式(`docs/deep-dives/lighting_and_shadows.md:47-71`):

| 模式 | 距离定义 |
|---|---|
| Icosahedron mesh | 硬件 ray–triangle 求交给出 `RayTCurrent()` = 到 icosahedron 代理**前表面**的距离(近似椭球外壳) |
| AABB parametric | 自定义 intersection shader 用 `particleDensityHitInstance()` 算**最大密度点**(光线上离高斯中心最近点),`t = -(o_c·d_c)/(d_c·d_c)` |
| Billboard(3DGS/3DGUT) | 光线–billboard 平面交点,匹配光栅深度;hybrid 管线需要它把 RTX 主光线对齐 3DGS 训练目标 |
| 光栅 | 过粒子中心的屏幕对齐 billboard 深度,反变换回世界坐标 |

粒子深度模式由 `particleDepth` 参数控制(默认 `PARTICLE_DEPTH_ELLIPSOID`,`src/parameters.h:262`;取值定义见 `shaders/shaderio.h:184-186`)。

### 2.5 法线来源

每个粒子的法线从高斯协方差导出(`NormalMethod`:max-density-plane 或 iso-surface);沿光线按 alpha 加权累积成 `integratedNormal`(`:887`)。光栅的 front-to-back 通道用 interlocked(原子)操作 + 中间图像缓冲,在 fragment 阶段维护 running 透过率与拾取深度(`docs/deep-dives/lighting_and_shadows.md:73-112`,实现见 `shaders/threedgs_raster.frag.slang`)。FTB 只在需要拾取面时启用,其余情况用更快的 back-to-front(`docs/deep-dives/lighting_and_shadows.md:75`)。

### 2.6 关键澄清:surfel 是「接收点」,不是「求交面」

这是整套机制最容易误解的一点:

> **surfel 是被着色的接收点**——它提供:着色法线(`integratedNormal`)、阴影光线的起点(`isoSurfPos`)、以及 BSDF / NEE 的正面(front-face)判定。
> **surfel 不是阴影光线的求交面。** 阴影光线打的是**真实几何**——mesh 三角形 + 高斯代理体的透过率积分(见第 4 节)。

换言之,「逐光线重建一个点」用于**决定这一像素在哪里、朝哪、被什么材质接收光**;而遮挡测试是另一套完全独立的、体积化的光线遍历。二者不共用同一张面。

---

## 3. 着色管线:baseRadiance / emissive / albedo / PBR,与 mesh 统一

粒子着色的入口是 `evaluateLightingAndShadingParticles`(`shaders/threedgrt_raytrace.rgen.slang:1387`)。其逻辑链条如下(行号均在该文件):

**(1) 取回训练辐射并转线性空间。** splat 辐射是针对 sRGB 显示训练的,着色前转线性(`:1401`):

```cpp
const float3 baseRadiance = srgbToLinear(pixel.radiance);
```

**(2) 计算 emissive 贡献。** bounce 0(直接显示)时 `strength = 1.0`,不施加 `emissiveStrength`(训练辐射即 ground truth);二次弹射(作为其他表面的光源)时才用 `emissiveStrength` 缩放(`:1407-1408`):

```cpp
float  strength             = (bounce == 0) ? 1.0 : splatSetDesc.material.emissiveStrength;
float3 emissiveContribution = baseRadiance * splatSetDesc.material.emissive * strength;
```

**(3) 纯发射材质早退,退化为原版 3DGS。** 若 `needShading != 1`(默认),直接把 `emissiveContribution` 当作最终辐射返回,不做任何光照(`:1426-1431`):

```cpp
if(splatSetDesc.material.needShading != 1)
{
  pixel.radiance      = emissiveContribution;
  pixel.transmittance = savedTransmittance;
  return;
}
```

**(4) 否则进入 PBR 着色。** 关键一步:烘焙辐射既当**发射色**又当 **albedo**(`:1446`):

```cpp
mat.baseColor  = baseRadiance * splatSetDesc.material.baseColor;   // 烘焙色 × baseColor 作为 albedo
pixel.radiance = emissiveContribution;                            // 辐射先只放 emissive,光照往上加
```

随后:
- `pixel.transmittance = double3(1.0)`(`:1455`):baseRadiance 已编码高斯混合,着色用单位透过率,不是体积合成透过率。
- `toPbrMaterial(...)`(`:1458`)转成 GGX 的 `PbrMaterial`(转换实现见 `shaders/shading.h.slang:53-77`,含 metallic-roughness、KHR_materials_specular、clearcoat、`transmission → thickness` 推导)。
- **NEE**:`sampleOneLightNEE(...)`(`:1462`)随机选一盏 punctual 光或环境光(实现 `shaders/shading.h.slang:550-649`,按 0.5/0.5 权重在灯与环境间选)。正面判定 `neeFrontFace = dot(nee.direction, integratedNormal) > 0`;非正面且不透射则该样本无效(`:1467-1469`)。
- **GGX 评估 + MIS**:`bsdfEvaluate`(`:1477`),MIS 权重 `nee.pdf/(nee.pdf+evalData.pdf)`(DIRAC 光源权重 1)(`:1481-1482`)。
- **阴影**:仅当 `neeContribution` 有效且亮度 > 0.001 时打阴影光线(`:1487-1493`,详见第 4 节),`neeContribution *= shadow.transmittance`。
- 累加:`pixel.radiance += float3(pixel.transmittance) * neeContribution`(`:1494`)。
- **BSDF 弹射**:受逐材质 `maxBounces` 门控(`:1497`);`bsdfSample` 采样出射方向,把新光线原点设为 `isoSurfPos`,携带 `bsdf_over_pdf` 作为吞吐(`:1497-1512`)。

**与 mesh 着色完全统一。** mesh 走 `evaluateLightingAndShadingMeshes`(`:1523`),结构一模一样:先无条件加 emissive `pixel.radiance += transmittance * meshMaterial.emissive * meshMaterial.emissiveStrength`(`:1555`);`needShading != 1` 早退(`:1558-1562`);否则同样 `toPbrMaterial` + `sampleOneLightNEE`(`:1571`)+ `bsdfEvaluate`(`:1586`)+ MIS + shadow ray(`:1599`)+ `bsdfSample` 弹射(`:1606`)。两条路径共享同一套材质结构、同一个 NEE、同一个 `traceShadowRayForLight`。

---

## 4. 阴影:合并 mesh + particle,双端投影,只乘 NEE 项

### 4.1 合并入口:`traceShadowRayForLight`

单盏灯的遮挡由 `traceShadowRayForLight`(`shaders/threedgrt_raytrace.rgen.slang:1684`)统一处理,最终透过率是 **mesh 透过率 × particle 透过率**(`:1714`):

```cpp
result.transmittance = meshShadow.transmittance * particleShadow.transmittance;
```

- 先打 mesh(通常更少、更快剔除,`:1694-1702`)。
- 只有 mesh 未完全遮挡(`maxComponent(meshShadow.transmittance) > 0.001`)且场景有粒子时,才继续打 particle(`:1708-1713`)。注释明确:stochastic anyhit 下 anyhit 里 alpha=1 会让透过率恒为 0,所以要用这个短路条件保护。

### 4.2 mesh 阴影:`traceShadowRayMesh`

`traceShadowRayMesh`(`:1721`)对 `meshTlasAddress` 打光线,标志 `ACCEPT_FIRST_HIT_AND_END_SEARCH | CULL_BACK_FACING_TRIANGLES`(`:1736`)。命中后:不透明材质 → 透过率 0(全黑阴影);透射材质 → 透过率 = 该材质 `baseColor`(带色阴影,玻璃染色)(`:1747-1753`):

```cpp
result.transmittance = material.transmission > 0.0 ? material.baseColor : float3(0.0);
```

### 4.3 粒子阴影:`traceShadowRayParticle`

`traceShadowRayParticle`(`:1762`)是体积化的核心:

- **偏移起点**防自遮挡(粒子体积性所需较大偏移):`offsetShadowOrigin = shadowOrigin + shadowDir * particleShadowOffset`(`:1774`)。
- 光线标志 `CULL_BACK_FACING_TRIANGLES | SKIP_CLOSEST_HIT_SHADER`(`:1796`);`payload.rayBounce = 1`,非零表示阴影光线始终走 3DGRT 内核(`:1799`);对所有粒子 TLAS 遍历(多 TLAS 支持大场景,`:1800`)。
- **两种策略**:
  - **stochastic anyhit(averaging)**:anyhit 里已算好 alpha/radiance,用**平均**而非积分:`avgAlpha = shadowAlphaSum / count`,`transmittance = saturate(1 - avgAlpha)`(`:1802-1827`)。
  - **full-pass(integration)**:对每个命中粒子调 `threedgrtProcessHit` + `threedgrtIntegrate` 做**彩色透过率积分**(`:1857-1862`);早停条件 `maxComponent(shadowTransmittance) < particleShadowTransmittanceThreshold` → 置 0 并 break(`:1864-1868`)。
- **阈值 + 染色后处理**(`:1871-1880`):把 `TRANSMITTANCE_THRESHOLD` 当硬截断——T ∈ [0, threshold] → 黑;T ∈ (threshold, 1) → 彩色透射区;T = 1 → 无阴影。`particleShadowColorStrength` 控制逐通道染色强度(暗通道更暗、亮通道更快爬升),效果在 scaledT=1 处淡出以免给全亮区上色。

### 4.4 双端投影:mesh ↔ splat 互投 + 自投影

两个着色函数都在 surfel / mesh 命中点起阴影光线,而阴影光线又同时穿 mesh 和粒子,于是形成**双向体积互投**:

- 粒子着色在 `shadowOrigin = rayOrigin + isoSurfDepth * rayDirection`(即 `isoSurfPos`)起阴影光线(`:1451`,调用 `:1490`)→ 粒子可被 mesh 和其他粒子遮挡。
- mesh 着色在 `shadowOrigin = pixel.meshHitWorldPos` 起阴影光线(`:1564`,调用 `:1599`)→ mesh 可被粒子和其他 mesh 遮挡。

因此 **mesh 会在 splat 上投影、splat 会在 mesh 上投影,且各自自投影**。

### 4.5 阴影只乘 NEE 项,emissive 不受影响(关键!)

阴影**只**衰减 NEE(直接光照)贡献:

```cpp
neeContribution *= shadow.transmittance;      // rgen.slang:1491(粒子) / :1600(mesh)
```

而 `emissiveContribution` 是在着色一开始就**无条件**加进 `pixel.radiance` 的(粒子 `:1408`+`:1448`;mesh `:1555`)。源码注释点明这一设计意图(`shaders/threedgrt_raytrace.rgen.slang:1414-1416`):

> "the PBR path handles light occlusion via NEE shadow rays, but **the emissive term has no other occlusion**."

**推论:纯发射的辐射场收不到投影阴影。** 默认 splat set 的辐射 100% 来自 emissive,阴影只乘 NEE 项,而对纯发射材质 NEE 根本不会被计算(在 emissive 加完后就早退了,`:1426`),所以无论怎么打灯、怎么开阴影,一个纯发射 splat set 的外观都不会变暗。要让它接收阴影,必须让它有非零的 NEE 贡献,即让它「参与着色」(见第 7 节)。

**唯一的例外**是 emissive AO:若编译期启用 `PARTICLE_AO_ENABLED`,`computeSplatEmissiveAO`(`:1633-1680`)会对 mesh 邻近做余弦加权半球采样 AO,以 `aoFactor` 衰减 emissive(`:1416-1421`)。但这是对 mesh 几何的**环境遮挡**(只打 mesh TLAS,`:1671`),不是光源投影阴影,且默认关闭(`particleEmissiveAoEnabled = false`,`src/parameters.h:252`)。

### 4.6 软阴影:单抖动 disk 样本 + temporal 收敛

`SHADOWS_MODE` 有三档(`shaders/shaderio.h:189-191`):

| 值 | 宏 | 含义 |
|---|---|---|
| 0 | `SHADOWS_DISABLED` | 不打阴影光线 |
| 1 | `SHADOWS_HARD` | 硬阴影(点采样) |
| 2 | `SHADOWS_SOFT` | 软阴影(围绕光源 disk 采样) |

软阴影**不是**单帧内循环多样本,而是**每帧只抖动一个 disk 样本**,靠 temporal 累积逐帧收敛。实现见 `computeLightToSurfaceVector`(`shaders/shading.h.slang:218-255`):在 `SHADOWS_SOFT` 下,若 `light.radius > 0`,在垂直于「表面→光源」方向的 disk 上采一个随机点作为本帧光源采样位置(`:231-241`):

```cpp
#if SHADOWS_MODE == SHADOWS_SOFT
  if(light.radius > 0.0f)
  {
    float3 toLight = normalize(light.position - worldPos);
    float3 tangent, bitangent;
    buildOrthonormalBasis(toLight, tangent, bitangent);
    lightSamplePos += sampleDisk(seed, light.radius * 0.5f, tangent, bitangent);
  }
#endif
```

辅助函数 `buildOrthonormalBasis`(Frisvad 法,`:263-277`)、`sampleDisk`(极坐标均匀 disk 采样,`sqrt(rand)` 保证面积均匀,`:280-286`)。disk 世界半径来自 `light.radius`(即光源代理体 `proxyScale`,`shaders/shading.h:127`)。半影随 `radius` 变宽,收敛由帧累积完成。

### 4.7 光源模型

`LightSource`(`shaders/shading.h:116-129`):

| 字段 | 默认 | 说明 |
|---|---|---|
| `type` | `ePointLight`(1) | 0=方向光 / 1=点光 / 2=聚光 |
| `color` | `(1,1,1)` | RGB |
| `intensity` | `1.0` | |
| `position` | `(0,0,0)` | 世界位置(来自 instance) |
| `range` | `10.0` | 点/聚光有效距离 |
| `direction` | `(0,0,-1)` | 由旋转算出 |
| `innerConeAngle` / `outerConeAngle` | `30` / `45`(度) | 聚光内外锥角 |
| `attenuationMode` | `2` | 0=无 / 1=线性 / 2=二次 / 3=物理 |
| `radius` | `1.0` | 软阴影光源半径(来自 proxyScale) |
| `enabled` | `1` | |

衰减公式见 `computeLightRadiance`(`shaders/shading.h.slang:418-452`):线性 `1-d/range`、二次 `1/(1+d²)`、物理 `1/(d²+0.01)`。方向光无衰减、`lightDist` 取 1e10(`:220-226`)。无灯时 NEE 会退化(`sampleOneLightNEE` 里 `wLight=0`,`shaders/shading.h.slang:568`),此外 `createHeadlight`(`shaders/shading.h.slang:291-304`)可在无灯时提供相机位随动的默认头灯。

### 4.8 阴影参数表

| 参数 | 默认 | 位置 | 作用 |
|---|---|---|---|
| `particleShadowOffset` | `0.2` | `src/parameters.h:248` | 粒子阴影光线起点偏移(防自遮挡) |
| `particleShadowTransmittanceThreshold` | `0.8` | `src/parameters.h:249` | 粒子阴影透过率早停/黑截断阈值 |
| `particleShadowColorStrength` | `0.0` | `src/parameters.h:250` | 粒子彩色阴影染色强度(0=单色,1=全彩) |
| `depthIsoThresholdRTX` | `0.7` | `src/parameters.h:260` | RTX 路径 iso-opacity 深度拾取阈值 |
| `particleEmissiveAoEnabled` | `false` | `src/parameters.h:252` | 是否启用 emissive AO(编译期宏门控) |

---

## 5. 管线适用性

VKGS 有 6 条管线(`shaders/shaderio.h:103-108`)。**光线追踪阴影只在使用 RT 的管线里可用**,即 `isRtxPipelineActive()` 返回真的三条:RTX、HYBRID、HYBRID_3DGUT(`src/gaussian_splatting.h:170-173`)。

| 值 | 宏 | 类型 | 是否有 RT 阴影 |
|---|---|---|---|
| 0 | `PIPELINE_VERT` | 纯光栅(3DGS,顶点管线) | 否 |
| 1 | `PIPELINE_MESH` | 纯光栅(mesh shader) | 否 |
| 2 | `PIPELINE_RTX` | 纯光线追踪 | 是 |
| 3 | `PIPELINE_HYBRID` | 光栅主光线(3DGS)+ 光追次光线(3DGRT) | 是 |
| 4 | `PIPELINE_MESH_3DGUT` | 3DGUT 光栅(mesh shader) | 否 |
| 5 | `PIPELINE_HYBRID_3DGUT` | 光栅主光线(3DGUT)+ 光追次光线(3DGRT) | 是 |

- 宏 `HYBRID_ENABLED` = `(PIPELINE_HYBRID || PIPELINE_HYBRID_3DGUT)`,在 `src/gaussian_splatting.cpp:2284-2285` 写入。
- 光栅路径(0/1/4)的光照走 deferred shading(`m_computePipelineDeferredShading`,`src/gaussian_splatting.cpp:1044-1045`),用 `wavefrontComputeShadingDirectOnly`(`shaders/shading.h.slang:463-495`),其遮挡是一个布尔 `inShadow`(`:472-473` 直接 return),**不打 RT 阴影光线**——纯光栅管线拿不到 mesh↔splat 体积互投阴影。
- CPU 端也据此决定是否需要重建 RTX 加速结构:`shadowsMode == eShadowsSoft && isRtxPipelineActive() && lightingEnabled`、或 `isRtxPipelineActive() && lightingEnabled` 等条件(`src/gaussian_splatting.cpp:432-435`)会触发 RTX 状态就绪。

---

## 6. 与 3dgrut 的对比(简述)

参考 `/home/lxruan/dev/3dgrut`(`threedgrut_playground/include/playground/kernels/cuda/trace.cuh`、`src/kernels/cuda/playgroundKernel.cu`)。两点主要差异:

| 维度 | 3dgrut | VKGS |
|---|---|---|
| 阴影光线打谁 | **只打 mesh BVH**;高斯不参与投影 | **mesh + 高斯双向体积互投**(`rgen.slang:1714`) |
| shadow catcher | 有(`pendingShadowVis` 延后处理,绕开 OptiX `maxTraceDepth=1`) | **无** shadow catcher |

即:3dgrut 里高斯不投影、且有专门的 shadow catcher 机制来在有限 trace 深度下承接阴影;VKGS 走完整的 mesh↔splat 体积互投,但没有 shadow catcher。

---

## 7. 操作清单:让某个 splat set 被打光 + 投影

要把一个默认(纯发射)splat set 变成能接收合成光并投射/接收阴影的对象,需要:

| 步骤 | 操作 | 原因 |
|---|---|---|
| ① | 把 `baseColor` 抬到非黑(和/或 `metallic` / `transmission` 非零) | 让 `needShading` 判定成立(`shaders/shading.h:98-102`),否则纯发射早退(`rgen.slang:1426`) |
| ② | 调低 `emissive` | 否则 emissive 无条件加进辐射(`:1408`),会把阴影/光照冲淡到看不见 |
| ③ | `lightingEnabled = 1` | 触发 `NEED_SURFACE_INFO`(`src/gaussian_splatting.h:221`),开始重建 surfel |
| ④ | 至少 1 盏启用的灯 | NEE 需要光源,否则 `wLight=0`(`shaders/shading.h.slang:568`) |
| ⑤ | 选 RTX / HYBRID / HYBRID_3DGUT + `shadowsMode ≠ disabled` | RT 阴影只在这三条管线(`src/gaussian_splatting.h:170-173`);阴影光线受 `SHADOWS_MODE != SHADOWS_DISABLED` 门控(`rgen.slang:1489`) |
| ⑥ | (可选)`maxBounces > 0` | 需要反射/折射弹射时,默认 splat 材质 `maxBounces=0`(`src/splat_set_manager_vk.cpp:261`) |

**原理速记**:`albedo = 烘焙色 × baseColor`(`:1446`);`emissive 项 = 烘焙色 × emissive × strength`(`:1408`)。二者此消彼长——抬 baseColor 是给它 albedo 让光打得上,降 emissive 是给阴影腾出可见的对比度。

---

## 8. 局限与两条路

**局限:想「保留烘焙外观、又叠加阴影」,现状不行。**
因为阴影只乘 NEE 项、emissive 无条件加(`:1414-1416`),纯发射辐射场收不到投影阴影(见 4.5)。若走物理路径(降 emissive、抬 baseColor),就不再是原始训练外观,而是重新用合成光点亮的结果。两条可选路径:

**路径 A(现状可行,物理化)**:按第 7 节降 `emissive`、抬 `baseColor`,让 splat set 走 PBR 着色。代价是外观偏离烘焙原貌,依赖合成光质量。

**路径 B(需改 shader,辐射场版 shadow catcher)**:加一小段改动,让阴影可见性去调制 emissive,例如:

```cpp
// 概念示意,非现有代码
emissiveContribution *= shadowMin + (1.0 - shadowMin) * visibility;
```

其中 `visibility` 由一次(对 mesh、或对全部几何的)阴影光线得到,`shadowMin` 是阴影最暗保底。这样在**保留烘焙色**的同时,让辐射场也能被投影阴影压暗——本质上是「辐射场版的 shadow catcher」。这需要在 `evaluateLightingAndShadingParticles` 里(`emissiveContribution` 计算之后、`needShading` 早退之前,即 `:1408` 与 `:1426` 之间)插入一次阴影可见性查询,并放宽早退逻辑,让纯发射材质也能进入这段调制。

---

### 附:关键源码位置速查

| 主题 | 位置 |
|---|---|
| `NEED_SURFACE_INFO` 门控 | `src/gaussian_splatting.h:219-229` / 宏写入 `src/gaussian_splatting.cpp:2326` |
| `needShading` 推导 | `shaders/shading.h:91-104` |
| splat 默认纯发射材质 | `src/splat_set_manager_vk.cpp:253-261` |
| iso-opacity 深度拾取(RTX) | `shaders/threedgrt_raytrace.rgen.slang:887-903` |
| 粒子着色主逻辑 | `shaders/threedgrt_raytrace.rgen.slang:1387-1518` |
| mesh 着色主逻辑 | `shaders/threedgrt_raytrace.rgen.slang:1523-1626` |
| 阴影合并入口 | `shaders/threedgrt_raytrace.rgen.slang:1684-1717` |
| mesh 阴影 | `shaders/threedgrt_raytrace.rgen.slang:1721-1757` |
| 粒子阴影 | `shaders/threedgrt_raytrace.rgen.slang:1762-1884` |
| emissive AO | `shaders/threedgrt_raytrace.rgen.slang:1633-1680` |
| 软阴影 disk 采样 | `shaders/shading.h.slang:218-287` |
| NEE 采样 | `shaders/shading.h.slang:550-649` |
| `LightSource` / `Material` | `shaders/shading.h:29-129` |
| 阴影/光照参数 | `src/parameters.h:248-260` |
| 管线枚举 | `shaders/shaderio.h:103-108` |
| RTX 管线判定 | `src/gaussian_splatting.h:170-173` |
| `HYBRID_ENABLED` 宏 | `src/gaussian_splatting.cpp:2284-2285` |
| 权威深潜文档 | `docs/deep-dives/lighting_and_shadows.md` |

---

## 附:feat/shadow-mask 已实现「辐射场版 shadow catcher」

本文第 8 节所述的「保留烘焙外观又叠加阴影」局限,已在分支 `feat/shadow-mask` 落地为正式功能(GS shadow mask):

- **灯光标志** `LightSource.shadowOnly`(GUI「GS shadow only」/ .vkgs `lights.assets[].shadowOnly`):该灯仅产生阴影 mask,被 NEE 与所有直接光循环跳过,不照亮任何物体。
- **渲染选项** `gsShadowMask`(+`gsShadowMaskMin` 阴影下限、`gsShadowMaskFromParticles` 粒子遮挡开关、独立的 `forceSurfel`):开启后隐含 LIGHTING_MODE 编译与 surfel 重建;纯发射 splat set 视觉不变,emissive 项在 needShading 早退前被 `computeGsShadowMask`(rgen.slang,以 `computeSplatEmissiveAO` 为骨架)乘暗。
- **语义**:多灯按 intensity×luminance 加权平均可见度(重叠阴影不叠乘);遮挡源默认 mesh-only(避免接收 GS 自遮挡),粒子遮挡 opt-in;`SHADOWS_MODE=SOFT` + 灯 radius>0 时自动获得软阴影(temporal 收敛),radius=0 恒为零噪声硬阴影。
- **范围**:仅 RT 管线 pipeline 2/3/5;Python 层 `LightAsset.shadow_only`、`RendererSettings.gs_shadow_mask*` 已同步;compat shim 的 3dgrut `SHADOW_CATCHER` 自动映射到本功能(CompatWarning 说明近似)。
