# Python 操作层 `vkgs` 端到端使用指南

本指南面向 `feat/py-binding` 分支新增的 Python 操作层 `vkgs`(位于仓库 `python/`),
覆盖**从构建可执行文件、安装 Python 包,一直到用 CLI command 编排渲染**的完整流程。
所有 API 签名、命令、`.vkgs`/`.cfg` 文本均已对照 `python/vkgs/` 下的实际源码核实。

> 术语约定:代码、命令、路径、标识符一律保留英文;正文说明用中文。
> 文中出现的“exe / 可执行文件”均指 `vk_gaussian_splatting`(Vulkan 渲染器本体)。

---

## 目录

1. [概述与架构](#1-概述与架构)
2. [构建 VKGS 可执行文件](#2-构建-vkgs-可执行文件)
3. [安装 vkgs Python 包](#3-安装-vkgs-python-包)
4. [快速上手:最小可跑脚本](#4-快速上手最小可跑脚本)
5. [Native API 详解](#5-native-api-详解)
6. [CLI command 编排详解(重点)](#6-cli-command-编排详解重点)
7. [视频渲染](#7-视频渲染)
8. [3dgrut 兼容 shim(EngineVKGS)](#8-3dgrut-兼容-shimenginevkgs)
9. [测试与验证](#9-测试与验证)
10. [常见问题与边界](#10-常见问题与边界)

---

## 1. 概述与架构

`vkgs` 不是 pybind/FFI 绑定,而是一层**纯「文件 + 子进程」编排**:Python 侧只负责
构造场景、序列化文件、拼命令、跑子进程、把渲染结果读回 numpy。渲染完全交给已有的
`vk_gaussian_splatting` 可执行文件在**无头(headless)**模式下完成。没有 pybind、
没有共享内存、没有 IPC。

```
┌─────────────────────────── Python (vkgs) ───────────────────────────┐
│                                                                      │
│  Scene (project.py)                RenderScript (sequence.py)         │
│    add_splats / add_mesh             load_block()                     │
│    add_light / set_environment       capture(cam, out)  ─┐            │
│    add_camera_preset(cam) -> idx                         │            │
│         │                                                │            │
│         ▼  Scene.save()                                  ▼  .write()  │
│   out_dir/scene.vkgs  (JSON v7)              out_dir/render.cfg       │
│         │                                                │            │
│         └───────────────┬────────────────────────────────┘           │
│                         ▼   HeadlessRunner.run() (runner.py)          │
│        subprocess: vk_gaussian_splatting --headless 1 --benchmark 1   │
│                    --sequencefile render.cfg --inputProject scene.vkgs│
│                         │                                             │
│                         ▼  exe 跑到 sequencer 结束后退出               │
│        out_dir/cam0_main.png  cam0_depth.png  ...   render.log        │
│                         │                                             │
│                         ▼   images.load_image() (images.py)           │
│                    numpy array (H, W, C)                              │
└──────────────────────────────────────────────────────────────────────┘
```

一次 `render_scene()` 内部严格按这个顺序执行(`facade.py`):

1. `Scene.save(out_dir/scene.vkgs)` —— 写出工程文件(含相机预设);
2. `RenderScript` 生成 `out_dir/render.cfg` —— 每台相机一对「渲染序列 + 保存序列」;
3. `HeadlessRunner.run()` 拉起一个子进程跑到序列结束、检查日志与产物;
4. `images.find_outputs()` 把带后缀的输出文件映射回 numpy(懒加载 + 缓存)。

这套设计的直接好处:GUI 里手工搭好的场景可以 `Scene.load()` 进来在 Python 里后处理,
Python 里搭的场景也能 `Scene.save()` 出去在 GUI 里打开——两侧读写的是**同一个** v7
工程格式(`project.py` 的 dataclass 树与 `src/vkgs_project_writer.cpp` 一一对应)。

---

## 2. 构建 VKGS 可执行文件

渲染必须有一个能跑的 `vk_gaussian_splatting`。Python 层不会替你编译,只会去若干位置
**查找**它(见 [§6.4](#64-可执行文件查找顺序))。下面这套流程已在一台全新的 Ubuntu VM 上
跑通(2026-08-05)。

### 2.0 构建前须知(两个已知踩坑)⚠️

从零构建时最容易卡的两处,先处理好再走 §2.1 起的常规流程:

**坑 1 —— `nvpro_core2` 子模块在 Manifold-ai fork 上克隆失败(404)**

`.gitmodules` 用的是**相对 URL** `../nvpro_core2.git`,git 会相对超级仓库的 origin 解析:
在 `Manifold-ai/vk_gaussian_splatting` fork 上它被解析成 `Manifold-ai/nvpro_core2.git`——
**该仓库不存在**,于是 `git submodule update --init --recursive` 报
`remote: Repository not found`。

本质:Manifold-ai 并没有 fork nvpro_core2,子模块只是 pin 了 **公开的 NVIDIA `nvpro-samples/nvpro_core2`**
的一个 commit(相对 URL 本是给上游用的)。修法——指向公开库即可,**无需任何私有权限**:

```bash
# 只改本地 .git/config,不动 tracked 的 .gitmodules(推荐)
git submodule init
git config submodule.nvpro_core2.url https://github.com/nvpro-samples/nvpro_core2.git
git submodule update --recursive
```

> 永久做法(团队统一,git ≥ 2.25):
> `git submodule set-url nvpro_core2 https://github.com/nvpro-samples/nvpro_core2.git` →
> `git submodule sync --recursive` → `git submodule update --init --recursive`。
> 长期最好由 Manifold-ai 把 `.gitmodules` 改成绝对上游 URL 提交,避免每个 clone 的人都踩。

**坑 2 —— `fatal error: shaderc/shaderc.hpp: No such file or directory`**

`nvvkglsl` 模块需要 shaderc,它期望躺在 Vulkan 的 include 目录里;发行版 `libvulkan-dev`
只带 vulkan 头、**不含 shaderc**(shaderc 随 LunarG Vulkan SDK 提供)。修法:装完整
**LunarG Vulkan SDK**(自带匹配版本的 shaderc/glslang/SPIRV-Tools),让环境生效后**干净重配**:

```bash
# tarball 安装:先 source 让 VULKAN_SDK 生效(apt 的 vulkan-sdk 包装进 /usr,可跳过)
source ~/VulkanSDK/<版本>/setup-env.sh

cmake --fresh -S . -B build          # 关键:旧缓存里是没有 shaderc 的发行版 Vulkan 路径
cmake --build build -j"$(nproc)"
```

> Slang 编译器由 CMake 在 configure 阶段**自动下载**(`cmake/FindSlang.cmake`),不用手动装。
> 重配时 configure 输出里 `ShaderC Import Library : .../libshaderc_shared.so` 显示成路径(非 NOTFOUND)即对。
> implot 的 `-Wdeprecated-enum-enum-conversion` 只是无害警告,忽略。

### 2.1 前提

- NVIDIA GPU / 完整支持 Vulkan 1.4 的 GPU
- [Vulkan 1.4 SDK](https://vulkan.lunarg.com/sdk/home)
- [CMake ≥ 3.22](https://cmake.org/download/)
- 支持 C++20 的编译器(Windows: MSVC 2019+;Linux: GCC 10.5+ 或 Clang)
- Linux 额外系统库:

  ```bash
  sudo apt install libx11-dev libxcb1-dev libxcb-keysyms1-dev libxcursor-dev \
      libxi-dev libxinerama-dev libxrandr-dev libxxf86vm-dev libtbb-dev
  ```

- `nvpro_core2` 是子模块(见 `.gitmodules`)。**在 Manifold-ai fork 上直接 `git submodule
  update --init --recursive` 会 404**,先按 [§2.0 坑 1](#20-构建前须知两个已知踩坑-️) 把子模块
  URL 指向公开的 `nvpro-samples/nvpro_core2` 再拉取。

### 2.2 配置与编译

在**仓库根目录**执行:

```bash
# (若当初 clone 时没加 --recurse-submodules,先补子模块)
git submodule update --init --recursive

# 配置
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 可选:禁止 CMake 下载并自动加载 "bouquet of flowers" 默认场景
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDISABLE_DEFAULT_SCENE=ON

# 编译
cmake --build build --config Release --parallel
```

### 2.3 产物路径

无论 `build` 目录叫什么,可执行文件都会输出到仓库的 **`_bin/{Release,Debug}/`**:

```
_bin/Release/vk_gaussian_splatting        # Linux
_bin/Release/vk_gaussian_splatting.exe    # Windows
```

这正是 `find_executable()` 默认第一优先查找的位置,所以在仓库内构建后,Python 层
**开箱即用**、无需任何额外配置。

### 2.4 用 `$VKGS_BIN` 指定(非默认位置 / 预编译包)

如果用的是 [Releases](https://github.com/nvpro-samples/vk_gaussian_splatting/releases)
预编译包,或把 exe 放在了别处,用环境变量指向它:

```bash
export VKGS_BIN=/path/to/vk_gaussian_splatting
python your_script.py
```

或在代码里显式传 `executable=`:

```python
render_scene(scene, cameras=[0], out_dir="out", executable="/path/to/vk_gaussian_splatting")
```

---

## 3. 安装 vkgs Python 包

要求 Python ≥ 3.9(`pyproject.toml`)。核心只依赖 `numpy`;读图 / 视频 / 测试是可选 extras。

> **本指南默认用 [`uv`](https://docs.astral.sh/uv/) 做包管理**(装得快、自带虚拟环境管理)。
> 没装 uv:`curl -LsSf https://astral.sh/uv/install.sh | sh`(或 `pipx install uv`)。
> 仍想用传统 pip 的,把下文 `uv pip` 换成 `pip`、`uv run` 换成先激活 venv 再直接跑即可。

### 3.1 建虚拟环境

```bash
uv venv                              # 在当前目录建 .venv(默认取系统 Python;指定版本:uv venv --python 3.11)
```

`uv` 不需要手动 `activate`:后续用 `uv pip` / `uv run` 会自动作用到该 `.venv`(也可 `source .venv/bin/activate` 传统方式)。

> 仓库里已带一个开发用 venv:`python/.venv`(维护者跑单测用)。你自己的项目建议在别处另建一个干净的 venv。

### 3.2 可编辑安装 + extras

以「可编辑」方式安装,改源码即时生效(`-e`):

```bash
uv pip install -e python/            # core:仅 numpy(足够构建场景 + 跑 exe + 读 .raw)
uv pip install -e "python/[image]"   # + imageio,读回 .png/.jpg/.hdr
uv pip install -e "python/[video]"   # + imageio + imageio-ffmpeg,组装 mp4
uv pip install -e "python/[dev]"     # + pytest(跑单测)
```

extras 对应关系(`pyproject.toml`):

| extra     | 追加依赖                        | 用途                                   |
|-----------|---------------------------------|----------------------------------------|
| `image`   | `imageio`                       | `load_image` 读 `.png/.jpg/.hdr`       |
| `video`   | `imageio` + `imageio-ffmpeg`    | `render_video` 输出 mp4                |
| `dev`     | `pytest` + `imageio`            | 跑 `python/tests`                      |

> 注意:`.raw`(float32 RGBA)读回**不需要** imageio,`images._load_raw` 用纯 numpy 解析;
> 只有 `.png/.jpg/.hdr` 需要 `[image]`。

### 3.3 验证安装

```bash
python -c "import vkgs; print(vkgs.__version__)"      # 0.1.0
python -c "from vkgs import Scene, Camera, render_scene, materials; print('ok')"
```

---

## 4. 快速上手:最小可跑脚本

下面是从零到出图的最短路径(需要已构建 exe + 一个 3DGS `.ply`):

```python
from vkgs import Scene, Camera, Pipeline, render_scene

scene = Scene()
scene.renderer.pipeline = Pipeline.MESH        # 默认光栅管线,最快最稳
scene.add_splats("garden.ply")                 # .ply / .spz / .splat

# 相机以 eye/ctr/up + 垂直 fov(度)描述;add_camera_preset 返回预设索引
idx = scene.add_camera_preset(Camera(eye=(1.7, 1.5, 1.7), ctr=(0, 0, 0), fov=60))

result = render_scene(
    scene,
    cameras=[idx],                # 也可直接传 Camera 对象:cameras=[Camera(...)]
    size=(1280, 720),
    spp=32,                       # 每台相机累计的帧数(sequenceframes)
    out_dir="out",
)

rgb = result.image(camera=0)      # numpy 数组 (H, W, 4) uint8(.png = RGBA8)
print(rgb.shape, result.path(camera=0))   # 打印形状与落盘路径
print("log:", result.log_path)
```

`render_scene` 的 `cameras` 参数很灵活:可以是**单个** `int`/`Camera`,也可以是它们的
**混合列表**。传 `Camera` 对象时会被自动追加到 scene 的一个浅拷贝里(不改你的原 scene)。

一个更完整、带命令行参数的例子见 `python/examples/01_build_and_render.py`。

---

## 5. Native API 详解

导出的公共 API 全集见 `python/vkgs/__init__.py` 的 `__all__`。下面按主题拆解,签名均来自源码。

### 5.1 Scene:场景容器与构建方法

`Scene`(`project.py`)是一个可脚本化的场景容器,构造后用 `add_*` 往里加东西,最后 `save()`。
它内部持有:`renderer`(`RendererSettings`)、`splats_globals`、`splat_sets`/`splat_instances`、
`mesh_assets`/`mesh_instances`、`light_assets`/`light_instances`、`camera`(活动相机)、
`camera_presets`、`environment`、`settings`、`tonemapping`。

#### add_splats —— 加 3DGS 点云

```python
scene.add_splats(
    path,                       # .ply/.spz/.splat;按绝对路径去重(同一文件只占一份 GPU)
    *,
    name=None,                  # 缺省 "Splat set {N} - {basename}"
    position=(0,0,0),
    rotation=(0,0,0),           # Euler XYZ 度(GLM 约定,见 §5.5)
    scale=(1,1,1),              # 支持标量(各向同性)或 (sx,sy,sz)
    material=None,              # 缺省 Material.splat_default()
    show=True,
    sh_format=None,             # 缺省取 splats_globals.sh_format
    rgba_format=None,
) -> SplatInstance
```

> splat 的默认材质(`Material.splat_default()`)是 `base_color=(0,0,0)`、
> `emissive=(1,1,1)`、`max_bounces=0`,即“照原色发光、不参与二次弹射”,对应 C++
> reader 给 splat 实例套的默认值。splat **不带贴图**(见 [§10](#10-常见问题与边界))。

#### add_mesh —— 加网格(玻璃/镜面/PBR 物体)

```python
scene.add_mesh(
    path,                       # .glb/.gltf/.obj;按路径去重
    *,
    name=None,                  # 缺省 "Model {N}"
    position=(0,0,0), rotation=(0,0,0), scale=(1,1,1),
    materials=None,             # 按顺序覆盖文件里的材质槽;空/省略 = 保留文件自带材质
    show=True,
) -> MeshInstance
```

`materials` 是一个 `Material` 列表,**按槽位顺序**覆盖网格文件里的材质;短于槽位数时,
剩余槽位保持文件里的原始材质。

#### add_light —— 加灯(创建 asset + 一个 instance)

```python
scene.add_light(
    type=LightType.POINT,       # DIRECTIONAL=0 / POINT=1 / SPOT=2
    *,
    name=None, color=(1,1,1), intensity=100.0,
    translation=(0,2,0),        # 注意:灯用 translation/rotation,不是 position
    rotation=(0,0,0),           # Euler 度;灯朝向 = R @ (0,0,-1)
    range=10.0, radius=1.0,     # radius>0 参与软阴影
    inner_cone_angle=30.0, outer_cone_angle=45.0,
    attenuation_mode=AttenuationMode.QUADRATIC,
    enabled=True,
) -> LightInstance
```

同一个灯 asset 想复用多个实例,用 `add_light_instance(asset, translation=..., rotation=...)`。

> 灯要真正生效,通常需要 `scene.renderer.lighting_enabled = True`;
> 完整光照/阴影效果在路径追踪管线(RTX/HYBRID/HYBRID_3DGUT)下最准。

#### set_environment / set_tonemapping / set_camera / add_camera_preset

```python
# 环境:天空 或 HDR IBL
scene.set_environment(
    mode=EnvMode.SKY,           # NONE=0 / SKY=1 / HDR=2
    *,
    hdr_file="",                # mode=HDR 时才加载(绝对化)
    ibl_intensity=1.0, ibl_rotation=(0,0,0),
    sun_direction=None,         # 便捷入口;其余 sky 参数用 **sky_params 关键字传
    **sky_params,               # 例如 haze=0.2, ground_color=(...);未知键会抛 TypeError
) -> Environment

# 色调映射:激活后新增 _ldr 输出缓冲
scene.set_tonemapping(active=True, method=0, exposure=1.0, ...) -> Tonemapping

# 设置“活动相机”(headless 下作用不大,主要用于 GUI 打开时的初始视角)
scene.set_camera(camera) -> None

# 追加一个相机预设,返回其索引(= --activateCameraPreset 的值,零偏移)
idx = scene.add_camera_preset(camera) -> int
```

> **关键契约**:`add_camera_preset(cam)` 返回的 `idx` **直接**对应 CLI 的
> `--activateCameraPreset idx`,无任何偏移。C++ reader 加载工程后会清空并按
> `cameras` 数组顺序重建预设,所以顺序即索引。

#### save / load —— v7 工程往返

```python
abs_path = scene.save("out/scene.vkgs")   # 返回绝对路径;后缀非 .vkgs 会被改成 .vkgs
scene2 = Scene.load("out/scene.vkgs")      # 版本 < 5 会拒绝(需用 GUI 迁移)
```

- 写出的 JSON **按键名字母序排序、4 空格缩进**,与 C++ 的 `nlohmann::json` 输出对齐,
  方便两侧 diff。
- 资产路径(splat/mesh/HDR)在文件里存为**相对工程目录**的 POSIX 路径;`load` 时再绝对化。
- 写文件版本号 `PROJECT_FILE_VERSION = 7`;`load` 接受 `>= 5`(`constants.py`)。

> reader 有硬性必填字段(缺了会**整份工程静默加载失败**):`splatSets[].id/.path`、
> `splats[].splatSetId`、`meshAssets[].id/.path`、`meshInstances.items[].meshAssetId`、
> `lights.assets[].id`、`lights.instances[].assetId`。用 `Scene`/`add_*` 构建时这些
> 都会自动写全,手写 JSON 时要特别注意。

一个真实的 `.vkgs` 片段(`add_splats + add_mesh(glass) + add_light + 一个相机预设`
后 `save()` 的实际输出,已核实):

```json
{
    "cameras": [
        { "aperture": 0.001, "clip": [0.1, 2000.0], "ctr": [0.0, 0.5, 0.0],
          "dofMode": 0, "eye": [3.0, 1.5, 2.0], "focusDist": 1.3, "fov": 50.0,
          "model": 0, "up": [0.0, 1.0, 0.0] }
    ],
    "meshInstances": {
        "items": [{
            "materials": [{ "baseColor": [1.0,1.0,1.0], "ior": 1.5, "maxBounces": 8,
                            "metallic": 0.0, "name": "glass", "opacity": 1.0,
                            "roughness": 0.0, "transmission": 1.0, ... }],
            "meshAssetId": 0, "name": "Model 0", "position": [0.0,0.5,0.0], ...
        }], "nextNamingNumber": 2
    },
    "renderer": { "pipeline": 3, "lightingEnabled": true, "rtxMaxBounces": 3, ... },
    "splatSets": [{ "id": 0, "path": "../../data/garden.ply", "rgbaFormat": 2,
                    "shFormat": 2, "storage": 0 }],
    "splats": [{ "splatSetId": 0, "name": "Splat set 0 - garden.ply", ... }],
    "version": 7
}
```

### 5.2 Material 与 materials 预设

`Material`(`project.py`)是 metallic-roughness PBR 因子集合(与 `shaders/shading.h` 对齐):

```python
Material(
    base_color=(0.7,0.7,0.7), metallic=0.0, roughness=0.5,
    emissive=(0,0,0), emissive_strength=1.0, max_bounces=3,
    ior=1.5, transmission=0.0, opacity=1.0,
    specular_factor=1.0, specular_color_factor=(1,1,1),
    clearcoat_factor=0.0, clearcoat_roughness=0.0,
    name=None,                  # 仅 mesh 材质会序列化 name
)
```

`vkgs.materials`(`materials.py`)提供一批**零配置工厂**,命名与 3dgrut playground 一致,
每个都接受 `**overrides` 覆盖任意字段:

- 通用三件套(对应 3dgrut `OptixPrimitiveTypes`):
  - `glass(ior=1.5, transmission=1.0, color=(1,1,1), roughness=0.0)` —— 透明玻璃
    (`max_bounces=8`;要与 playground 未改动的玻璃**完全一致**传 `ior=1.33`)
  - `mirror(color=(1,1,1))` —— 完美镜面(`metallic=1, roughness=0, max_bounces=4`)
  - `diffuse(color=(0.7,0.7,0.7))` —— 纯朗伯(`roughness=1, max_bounces=3`)
- 命名预设(因子严格复刻 3dgrut `register_default_materials`):
  `solid`、`brushed_copper`、`rose_gold`、`blue_plastic`、`oak_wood`、`black_rubber`、
  `polished_marble`、`blue_glass`、`jade`、`diamond`、`ruby_red`、`luminous_yellow`。
- `materials.PRESETS` 是 `name -> factory` 的字典,便于按名取。

```python
from vkgs import materials
scene.add_mesh("sphere.obj", materials=[materials.diamond()])
scene.add_mesh("teapot.obj", materials=[materials.brushed_copper(roughness=0.2)])
```

> 材质是**纯因子**;贴图必须由 mesh 文件自带(glTF 的 PBR 贴图)。3dgrut 的 `checkboard`
> 这种程序化贴图预设**没有**因子等价物,`materials.py` 里也不提供,可用
> `diffuse(color=(0.375,0.375,0.375))` 近似。

### 5.3 RendererSettings 关键项

`scene.renderer`(`RendererSettings`,`project.py`)有几十个字段,全部与 C++ 参数默认值对齐。
最常用的几项:

| 属性                         | 含义 / 取值                                                        |
|------------------------------|-------------------------------------------------------------------|
| `pipeline`                   | `Pipeline`:VERT=0 / MESH=1(默认)/ RTX=2 / HYBRID=3 / MESH_3DGUT=4 / HYBRID_3DGUT=5 |
| `max_sh_degree`              | 球谐阶数上限(默认 3)                                             |
| `lighting_enabled`           | 是否启用灯光(默认 False)                                         |
| `shadows_mode`               | `ShadowsMode`:DISABLED=0 / HARD=1 / SOFT=2                        |
| `sorting_method`             | `SortingMethod`:GPU_SYNC_RADIX=0 … STOCHASTIC_SPLAT=3(3 为随机、需累计) |
| `rtx_max_bounces`            | RTX 最大弹射次数(默认 3)                                         |
| `kernel_degree`              | `KernelDegree`(默认 QUADRATIC=2)                                 |
| `color_format`               | 原始 `VkFormat` 值(默认 `R32G32B32A32_SFLOAT=109`);注意这与 `.cfg` 的 `--colorBufferFormat` 索引不同 |
| `splat_scale`                | 全局 splat 缩放(默认 1.0)                                        |
| `alpha_cull_threshold`       | alpha 剔除阈值(默认 1/255)                                       |

> 枚举都在 `vkgs.constants`(`Pipeline`/`LightType`/`ShadowsMode`/`SortingMethod`/
> `KernelDegree`/`EnvMode`/`DofMode`/`CameraModel`/`Visualize`/`ColorBufferFormat` 等),
> 均为 `IntEnum`,数值**严格**对齐 C++,请勿重新编号。

### 5.4 Camera 与位姿转换

`Camera`(`camera.py`)镜像 C++ `Camera` 结构:

```python
Camera(
    model=CameraModel.PINHOLE,   # PINHOLE=0 / FISHEYE=1
    eye=(1.7,1.5,1.7), ctr=(0,0,0), up=(0,1,0),
    fov=60.0,                    # 垂直 fov,度
    clip=(0.1, 2000.0),
    dof_mode=DofMode.DISABLED,   # DISABLED=0 / FIXED_FOCUS=1 / AUTO_FOCUS=2
    focus_dist=1.3, aperture=0.001,
)
```

提供的转换(签名均已核实):

- `to_view_matrix()` / `from_view_matrix(view, fov=60, focus=1, **kw)` —— 右手 world→camera
  视图矩阵(相机看 -Z)互转。
- `to_camera_to_world()` —— 视图矩阵的逆(camera→world)。
- `from_colmap(position, rotation, fy=None, height=None, **kw)` —— 复刻
  `importCamerasINRIA` 的 RDF→RUB 数学;传 `fy`/`height` 时会**正确算出**垂直 fov
  (in-app 的 `--loadCameraPresets` 反而把 fov 固定成 60)。
- `from_threedgrut_world(camera_to_world, fov=60, focus=1, **kw)` —— 把 3dgrut 世界系
  (PLY-native / COLMAP RDF)的 camera→world 位姿转成 VKGS(RUB)相机。**坐标系换基
  `M = diag(1, -1, -1)`**(绕 X 转 180°)。
- `from_kaolin(kaolin_camera, focus=1, **kw)` —— 从 kaolin `Camera` 转(需 torch/kaolin;
  自动处理水平 fov→垂直 fov、batched 取 index 0)。

INRIA 相机文件读写(模块级函数):

```python
from vkgs import load_inria_cameras, save_inria_cameras
cams = load_inria_cameras("cameras.json")        # 带正确 fov
save_inria_cameras("out.json", cams, width=1920, height=1080)  # 可喂给 --loadCameraPresets
```

### 5.5 Euler 角约定(重要)

所有实例(splat/mesh/light)的 `rotation` 都是 **Euler 角、度**,按 **GLM 四元数约定**
应用:`R = Rz @ Ry @ Rx`(即 X→Y→Z 外旋 / yaw-pitch-roll ZYX)。`camera.py` 提供
`rotation_matrix_from_euler_deg` / `euler_deg_from_rotation_matrix` / `decompose_trs`
用于与旋转矩阵、TRS 矩阵互转。

### 5.6 render_scene 与 RenderResult

```python
render_scene(
    scene, cameras, *,
    size=(1920,1080),
    spp=64,                     # 每台相机 sequenceframes;随机/时序管线需 >= 目标采样数
    buffers=("main",),          # 要读回的缓冲:main/aux1/normal/depth,tonemapping 激活时可加 "ldr"
    out_dir,                    # 必填(keyword-only)
    hdr=False,                  # True 存 .hdr(float32 RGB,渲染进 RGBA32F);否则 .png(RGBA8)
    executable=None, gpu=None,
    keep_files=True,            # False 时立即读入缓存并删掉中间文件(图/.vkgs/.cfg),日志保留
    timeout=1800,
) -> RenderResult
```

参数校验(源码即行为):

- `buffers` 里出现未知名会 `ValueError`;`"ldr"` 但 `tonemapping.is_active` 为假会报错;
  `"comparison"` 直接拒绝(需交互式图像对比模式)。
- `spp < 1` 报错。
- 随机管线(RTX/HYBRID/MESH_3DGUT/HYBRID_3DGUT)、或 `STOCHASTIC_SPLAT` 排序、或任一相机
  开了 DoF,而 `spp < 8` 时,会 `warnings.warn` 提示噪点(需要更多帧收敛)。

`RenderResult`(`facade.py`):

```python
result.image(camera=0, buffer="main")   # 懒加载 + 缓存的 numpy 数组
result.path(camera=0, buffer="main")     # 落盘绝对路径;取不到时抛 KeyError 并列出可用项
result.sequences                          # 解析出的每序列 timer 数据(见 runner.SequenceInfo)
result.log_path                           # render.log 绝对路径
result.out_dir
```

`camera` 是你传给 `render_scene` 的 `cameras` 列表里的**位置**(0-based),不是预设索引;
`buffer` 是 `BUFFER_POSTFIXES` 的键。

---

## 6. CLI command 编排详解(重点)

这一节把「Python 生成了什么、子进程到底怎么被调起来」摊开讲——包括**真实的 `.cfg` 文本**、
**真实的 exe 命令行**、saveImage 的成对时序,以及一节**完全脱离 Python、手写文件跑 exe**
的等价流程。

### 6.1 saveImage 的时序契约(为什么要成对)

这是整个编排里最反直觉、也最关键的一点(`sequence.py` 模块 docstring):

> `--saveImage` 回调在参数被应用的**那一刻**(即序列**开始**时)就执行 `saveBufferToFile`,
> 因此它抓到的是**上一个序列结束时**的帧缓冲。

所以 `RenderScript.capture()` 永远发出**一对**序列:

1. **渲染序列**:`--activateCameraPreset N` + `--sequenceframes spp`(累计收敛,不存图);
2. **保存序列**:`--saveImageBuffer -1` + `--saveImage <stem>.ext` + `--sequenceframes 1`
   (在这一序列开始时抓上一序列的最终帧)。

绝不能把 `saveImage` 放进渲染序列本身。此外整个脚本的**第一个**序列一定是
`load_block`(load/settle):工程在这个序列期间才加载完成,相机预设**只有在它结束后**才可用,
所以 `activateCameraPreset` 绝不能出现在 load_block 里。

另外两条硬规则:

- `--screenshot` 读的是 swapchain,headless 下无效——本模块**拒绝**发出它(`test_no_screenshot_ever`)。
- `--saveImageBuffer` 的**数字索引会随激活的功能漂移**(可 dump 缓冲向量是条件构建的),
  所以统一发 `-1`(全部缓冲),读回时**按文件名后缀挑**,绝不按数字索引(见 [§6.5](#65-输出文件后缀与读回))。

### 6.2 RenderScript:生成 .cfg

`.cfg` 格式:一串 `SEQUENCE "name"` 头,每个头下面跟若干 `--param value` 行(`#` 起注释)。
`RenderScript`(`sequence.py`)的 API:

```python
script = RenderScript(frames=64, averages=32, reset_frames=0)   # 每序列默认值

script.load_block(name="Load scene and settle", *, frames=None, **params)  # 必须是第一个块
script.sequence(name, *, frames=None, averages=None, reset_frames=None, **params)  # 追加一个 SEQUENCE
stem = script.capture(name, camera_preset, out_stem, *, frames=None, buffers=None, hdr=False, **params)
text = script.text()              # 返回 .cfg 文本
abs_cfg = script.write(path)      # 落盘,返回绝对路径
```

参数名书写:可用 python_snake(自动映射到精确 cfg 拼写)或直接用精确拼写。值的编码:
`bool → 1/0`、`int/float → 文本`、`str → 加引号`;`updateData` 这类 flag 参数为 True 时
裸发(无值);`None` 跳过。路径类参数(`saveImage`/`loadCameraPresets`/`inputProject`/
`inputFile`)会被**绝对化**(exe 的 cwd 是它自己所在目录,不是你的 cwd)。

**真实 .cfg 文本**(下面是 `RenderScript` 为 `spp=64`、两台相机、`buffers=["main","depth"]`
生成的**实际输出**,与 `facade.render_scene` 内部一致,已核实):

```cfg
# Generated by vkgs.sequence.RenderScript

SEQUENCE "Load scene and settle"
--sequenceframes 64
--sequenceaverages 32
--sequenceresetframes 0

SEQUENCE "cam0"
--sequenceframes 64
--sequenceaverages 32
--sequenceresetframes 0
--activateCameraPreset 0

SEQUENCE "cam0 save"
--sequenceframes 1
--sequenceaverages 1
--sequenceresetframes 0
--saveImageBuffer -1
--saveImage "/abs/out/cam0.png"

SEQUENCE "cam1"
--sequenceframes 64
--sequenceaverages 32
--sequenceresetframes 0
--activateCameraPreset 1

SEQUENCE "cam1 save"
--sequenceframes 1
--sequenceaverages 1
--sequenceresetframes 0
--saveImageBuffer -1
--saveImage "/abs/out/cam1.png"
```

注意:每个块都**重新声明** `sequenceframes/sequenceaverages/sequenceresetframes`,
不依赖 sequencer 跨块保留状态;第一个块没有 `activateCameraPreset`;`--saveImage` 的路径
永远是绝对路径且带 `.png`/`.hdr` 后缀但**不带**缓冲后缀(缓冲后缀由 exe 追加)。

### 6.3 HeadlessRunner:拼出并跑 exe 命令

`HeadlessRunner`(`runner.py`):

```python
runner = HeadlessRunner(executable=None)   # None 时走 find_executable() 查找
run = runner.run(
    cfg,                        # .cfg 路径(会被绝对化)
    project=None,               # .vkgs 路径 → --inputProject
    input_files=(),             # 可重复的 .ply/.spz/.splat → 多个 --inputFile(与 project 二选一)
    size=(1920,1080),           # → --size W H
    gpu=None,                   # → --forcegpu N(可选)
    timeout=1800,
    extra_args=(),              # 追加到命令末尾
    log_path=None,              # 缺省 <cfg>.log
    expected_outputs=(),        # 跑完检查这些文件是否存在,缺了报错
) -> RunResult
```

**子进程精确命令**(直接来自 `runner.py:179-194` 的构造顺序,已核实):

```
vk_gaussian_splatting \
    --size <W> <H> \
    --benchmark 1 \
    --headless 1 \
    --sequencefile <abs cfg> \
    [--inputProject <abs vkgs>] \
    [--inputFile <abs ply>  (可重复,替代 inputProject)] \
    [--loadDefaultScene 0   (当给了 project 或 input_files 时追加)] \
    [--forcegpu <N>         (gpu 非 None 时)] \
    [<extra_args...>]
```

要点:

- **参数顺序**就是上面这样:`--size` 在最前,然后 `--benchmark 1 --headless 1
  --sequencefile`。(有些文档摘要把顺序写成 headless 在前,以**源码**为准。)
- `--headless 1 --benchmark 1` 下,`headlessFrameCount = UINT32_MAX`,进程**只有在
  sequencer 跑完时才退出**;所以 `.cfg` 必须能终止(否则会 `timeout`)。
- 加了 `--inputProject` 或任意 `--inputFile` 时,**总会**追加 `--loadDefaultScene 0`,
  免得内置演示场景垫在下面。
- 子进程 `cwd` 设为 **exe 所在目录**,`stdout+stderr` 合并写进 `log_path`。

**RunResult**:

```python
run.returncode        # 退出码
run.log_path          # 日志绝对路径
run.log_text          # 日志全文
run.sequences         # List[SequenceInfo]:每个 'ParameterSequence N "name" =' 段 + per-stage timer
run.warnings          # 命中告警正则的日志行
run.duration_s        # 墙钟耗时
run.output_files      # 传入的 expected_outputs(绝对化)
```

**日志解析**(移植自 `benchmark.py`):

- 序列正则:`ParameterSequence (\d+) "([^"]+)" =`
- Timer 正则:`Timer "..."; GPU; avg N; ... CPU; avg M;` → `{"gpu_ms": N/1000, "cpu_ms": M/1000}`
- 告警正则会收集含 `LOGW/LOGE/WARNING/ERROR/Camera preset index/Failed to load/out of
  range/No buffers available` 的行到 `run.warnings`。

**报错条件**(任一命中即抛 `RunError`):

- 退出码非 0;
- `subprocess.TimeoutExpired`(超时,提示检查 `.cfg` 是否能终止);
- 日志里出现**致命标记** `FATAL_LOG_MARKERS = ("Camera preset index", "Failed to load",
  "ERROR")`(注意:相机预设越界、加载失败会让整份 run 失败);
- `expected_outputs` 里有文件没生成。

### 6.4 可执行文件查找顺序

`find_executable(explicit=None)`(`runner.py`)按此顺序找,找不到抛带构建提示的
`FileNotFoundError`:

1. `explicit` 参数(传了但不存在/不可执行 → 直接报错);
2. 环境变量 `$VKGS_BIN`;
3. 仓库 `_bin/{Release,Debug}/vk_gaussian_splatting`(依次 Release、Debug);
4. 仓库 `build*/**` 下的匹配项(取 `mtime` 最新的一个)。

候选文件名:`vk_gaussian_splatting` / `vk_gaussian_splatting.exe` /
`vk_gaussian_splatting_app`。

### 6.5 输出文件后缀与读回

exe 的 `--saveImage <stem>.<ext>` **总会**在 stem 后追加缓冲后缀
(`BUFFER_POSTFIXES`,`constants.py`):

| buffer      | 后缀           | 何时存在                     |
|-------------|----------------|------------------------------|
| `main`      | `_main`        | 总有(HDR 颜色缓冲)          |
| `aux1`      | `_aux1`        | 总有(时序中间缓冲)          |
| `normal`    | `_normal`      | 总有                         |
| `depth`     | `_depth`       | 总有                         |
| `ldr`       | `_ldr`         | 条件:tonemapping 激活        |
| `comparison`| `_comparison`  | 条件:交互式图像对比(headless 不可用)|

也就是说 `--saveImage out/cam0.png` 实际会落盘 `out/cam0_main.png`、`out/cam0_depth.png`……

读回用 `vkgs.images`:

```python
from vkgs import load_image, find_outputs, resolve_outputs

img = load_image("out/cam0_main.png")     # 按扩展名决定 dtype/通道
# find_outputs:glob <stem>_*.* → {buffer_name: abs_path},对条件缓冲鲁棒
files = find_outputs("out/cam0")           # {"main": ".../cam0_main.png", "depth": ...}
# resolve_outputs:预测某 stem/ext 会产生哪些无条件缓冲文件
pred = resolve_outputs("out/cam0", ".png", tonemapping_active=False)
```

`load_image` 的格式契约(`images.py`,对应 `saveBufferToFile`):

- `.png` / `.jpg` → `uint8`,RGBA 原样保留 `(H, W, 4)`(GPU blit 到 `R8G8B8A8_UNORM`)。
- `.hdr` → `float32` `(H, W, 3)`,**丢 alpha**(stb 写 HDR 不带 alpha)。
- `.raw` → `float32` `(H, W, 4)`,带 **16 字节头**(四个 uint32:`width,height,channels=4,
  bytesPerChannel=4`)后跟行主序 float32 RGBA。这是**唯一保留 alpha 的无损浮点格式**,
  且**不需要** imageio。

### 6.6 手动 CLI 编排(不写 Python)

如果你想完全绕开 Python,只用 `vkgs` 帮忙生成/或干脆手写 `.vkgs` + `.cfg`,再自己跑 exe:

**方式 A:用 Python 只生成文件、打印命令(不跑)。** `examples/03` / `examples/04` 在找不到
exe 时正是这么做的——生成文件后打印手动命令:

```python
from vkgs import Scene, Camera
from vkgs.sequence import RenderScript

scene = Scene(); scene.add_splats("garden.ply")
scene.add_camera_preset(Camera(eye=(2.5,1.2,2.5), ctr=(0,0.5,0), fov=60))
vkgs_path = scene.save("out/scene.vkgs")

script = RenderScript(frames=64)
script.load_block()
stem = script.capture("view0", camera_preset=0, out_stem="out/view0")
cfg = script.write("out/render.cfg")
print("wrote", vkgs_path, "and", cfg)
```

然后手动跑(命令与 runner 拼的完全等价):

```bash
vk_gaussian_splatting \
    --size 1280 720 \
    --benchmark 1 --headless 1 \
    --sequencefile "$(pwd)/out/render.cfg" \
    --inputProject "$(pwd)/out/scene.vkgs" \
    --loadDefaultScene 0
# 产物:out/view0_main.png / out/view0_depth.png / ...(带缓冲后缀)
```

**方式 B:完全手写 `.cfg`,不用工程文件,直接喂 `.ply`。** 用 `--inputFile` 替代
`--inputProject`(可重复),相机改用 `--loadCameraPresets` 载入 INRIA cameras.json:

```cfg
# render.cfg —— 手写示例(load/settle + 一对渲染/保存序列)
SEQUENCE "Load scene and settle"
--sequenceframes 64
--sequenceaverages 32
--sequenceresetframes 0
--loadCameraPresets "/abs/cameras.json"

SEQUENCE "shot0"
--sequenceframes 64
--sequenceaverages 32
--sequenceresetframes 0
--activateCameraPreset 0

SEQUENCE "shot0 save"
--sequenceframes 1
--sequenceaverages 1
--sequenceresetframes 0
--saveImageBuffer -1
--saveImage "/abs/out/shot0.png"
```

```bash
vk_gaussian_splatting \
    --size 1920 1080 --benchmark 1 --headless 1 \
    --sequencefile /abs/render.cfg \
    --inputFile /abs/garden.ply \
    --loadDefaultScene 0
```

> 手写时务必遵守 [§6.1](#61-saveimage-的时序契约为什么要成对) 的成对时序:
> 每台相机 =「渲染序列 + 保存序列」,`saveImageBuffer -1` 必须在 `saveImage` **之前**,
> 保存序列 `sequenceframes 1`,第一个块永远是 load/settle。

---

## 7. 视频渲染

`vkgs.video`(`video.py`)把关键帧插值成逐帧相机,全部塞进一个 `.vkgs`,**一次 headless
run** 渲染所有帧,再用 imageio-ffmpeg 组装 mp4。需要 `[video]` extra。

```python
from vkgs import render_video

out = render_video(
    scene,
    keyframes=cams,             # 一组 Camera 关键帧
    out="orbit.mp4",
    *,
    mode="spline",             # "spline"(Catmull-Rom,默认)/ "linear" / "cyclic"(闭环)
    frames_between=30,         # 每段插值帧数
    fps=30, spp=32,            # spp:每视频帧累计帧数
    size=(1280,720),           # 宽高被 16 整除可免 imageio 宏块缩放
    executable=None, workdir=None, keep_frames=False, gpu=None,
) -> str                       # 返回 mp4 绝对路径
```

- `mode` 别名:`smooth`/`catmull_rom`/`path_spline`/`path_smooth` 都归一到 `"spline"`;
  `"cyclic"` 需要 ≥ 3 个关键帧且返回帧数 `len(kf)*frames_between + 1`(末帧回到 kf0);
  开放模式返回 `(len(kf)-1)*frames_between + 1` 帧,精确穿过每个关键帧。
- 插值细节:eye/ctr 走路径;up 用 nlerp(始终单位长);fov/aperture/focus_dist/clip 逐段 lerp;
  离散字段(model/dof_mode)吸附到更近的关键帧。

单独取插值相机:

```python
from vkgs import interpolate_cameras
frames = interpolate_cameras(keyframes, frames_between=30, mode="spline")
```

**轨迹互通**(与 3dgrut `VideoRecorder` 互操作):

```python
from vkgs import save_trajectory, load_trajectory
save_trajectory("traj.npy", keyframes)   # 纯 numpy 结构化 .npy(无 pickle)
cams = load_trajectory("traj.npy")        # 也能读 3dgrut 的 torch.save(kaolin cameras) —— 需 torch+kaolin
```

`save_trajectory` 存的是 kaolin 约定的 camera→world 矩阵、且在 3dgrut 世界系下
(存/读都过 `THREEDGRUT_TO_VKGS`)。`load_trajectory` 兼容三种布局:本包结构化 `.npy`、
纯 `(N,4,4)` c2w 栈、以及 3dgrut 的 `torch.save` 列表。

完整例子见 `python/examples/02_turntable_video.py`(绕场景一圈的 turntable)。

---

## 8. 3dgrut 兼容 shim(EngineVKGS)

`vkgs.compat`(`compat/`)提供一层**镜像 3dgrut playground `Engine3DGRUT` 外部接口**的
shim,便于把 playground 脚本以最小改动移植过来。核心是 `EngineVKGS`:懒状态 +
flush-on-render(每次 `render()` 才把状态序列化成 `.vkgs`/`.cfg` 并跑一个子进程)。

```python
from vkgs.compat import EngineVKGS, OptixPrimitiveTypes

engine = EngineVKGS(
    gs_object="garden.ply",        # 只接受 .ply/.spz/.splat;.pt/.ingp 会报错并提示导出
    mesh_assets_folder="assets",   # 资产按“首字母大写文件名”注册
    *,
    executable=None, out_dir=None, size=(1920,1080),
    return_torch=False,            # True 则把结果 torch.from_numpy 包一层
    pipeline=Pipeline.HYBRID,      # 默认 HYBRID(最接近 playground 的 hybrid tracer)
)
engine.camera_type = "Pinhole"           # "Pinhole" / "Fisheye"
engine.antialiasing_mode = "8x MSAA"     # 映射到累计帧数(4/8/16),Sobol 保持当前 spp
engine.primitives.add_primitive("Sphere", OptixPrimitiveTypes.GLASS)
engine.add_light(light_type=2, position=(0,-2,-1), intensity=30, angular_radius=0.02)

fb = engine.render(camera)   # camera 可为 vkgs Camera / kaolin Camera / 4x4 c2w 矩阵
# fb = {'rgb': (1,H,W,3) float32 in [0,1], 'opacity': (1,H,W,1), 'rgb_buffer': rgb 的别名}
```

- 构造时会像 `Engine3DGRUT` 一样**默认塞一个玻璃 Sphere**;移植脚本里通常先
  `for n in list(engine.primitives.objects): engine.primitives.remove_primitive(n)` 清掉。
- `render()` 内部走 `render_scene(..., hdr=True)`,tonemap/gamma 用 3dgrut **原公式**在
  Python 侧对 HDR 读回做(`compat/tonemap.py`,`"None"/"Reinhard"/"Filmic"` 像素级一致)。
- 多相机一次跑完:`engine.render_many([cam0, cam1, ...])`。

### 8.1 映射表(3dgrut → VKGS)

| 3dgrut playground             | VKGS 兼容做法                                          |
|-------------------------------|--------------------------------------------------------|
| `OptixPrimitiveTypes.MIRROR`  | `materials.mirror()`(metallic=1, roughness=0)         |
| `OptixPrimitiveTypes.GLASS`   | `materials.glass(ior=1.33)`(3dgrut 默认折射率)        |
| `OptixPrimitiveTypes.DIFFUSE` | `materials.diffuse()`                                  |
| `OptixPrimitiveTypes.PBR/NONE`| PBR 保留 glTF 自带材质;NONE 被隐藏                     |
| 世界坐标系                    | `M = diag(1,-1,-1)`(`THREEDGRUT_TO_VKGS`)             |
| 相机                          | `camera_to_vkgs()` / `Camera.from_kaolin` / `from_threedgrut_world` |
| DIRECTIONAL/POINT 灯          | `light_to_vkgs()` → `Scene.add_light`;有灯即开 shadows(soft 当 angular_radius>0) |
| tonemap/gamma                 | `compat/tonemap.py` 在 HDR 读回上做(像素级复刻)       |
| 程序化 `Quad`/`Sphere`        | `geometry.ensure_procedural(...)` 写出 `.obj`          |

### 8.2 差距与 CompatWarning 清单(附 workaround)

以下 playground 能力在 VKGS 侧**不可支持或仅近似**,均以 `CompatWarning`(或
`NotImplementedError`)提示并附带绕法(`compat/convert.py` / `engine.py` / `primitives.py`):

- **`.pt`/`.ingp` checkpoint 不自动转**:构造 `EngineVKGS` 传这类文件会 `ValueError`,
  提示在 3dgrut 环境里先 `model.export_ply('exported.ply')` 再喂 `.ply`。
- **`raygen()` / `RayPack`**(自定义逐像素射线):`NotImplementedError`。绕法:只用
  Pinhole/Fisheye;要奇异投影就渲染 cube faces 后在 Python 重采样。
- **`render_pass` 渐进细化**:批模型下**不跨进程累计**——一次 run 内把所有采样跑完;
  `render_pass` 退化为「dirty 就重渲、否则返回缓存」,`has_progressive_effects_to_render()`
  永远 False。
- **`load_shadow_catcher` / SHADOW_CATCHER 图元**:`CompatWarning + NotImplementedError`。
  绕法:用 DIFFUSE 地面网格,或渲两遍(带/不带遮挡物)在 Python 合成阴影比。
- **AREA 面光源**:无原生等价,近似成**发光四边形网格**(用 tangent_u/tangent_v 定尺寸),
  只在路径追踪管线下发光且更噪;替代方案用带 radius 的 POINT 光。
- **`use_optix_denoiser`**:忽略,VKGS 无 OptiX denoiser(只有 `USE_DLSS` 构建 + RTX 硬件
  的 DLSS-RR);用提高 spp 换低噪。
- **`shadow_min` / `shadow_spp`**:属于 3dgrut shadow-catcher 路径,无等价;近似用
  `renderer_overrides['particle_shadow_transmittance_threshold' / 'shadows_mode']`,
  软阴影靠累计帧收敛。
- **`scene_mog` 张量访问**:无(子进程模型,拿不到进程内张量);`scene_mog` 恒为 `None`。
- **MSAA/Sobol 抗锯齿**:只映射到累计帧数,VKGS 用自己的时序 jitter,per-sample 图案与
  3dgrut 不一致(收敛后可比)。
- **`envmap_offset`**:仅近似(方位角→IBL Y 旋转、极角→X 旋转,对偏轴内容并非同一球面平移)。
- **`'White'` 环境**:不支持常量色环境,退化为无环境;绕法是自制一张全白 `.hdr` 选它。
- **opacity 通道**:`.hdr` 读回不带 alpha,故 `opacity` 目前恒为全 1;真实覆盖需 `.raw`
  RGBA32F 路径,且 RTX 管线还需 C++ 侧 alpha 补丁(见 [§10](#10-常见问题与边界))。

完整移植示例见 `python/examples/04_playground_port.py`(逐 cell 对照 3dgrut headless notebook,
用 DIFF-NOTE 标注每处 API 差异)。

---

## 9. 测试与验证

单测在 `python/tests`,用 `dev` extra 的 `pytest`。gpu 标记的用例需要**真实 exe**。

```bash
# CPU-only 单测(不碰 GPU / 不需要 exe)—— 已核实 117 passed
uv run pytest python/tests -m "not gpu"

# gpu 冒烟(需要已构建 exe + 数据集;通过 $VKGS_BIN 或 _bin/Release 提供)
uv run pytest python/tests -m gpu
```

`uv run` 会自动在已安装 `[dev]` extra 的 `.venv` 里执行(没建就先 `uv venv && uv pip install -e "python/[dev]"`)。
`-m gpu` 标记定义在 `pyproject.toml`(`gpu: tests that launch the real renderer executable`)。
用仓库自带的开发 venv 也行:

```bash
python/.venv/bin/python -m pytest python/tests -m "not gpu"
```

跑 examples(都需要 exe + 一个 `.ply`):

```bash
python python/examples/01_build_and_render.py --ply garden.ply --out out
python python/examples/02_turntable_video.py  --ply garden.ply --out orbit.mp4
python python/examples/03_from_3dgrut_ply.py  --ply garden.ply --out out
python python/examples/04_playground_port.py  --ply garden.ply --out out --size 512 512
```

> `03`/`04` 在找不到 exe 时**不会崩**:它们会写出 `.vkgs`(+`.cfg`)并打印等价的手动命令,
> 方便在有 GPU 的机器上接着跑。

---

## 10. 常见问题与边界

- **`.pt`/`.ingp` 桥**:Python 层**不**加载 PyTorch/3dgrut checkpoint。先在 3dgrut 环境里
  导出 `.ply`(`MixtureOfGaussians.export_ply(...)`,见 `examples/03` 的
  `export_ply_from_3dgrut`),再喂给 `add_splats` / `EngineVKGS`。

- **贴图**:材质是**纯因子**;贴图只能靠 **mesh 文件自带**(glTF 的 PBR 贴图)。splat
  **完全无贴图**。`geometry` 写出的程序化 `.obj`(`write_quad_obj`/`write_sphere_obj`/
  `ensure_procedural`)不含材质库,加载时被套一个默认灰材质,请用 `add_mesh(..., materials=[...])`
  覆盖。

- **逐相机分辨率**:一次 headless run **只有一个 `--size`**。所有相机同尺寸渲染。
  `EngineVKGS.render_many` 里若各相机(如 kaolin)携带不同分辨率,会 `CompatWarning`
  并统一用第一个。需要不同分辨率就**分多次** `render_scene`。

- **opacity / alpha(RTX alpha 补丁)**:
  - `.png`/`.raw` 带 alpha,`.hdr` 不带(读回 `(H,W,3)`)。
  - **光栅**管线的 alpha 已经是有意义的覆盖(`1 - transmittance`)。
  - **RTX** 管线历史上写 alpha=1.0;`feat/py-binding` 相关改动让 RTX 管线(2/3/5)也把
    累计覆盖写进 main 缓冲的 alpha,使 `--saveImage` 的 PNG/RAW 带可用 opacity(见
    `docs/python-scripting.md`「Benchmark-script extras」)。要拿真实覆盖建议走
    **`.raw` RGBA32F** 读回。

- **找不到 exe 的报错**:`find_executable()` 抛 `FileNotFoundError`,消息里带构建提示
  (`cmake -S . -B build ...` + `_bin/Release/`)和 `$VKGS_BIN` 提示。修法:按 [§2](#2-构建-vkgs-可执行文件)
  构建,或 `export VKGS_BIN=/path/to/exe`,或 `executable=` 显式传。

- **相机预设越界 / 加载失败**:`.cfg` 里 `--activateCameraPreset` 的值超过工程里的相机数,
  或工程/资产加载失败,会在日志里留下 `Camera preset index ... out of range` /
  `Failed to load`——这两个是 `FATAL_LOG_MARKERS`,`HeadlessRunner.run` 会据此抛 `RunError`。
  用 `add_camera_preset` 返回的索引就不会越界。

- **子进程卡住(timeout)**:headless+benchmark 只在 sequencer 跑完才退出。手写 `.cfg`
  时若忘了终止(比如序列无限循环)会命中 `timeout` 并抛 `RunError`。用 `RenderScript`
  生成的脚本天然有限、会正常结束。

- **`.cfg` 参数命名**:优先用 `RenderScript`;确需手传冷门参数时,注意 snake→camelCase
  的特例(如 `sorting_method → sortStrategy`、`shadows_mode → shadowMode`、
  `use_aabbs → useAABBs`、`sh_format → shformat`),`sequence._SNAKE_TO_CFG` 里列全了。

---

### 相关文件索引(仓库内)

- Python 包:`python/vkgs/`(`project.py` / `camera.py` / `sequence.py` / `runner.py` /
  `facade.py` / `images.py` / `materials.py` / `geometry.py` / `video.py` / `constants.py`)
- 兼容 shim:`python/vkgs/compat/`(`engine.py` / `convert.py` / `primitives.py` / `tonemap.py`)
- 示例:`python/examples/01–04*.py`
- 单测:`python/tests/`
- 打包:`python/pyproject.toml`、`python/README.md`
- 另一篇较概览的脚本文档:`docs/python-scripting.md`
