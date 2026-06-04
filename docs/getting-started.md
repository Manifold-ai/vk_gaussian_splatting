# Getting Started

## Runtime Requirements

- NVIDIA GPU
- 64-bit Windows or Linux

## Downloading Pre-builds

Pre-built binaries for Windows 64-bit and Linux 64-bit are available on the [GitHub Releases](https://github.com/nvpro-samples/vk_gaussian_splatting/releases){:target="_blank"} page.

## Running

Double-click the executable file from the archive or run from the command line.

``` sh
# Run with optional path to ply file (can be opened later on through UI)
./vk_gaussian_splatting.exe [path_to_ply]
```

In your operating system you can also associate `.ply`, `.obj`, `.gltf` and `.vkgs` extensions to be opened by the software, for instance by right-clicking a file, selecting "Open with", and then choosing "Always".

## Opening 3DGS PLY and SPZ Files

The application supports:

* PLY files in the format introduced by INRIA [[Kerbl2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)]. 
* SPZ files as defined by [nianticlabs](https://github.com/nianticlabs/spz). 

PLY and SPZ files can be opened using any of the following methods:

* **Command Line** – Provide the file path as last argument when launching the application.
* **File Menu** – Use "File > Open" to browse and load a PLY or SPZ file.
* **Drag and Drop** – Simply drag and drop the PLY or SPZ file into the viewport.

### Compatibility

* [Jawset Postshot](https://www.jawset.com/) and [3DGRUT](https://github.com/nv-tlabs/3dgrut) output ply files are compatible with the INRIA format and can be opened directly.
* Other reconstruction software's ply outputs such as [NerfStudio](https://docs.nerf.studio/nerfology/methods/splat.html), [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) should also work.

### Downloading models to play with

!!! important
    Visualization of 3D Gaussian models is most effective when using the same rendering algorithm and settings used during reconstruction. For example, models generated using Postshot 3DGS with anti-aliasing enabled will be best visualized using one of the available 3DGS pipelines (Mesh or Vertex) and by enabling "Mip Splatting Anti-Aliasing" in the rendering > rasterization parameters. Since no generic format with proper metadata exists, users must manually configure the rendering settings. Note that original 3DGS models from INRIA do not include anti-aliasing, which was introduced later [Yu2023].

* A simple lightweight bouquet of flowers model provided by NVIDIA ([download](http://developer.download.nvidia.com/ProGraphics/nvpro-samples/flowers_1.zip)).
* 3DGS, 3DGRT and 3DGUT models can be found in the [datasets page](datasets.md).
* SPZ file import can be tested using the sample [models](https://github.com/nianticlabs/spz/tree/main/samples) provided by nianticlabs.

### Opening Sample Projects

The repository ships with sample projects that showcase complete scene setups with advanced lighting. See the [Sample Projects](sample-projects/index.md) section to get started.

## Compilation Requirements

- [Vulkan 1.4 SDK](https://vulkan.lunarg.com/sdk/home)
- [CMake 3.22 or higher](https://cmake.org/download/)
- Compiler supporting basic C++20 features
  - MSVC 2019 on Windows
  - GCC 10.5 or Clang on Linux
- Additional Libraries on Linux
    - `sudo apt install libx11-dev libxcb1-dev libxcb-keysyms1-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev libxxf86vm-dev libtbb-dev`
- [CUDA v12.6](https://developer.nvidia.com/cuda-downloads) is **optional** and can be used to activate **NVML GPU monitoring** in the sample. 
- NVIDIA DesignWorks [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2) will be automatically downloaded if not found next to the sample directory.

## Building from Source

``` sh
# Clone the repository (use --recursive instead for git version < 2.13)
git clone --recurse-submodules https://github.com/nvpro-samples/vk_gaussian_splatting
cd vk_gaussian_splatting

# Configure 
cmake -S . -B build

# Alternatively, configure as follows to disable the default "bouquet of flowers"  
# scene download by CMake and prevent auto-loading in the sample
cmake -S . -B build -DDISABLE_DEFAULT_SCENE=ON

# Build
cmake --build build --config Release

# Run with optional path to ply file (can be opened later on through UI)
./_bin/Release/vk_gaussian_splatting.exe [path_to_ply]
```

## Documentation Preview

Install the required Python packages:

``` sh
pip install mkdocs-material mkdocs-macros-plugin
```

Then preview the documentation locally:

``` sh
mkdocs serve -a 127.0.0.1:8080 --livereload
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser. Changes to documentation files will automatically reload in the browser.
