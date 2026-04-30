# Radiance Fields / Splat Set

Once a .ply file is opened, a **Splat set** entry appears in the **Assets** panel under the **Radiance Fields** tree. Multiple splat sets can be loaded simultaneously. Each splat set can also be duplicated to create additional instances that share the same underlying data but have independent transforms and materials.

The **Properties** panel shows different content depending on what is selected in the **Radiance Fields** tree:

- Selecting the **Radiance Fields** root node displays **shared properties** that apply to all splat sets (data format in VRAM and RTX acceleration structures).
- Selecting an individual **Splat set** entry displays **per-instance properties** (info, transform, material, and storage mode).

## Shared Properties (Radiance Fields selected)

### Splat Set Format in VRAM

The **Splat Set Format in VRAM** group controls the data format used for all splat sets in VRAM.

- **SH format** – Selects between **Float32**, **Float16**, and **Uint8** for SH coefficient storage, balancing precision and memory usage.
    *  By lowering the size of the SH data to 8 bits (**Uint8**), one can achieve very high performance gains when using the **rasterization** and **hybrid** pipelines, with perceptually invisible quality loss. The pipeline, which is already running at very high performance when using **Float32**, is bounded by the data fetch from memory. Hence, reducing the size of the coefficients, which make up a large part of the payload, massively reduces the data throughput and reduces the frame rendering time.
- **RGBA format** – Selects between **Float 32**, **Float 16**, and **Uint8** for RGBA color and alpha storage, balancing precision and memory usage (16, 8, or 4 bytes per splat respectively).

### RTX Acceleration Structure

The **RTX Acceleration Structure** group configures the acceleration structure used by the ray tracing rendering pipeline. 

Those options have strong impact on performance and memory consumption. They are documented and devised more in detail in the [VK3DGRT: 3D Gaussian Ray Tracing (3DGRT) [Moënne-Loccoz2024] using Vulkan RTX](../deep-dives/ray_tracing_3d_gaussians.md) page.

## Per-Instance Properties (Splat set selected)

Selecting an individual splat set instance in the tree shows the following property groups:

- **Splat Set Info** – Displays the total number of splats, SH degree, source file path, and how many instances share this splat set data.
- **Model Transform** – Per-instance translation, rotation, and scale. This allows placing multiple instances of the same splat set at different locations in the scene.
- **Material** – Per-instance material properties (ambient, diffuse, specular, emission, shininess) used when lighting is enabled. By default radiance fields are 100% emissive, meaning they appear as baked-in colors unaffected by lights. To see the impact of lighting, reduce the emission value and raise the diffuse value so that the splat set responds to the scene's light sources.
- **Splat Set Storage in VRAM** – Per-splat-set **Storage** mode, selecting between **Data Buffers** and **Textures** for storing model attributes including position, color and opacity, covariance matrix, and SH coefficients (for degrees higher than 0). Changes to this setting impact all instances sharing the same splat set.

This **Storage** option impacts memory access patterns and performance, allowing comparisons between different storage strategies. In both modes, splat attributes are stored linearly in memory in the order they are loaded from disk.

* **Data Buffer Mode** – Uses a separate buffer for each attribute type.
    * This layout improves memory lookups during shader execution, as threads access attributes in sequential stages (e.g., first positions, then colors, etc.).
    * Buffers are allocated and initialized by the `initDataBuffers` method (see [splat_set_vk.cpp]({{ source_base }}/src/splat_set_vk.cpp){:target="_blank"}).
* **Texture Mode** – Uses a separate texture map for each attribute type.
    * All textures are 4092 pixels wide, with the height determined as a power of two based on the attribute's memory footprint.
    * Linear storage in textures is suboptimal due to square-based cache for texel fetches, but data locality cannot be easily optimized as sorting is view-dependent.
    * Future work could explore organizing data as in [Morgenstern2024] to leverage texture compression.
    * Textures are allocated and initialized by the `initDataTextures` method (see [splat_set_vk.cpp]({{ source_base }}/src/splat_set_vk.cpp){:target="_blank"}).
