# Renderer Pipelines and Properties

When starting the application the Renderer is selected in the **Asset panel** and the properties of the selected pipeline appear below in the **Properties panel**. The **down arrow** at the right of the pipeline name in the **Assets > Renderer** section allows to switch the active rendering pipeline. 

The **Renderer** properties panel is organized in several groups. The first groups contain settings common to all pipelines, followed by pipeline-specific tabs documented in the pages linked in the table below.

*   **Pipeline** – Selects the active rendering pipeline (see table below).

## Global Settings

These settings apply to all rendering pipelines:

*   **V-Sync** – Toggles vertical synchronization on or off.
*   **Default settings** – Resets all renderer settings to their defaults.

*   **Color Format** – Selects the color buffer format, trading precision for memory. Available formats are **R8G8B8A8 UNORM** (32-bit, lowest memory), **R16G16B16A16 SFLOAT** (64-bit, default), and **R32G32B32A32 SFLOAT** (128-bit, highest precision). Higher precision improves temporal accumulation quality.

* **Visualize** selector is available when a ray tracing pipeline is active (pure RTX or hybrid). It switches the viewport output between the final render and various debug views:
    *   **Final render** – Standard composited output.
    *   **Clock cycles** – GPU clock cycle heatmap with adjustable min/max range and shift.
    *   **Ray Hit Count** – Number of ray-particle intersections per pixel with adjustable min/max range and shift.
    *   **Depth (iso thres)** / **Depth (Closest hit)** / **Depth (for DLSS)** – Depth buffer visualizations with adjustable min/max range and shift.
    *   **Normal (Integrated)** / **Normal (closest hit)** / **Normal (For DLSS)** – Normal vector visualizations.
    *   **Splat ID (Harlequin)** – Per-splat identification with randomized colors.
    *   **DLSS** guide views (Input, Albedo, Specular, Normal, Motion, Depth) – Available only when DLSS is enabled, these display the G-buffer channels fed to DLSS.

*   **Wireframe** – Shows particle bounds in wireframe.
*   **Alpha culling threshold** – Discards splats with opacity below the threshold to skip low-contribution particles.
*   **Maximum SH degree** – Sets the highest Spherical Harmonics degree (0–3) used for view-dependent color.
*   **Show SH deg > 0 only** – Removes the base color from SH degree 0, applying only higher-degree SH over neutral gray. Useful for visualizing their contribution.
*   **Disable opacity gaussian** – Makes the full range of each Gaussian visible by disabling the alpha falloff, useful for analyzing splat distribution.
*   **Normal vectors** – Selects the method for computing normal vectors: **Max density plane** (tangent plane approximation, fast) or **Kernel ellipsoid** (ray-ellipsoid intersection, more accurate).
*   **Thin particle threshold** – Scale below which a particle axis is considered degenerate; such particles are treated as flat disks for normal computation.
*   **Lighting mode** / **Shadows mode** – Controls lighting and shadow evaluation on the splat set. These features are described in detail in the [Lighting and Shadows](../deep-dives/lighting_and_shadows.md) page.
*   **Temporal sampling** – Controls temporal accumulation of frames (**Automatic**, **Force enabled**, **Force disabled**). When enabled, results are accumulated over a configurable number of frames, improving quality for effects like depth of field and stochastic ray tracing.

## Pipeline-Specific Properties

Each pipeline exposes additional properties in dedicated tabs (Rasterization or Ray tracing). The description of the respective properties and the implementation details of the different pipelines is devised in the following sections.

| Pipeline name                | Implementation details |
|--|--|
| **Raster vertex shader 3DGS**  | [VK3DGSR: 3D Gaussian Splatting (3DGS) [Kerbl2023] using Vulkan Rasterization](../deep-dives/rasterization_of_3d_gaussian_splatting.md) |
| **Raster mesh shader 3DGS**| [VK3DGSR: 3D Gaussian Splatting (3DGS) [Kerbl2023] using Vulkan Rasterization](../deep-dives/rasterization_of_3d_gaussian_splatting.md) |
| **Raster mesh shader 3DGUT**| [VK3DGUT: 3D Gaussian Unscented Transform (3DGUT) [Wu2024] Using Vulkan Rasterization](../deep-dives/rasterization_of_3dgut.md) |
| **Ray tracing 3DGRT**        | [VK3DGRT: 3D Gaussian Ray Tracing (3DGRT) [Moënne-Loccoz2024] using Vulkan RTX](../deep-dives/ray_tracing_3d_gaussians.md)    |
| **Hybrid 3DGS+3DGRT**        | [VK3DGHR: 3D Gaussian Hybrid Rendering Using Vulkan RTX and Rasterization](../deep-dives/hybrid_rendering_3d_gaussians.md)                   |
| **Hybrid 3DGUT+3DGRT**        | [VK3DGHR: 3D Gaussian Hybrid Rendering Using Vulkan RTX and Rasterization](../deep-dives/hybrid_rendering_3d_gaussians.md)                   |
