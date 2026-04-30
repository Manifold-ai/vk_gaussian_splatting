# Monitoring Panels  

The **Memory Statistics** panel reports RAM (host) and VRAM (device) usage organized in an expandable tree with the following categories:

*   **Model data** – Memory consumed by splat set attributes (centers, scales, rotations, covariances, SH coefficients, index tables, descriptor buffer).
*   **Rasterization** – Buffers used for sorting (distances, indices, indirect parameters) and GPU radix sort internals.
*   **Ray tracing** – Acceleration structure memory (TLAS, BLAS, geometry buffers, scratch buffers, instance buffers).
*   **Renderer commons** – Shared rendering resources (UBO, quad buffers, G-Buffers for color and depth).

Each category shows three columns: **Host used**, **Device used**, and **Device allocated**. A **Grand Total** row sums all categories. Expand/Collapse all buttons are provided for quick navigation.

The **Profiler** panel reports the **GPU** and **CPU** time spent on different stages of the rendering process. The set of timers varies depending on the selected pipeline and sorting method. Results can be viewed as a table, pie chart, or line chart.

The **Rendering Statistics** panel provides additional information organized in three groups:

*   **Scene** – Number of splat sets and instances, total particle count.
*   **Rasterization** – Number of rasterized splats and mesh shader work groups (when applicable).
*   **Ray Tracing** – Acceleration structure mode, TLAS/BLAS counts and entry counts.

!!! note
    To properly assess the performance of the pipelines, one should **deactivate vertical synchronization** (V-Sync) either from the **View > V-Sync** menu, the **Renderer properties panel**, or the toolbar. Otherwise, the system does not run at optimal performance, and the reported timings in the window title bar and Profiler panel are generally higher (and fps lower) than what is possible to achieve. The V-Sync option is enabled by default for energy saving.
