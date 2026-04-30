# Release Notes

## Version 2026.1.7

- **Pre-built binaries** available on GitHub Releases page
- **New online [documentation](https://nvpro-samples.github.io/vk_gaussian_splatting/) website** — This site.
- **Simplified root github repo README.md**
- **Rendering pipeline selector** added to the menu bar
- **Navigation mode icons** in toolbar with improved camera controls
- **Fix profiler reporting** during camera drag in raster/hybrid pipelines
- **Fix 32-bit addressing overflow** in sorting buffers by converting to LargeBuffer (supports larger models)

## Version 2026.1

- Multi-instance splat set architecture with global index tables and unified sorting
- Centralized bindless asset management with a root bindless scene assets buffer
- Multi-light lighting system (point / spot / directional, ray traced hard / soft shadows)
- Front-to-back rasterization with depth consolidation for lighting
- Deferred shading for rasterized Gaussian splats
- GPU-built particle acceleration structures with multi-TLAS / multi-BLAS chunking and VRAM budget pre-checking
- Ray tracing now supports very large models and updates faster
- Stochastic splat sorting and Monte Carlo trace strategy for enhanced interactivity
- DLSS Ray Reconstruction for anti-aliasing, upscaling and denoising
- Expanded visualization modes in ray tracing and hybrid pipelines (normals, depth, DLSS buffers, splat ID)
- UI / UX enhancements (summary overlay, shader feedback and ray hit profile, editing mode)
- 3D transform and infinite grid visual helpers
- Runtime image comparison tool with MSE / PSNR / FLIP metrics
- Added **Winter house** and **Large city** models from <a href="https://teleportour.com/" target="_blank">teleportour.com</a> to [datasets page](datasets.md#models-from-teleportourcom). Thanks to <a href="https://www.linkedin.com/in/andrii-shramko" target="_blank">Andrii Shramko</a>.

## Older Versions

- [2025/10] Migration from GLSL to SLANG.
- [2025/09] Added Unscented Transform (3DGUT) and hybrid rendering (3DGUT/3DGRT) pipelines, with support for fisheye and depth of field.
- [2025/09] Added import of .spz file format from [nianticlabs](https://github.com/nianticlabs/spz).
- [2025/08] Added depth of field to raytracing (3DGRT) pipeline.
- [2025/08] Added raytracing (3DGRT) and hybrid rendering (3DGS/3DGRT) pipelines + 3DGRT dataset.
- [2025/08] Added compositing with meshes and project save/load functionality.
- [2025/06] Ported to new NVIDIA DesignWorks [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2).
- [2025/03] Added new models from [3ds-scan.de](https://3ds-scan.de/). Thanks to Christian Rochner.
- [2025/03] First release with 3DGS rasterization.
