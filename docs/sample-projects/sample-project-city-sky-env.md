# Large City — Sky Environment Lighting

<div><img-comparison-slider>
  <figure slot="first" class="before">
    <img src="../../images/city-sky-environment-light-off.jpg" />
    <figcaption>Light Off</figcaption>
  </figure>
  <figure slot="second" class="after">
    <img src="../../images/city-sky-environment.jpg" />
    <figcaption>Sky Environment</figcaption>
  </figure>
</img-comparison-slider></div>

**File:** `3dgs_large_city_sky_environment_lighting.vkgs`

This project features a large outdoor city scene captured from approximately 20,000 photographs, producing over 103 million Gaussian splats. It demonstrates:

- **Rasterization and Raytracing** of a very large splat set
- **Sky environment lighting** with physical sun/sky model
- **Ray Traced Shadow** on a large-scale urban scene

!!! note
    If RTX is not supported, the system will fall back to 3DGS Raster. Shadows, AO and advanced mesh materials will not show up in this mode.

## Interacting with the scene

- Switch cameras by pressing the space bar
- Toggle the lighting on and off to see its impact
- Switch DLSS on and experiment with the different upscaling modes (MAX, OPTIMAL, MIN)
- Play with the visualization modes
- Explore the scene and the UI!

## Assets

| Type | Source |
|------|--------|
| Splat model | 20K-Photo-103Mspats-4x2KM (teleportour.com, by Andrii Shramko) |
