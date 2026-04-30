# Cameras

The **Camera** asset allows you to visualize and set the current camera settings and interaction modes. While navigating to an interesting location, it is also possible to **store** the current camera setting as a preset. Later on, this preset can be reloaded using its associated **load** button.

It is also possible to **import** camera presets (cameras.json files) as defined in the [INRIA model bundle]({{ source_base }}/README.md#Datasets){:target="_blank"}. This is very useful for running benchmarks.

Pressing the `space bar` will **load** the next camera preset, making it convenient to quickly activate the presets one after the other.

The camera properties also expose:

*   **Camera type** – Switches between **Pinhole** (standard perspective projection) and **Fisheye** (wide-angle lens simulation).
*   **Depth of Field** – Selects the DoF mode: **Disabled**, **Fixed focus** (manual focus distance), or **Auto focus** (focus at the point under the cursor). When enabled, an **Aperture** control adjusts the strength of the bokeh blur. Depth of field requires temporal sampling to converge.

Fisheye and Depth of Field are not available with the 3DGS pure rasterization pipelines.
