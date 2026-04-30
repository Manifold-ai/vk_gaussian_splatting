# Workflow

The visualization workflow follows these main steps:

1. Loading the 3DGS Model into RAM.
2. Data Transformation & Upload to VRAM
    - The splat attributes (positions, opacity, Spherical Harmonic (SH) coefficients from degree 0 to 3, and rotation scale) are transformed if necessary and uploaded into VRAM. 
    - Additional acceleration structures are generated in VRAM for Ray tracing or Hybrid rendering, see details in the ray tracing page.
    - The data storage, format and acceleration structure related parameters can be updated during the visualization, the data in VRAM and the pipelines are then regenerated on the flight.
    - Changing those parameters may have strong impact on rendering performances, this topic is devised in the rendering pipeline pages.
3. Rendering
    - The rendering pipeline/method can be changed dynamically during the visualization and allows to easily compare the performance of the different approaches.
    - Several monitors can be visualized at any moment to compare the performances and resource consumption of the different approaches.
