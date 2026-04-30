# Mesh Models

The **Mesh Models** entry in the asset manager allows importing 3D meshes from .obj files along with their material definitions defined in accompanying .mtl files. Note that texture maps are not yet supported. To import a mesh, use the **Import** button and select an .obj file.

Some interesting .obj files are automatically downloaded by the CMake and can be found in the **_downloaded_resources** folder of the repository.

Once imported, a mesh can be selected in the asset tree which shows its transform and materials properties in the **Properties** panel. 

The Material Properties panel allows selecting three shading **Models**: 

1. **No indirect**: Computes the shading without taking into account indirect contributions. The mesh will not present any reflection or refraction. This mode works for all the rendering pipelines.
2. **Reflective**: Activates the tracing of secondary rays to compute reflections of the environment (splat set and other meshes) onto the mesh. The proportion of reflection is controlled by the **specular** field.
3. **Refractive**: Activates the tracing of secondary rays to compute refractions, showing the environment (splat set and other meshes) through the mesh. Using this mode, one shall set **transmittance** to a value greater than 0.0 to render transparency. The **IOR** field is used to change the index of refraction (use 1.5 for glass material).

**Reflective** and **Refractive** modes have no effect when using rasterization pipelines. 
