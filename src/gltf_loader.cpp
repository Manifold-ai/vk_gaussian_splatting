/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "gltf_loader.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

namespace {

// Parse KHR_texture_transform from a tinygltf TextureInfo's extensions.
// Returns the UV transform matrix and (via out params) the texCoord override.
template <typename T>
void parseTextureTransform(const T& texInfo, glm::mat3x2& uvTransform, int& texCoord)
{
  uvTransform = glm::mat3x2(1.0f);  // identity
  texCoord    = texInfo.texCoord;   // base texCoord (0 by default)

  auto extIt = texInfo.extensions.find("KHR_texture_transform");
  if(extIt == texInfo.extensions.end())
    return;

  const auto& ext = extIt->second;

  glm::vec2 offset(0.0f);
  float     rotation = 0.0f;
  glm::vec2 scale(1.0f);

  if(ext.Has("offset"))
  {
    const auto& arr = ext.Get("offset");
    if(arr.ArrayLen() >= 2)
      offset = glm::vec2(static_cast<float>(arr.Get(0).GetNumberAsDouble()), static_cast<float>(arr.Get(1).GetNumberAsDouble()));
  }
  if(ext.Has("rotation"))
    rotation = static_cast<float>(ext.Get("rotation").GetNumberAsDouble());
  if(ext.Has("scale"))
  {
    const auto& arr = ext.Get("scale");
    if(arr.ArrayLen() >= 2)
      scale = glm::vec2(static_cast<float>(arr.Get(0).GetNumberAsDouble()), static_cast<float>(arr.Get(1).GetNumberAsDouble()));
  }
  if(ext.Has("texCoord"))
    texCoord = ext.Get("texCoord").GetNumberAsInt();

  float c = std::cos(rotation), s = std::sin(rotation);
  // Column-major mat3x2: each column is a vec2
  // col0 = (sx*c, sx*s), col1 = (-sy*s, sy*c), col2 = (tx, ty)
  uvTransform = glm::mat3x2(scale.x * c, scale.x * s, -scale.y * s, scale.y * c, offset.x, offset.y);
}

// Parse KHR_texture_transform from a tinygltf::Value textureInfo (extension textures).
// Equivalent to parseTextureTransform() above but works with the Value API used for
// extension texture info objects that tinygltf stores as generic JSON.
void parseTextureTransformFromValue(const tinygltf::Value& texInfoVal, glm::mat3x2& uvTransform, int& texCoord)
{
  uvTransform = glm::mat3x2(1.0f);
  texCoord    = 0;

  if(texInfoVal.Has("texCoord"))
    texCoord = texInfoVal.Get("texCoord").GetNumberAsInt();

  if(!texInfoVal.Has("extensions"))
    return;
  const auto& exts = texInfoVal.Get("extensions");
  if(!exts.Has("KHR_texture_transform"))
    return;
  const auto& ext = exts.Get("KHR_texture_transform");

  glm::vec2 offset(0.0f);
  float     rotation = 0.0f;
  glm::vec2 scale(1.0f);

  if(ext.Has("offset"))
  {
    const auto& arr = ext.Get("offset");
    if(arr.ArrayLen() >= 2)
      offset = glm::vec2(static_cast<float>(arr.Get(0).GetNumberAsDouble()), static_cast<float>(arr.Get(1).GetNumberAsDouble()));
  }
  if(ext.Has("rotation"))
    rotation = static_cast<float>(ext.Get("rotation").GetNumberAsDouble());
  if(ext.Has("scale"))
  {
    const auto& arr = ext.Get("scale");
    if(arr.ArrayLen() >= 2)
      scale = glm::vec2(static_cast<float>(arr.Get(0).GetNumberAsDouble()), static_cast<float>(arr.Get(1).GetNumberAsDouble()));
  }
  if(ext.Has("texCoord"))
    texCoord = ext.Get("texCoord").GetNumberAsInt();

  float c = std::cos(rotation), s = std::sin(rotation);
  uvTransform = glm::mat3x2(scale.x * c, scale.x * s, -scale.y * s, scale.y * c, offset.x, offset.y);
}

template <typename T>
const T* getAccessorData(const tinygltf::Model& model, int accessorIdx)
{
  const auto& accessor   = model.accessors[accessorIdx];
  const auto& bufferView = model.bufferViews[accessor.bufferView];
  const auto& buffer     = model.buffers[bufferView.buffer];
  return reinterpret_cast<const T*>(buffer.data.data() + bufferView.byteOffset + accessor.byteOffset);
}

int accessorCount(const tinygltf::Model& model, int accessorIdx)
{
  return static_cast<int>(model.accessors[accessorIdx].count);
}

void generateFlatNormals(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, size_t indexStart)
{
  for(size_t i = indexStart; i < indices.size(); i += 3)
  {
    glm::vec3 v0                 = vertices[indices[i + 0]].pos;
    glm::vec3 v1                 = vertices[indices[i + 1]].pos;
    glm::vec3 v2                 = vertices[indices[i + 2]].pos;
    glm::vec3 nrm                = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    vertices[indices[i + 0]].nrm = nrm;
    vertices[indices[i + 1]].nrm = nrm;
    vertices[indices[i + 2]].nrm = nrm;
  }
}

// Compute the local transform for a glTF node.
// glTF nodes specify transforms as either a 4x4 matrix or separate TRS components.
glm::mat4 getNodeTransform(const tinygltf::Node& node)
{
  if(node.matrix.size() == 16)
  {
    // Column-major matrix stored as doubles
    glm::mat4 m;
    for(int i = 0; i < 16; i++)
      m[i / 4][i % 4] = static_cast<float>(node.matrix[i]);
    return m;
  }

  glm::mat4 T(1.0f), R(1.0f), S(1.0f);
  if(node.translation.size() == 3)
    T = glm::translate(glm::mat4(1.0f), glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
  if(node.rotation.size() == 4)
    R = glm::mat4_cast(glm::quat(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                                 static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2])));
  if(node.scale.size() == 3)
    S = glm::scale(glm::mat4(1.0f), glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
  return T * R * S;
}

}  // namespace

bool GltfLoader::load(const std::filesystem::path& filename, Mesh& mesh)
{
  mesh = {};

  tinygltf::Model    model;
  tinygltf::TinyGLTF loader;
  std::string        warn, error;

  // Capture raw image bytes so MeshManagerVk can load them later.
  // This handles both external file references (image.uri set) and
  // GLB-embedded images (raw bytes stored in image.image).
  loader.SetImageLoader(
      [](tinygltf::Image* image, const int, std::string*, std::string*, int, int, const unsigned char* bytes, int size, void*) {
        if(bytes && size > 0)
          image->image.assign(bytes, bytes + size);
        return true;
      },
      nullptr);

  const std::string filenameStr = filename.string();
  const std::string ext         = filename.extension().string();
  bool              result      = false;

  if(ext == ".gltf")
    result = loader.LoadASCIIFromFile(&model, &error, &warn, filenameStr);
  else if(ext == ".glb")
    result = loader.LoadBinaryFromFile(&model, &error, &warn, filenameStr);
  else
  {
    LOGE("GltfLoader: unsupported extension '%s' for %s", ext.c_str(), filenameStr.c_str());
    return false;
  }

  if(!warn.empty())
    LOGW("GltfLoader: %s", warn.c_str());

  if(!result)
  {
    LOGE("Cannot load %s: %s", filenameStr.c_str(), error.c_str());
    return false;
  }

  mesh.path = filenameStr;

  // Helper: register a glTF texture index and return a local mesh.textures[] index.
  // For external images, stores the URI. For embedded (GLB), stores a synthetic key
  // and captures the raw image bytes so MeshManagerVk can decode them later.
  // sRGB should be true for color textures (base color, emissive) and false for data
  // textures (normal, metallic-roughness, occlusion) so the correct VkFormat is used.
  auto addTexture = [&](int gltfTextureIndex, bool sRGB) -> int {
    if(gltfTextureIndex < 0)
      return -1;
    const auto& texture = model.textures[gltfTextureIndex];
    if(texture.source < 0)
      return -1;
    const auto& image = model.images[texture.source];
    std::string key   = image.uri.empty() ? "<embedded>:" + std::to_string(texture.source) : image.uri;
    mesh.textures.push_back(key);
    mesh.textureSRGB.push_back(sRGB);
    // Store embedded image bytes (empty vector for external files)
    mesh.embeddedImages.push_back(image.image);
    return static_cast<int>(mesh.textures.size()) - 1;
  };

  // --- Materials ---
  for(const auto& gltfMat : model.materials)
  {
    Material mat{};

    const auto& pbr = gltfMat.pbrMetallicRoughness;
    mat.baseColor   = glm::vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);
    mat.metallic    = static_cast<float>(pbr.metallicFactor);
    mat.roughness   = static_cast<float>(pbr.roughnessFactor);
    mat.emissive    = glm::vec3(gltfMat.emissiveFactor[0], gltfMat.emissiveFactor[1], gltfMat.emissiveFactor[2]);
    mat.opacity     = (gltfMat.alphaMode == "OPAQUE") ? 1.0f : static_cast<float>(pbr.baseColorFactor[3]);

    // KHR_materials_ior
    if(auto it = gltfMat.extensions.find("KHR_materials_ior"); it != gltfMat.extensions.end())
    {
      if(it->second.Has("ior"))
        mat.ior = static_cast<float>(it->second.Get("ior").GetNumberAsDouble());
    }
    else
    {
      mat.ior = 1.5f;
    }

    // KHR_materials_transmission
    if(auto it = gltfMat.extensions.find("KHR_materials_transmission"); it != gltfMat.extensions.end())
    {
      if(it->second.Has("transmissionFactor"))
        mat.transmission = static_cast<float>(it->second.Get("transmissionFactor").GetNumberAsDouble());
    }

    // KHR_materials_specular
    if(auto it = gltfMat.extensions.find("KHR_materials_specular"); it != gltfMat.extensions.end())
    {
      if(it->second.Has("specularFactor"))
        mat.specularFactor = static_cast<float>(it->second.Get("specularFactor").GetNumberAsDouble());
      if(it->second.Has("specularColorFactor"))
      {
        const auto& arr = it->second.Get("specularColorFactor");
        if(arr.ArrayLen() >= 3)
        {
          mat.specularColorFactor = glm::vec3(static_cast<float>(arr.Get(0).GetNumberAsDouble()),
                                              static_cast<float>(arr.Get(1).GetNumberAsDouble()),
                                              static_cast<float>(arr.Get(2).GetNumberAsDouble()));
        }
      }
      if(it->second.Has("specularTexture"))
      {
        const auto& texObj = it->second.Get("specularTexture");
        if(texObj.Has("index"))
          mat.specularTexture = addTexture(texObj.Get("index").GetNumberAsInt(), false);
        parseTextureTransformFromValue(texObj, mat.specularUvTransform, mat.specularTexCoord);
      }
      if(it->second.Has("specularColorTexture"))
      {
        const auto& texObj = it->second.Get("specularColorTexture");
        if(texObj.Has("index"))
          mat.specularColorTexture = addTexture(texObj.Get("index").GetNumberAsInt(), true);
        parseTextureTransformFromValue(texObj, mat.specularColorUvTransform, mat.specularColorTexCoord);
      }
    }

    // KHR_materials_clearcoat
    if(auto it = gltfMat.extensions.find("KHR_materials_clearcoat"); it != gltfMat.extensions.end())
    {
      if(it->second.Has("clearcoatFactor"))
        mat.clearcoatFactor = static_cast<float>(it->second.Get("clearcoatFactor").GetNumberAsDouble());
      if(it->second.Has("clearcoatRoughnessFactor"))
        mat.clearcoatRoughness = static_cast<float>(it->second.Get("clearcoatRoughnessFactor").GetNumberAsDouble());
      if(it->second.Has("clearcoatTexture"))
      {
        const auto& texObj = it->second.Get("clearcoatTexture");
        if(texObj.Has("index"))
          mat.clearcoatTexture = addTexture(texObj.Get("index").GetNumberAsInt(), false);
        parseTextureTransformFromValue(texObj, mat.clearcoatUvTransform, mat.clearcoatTexCoord);
      }
      if(it->second.Has("clearcoatRoughnessTexture"))
      {
        const auto& texObj = it->second.Get("clearcoatRoughnessTexture");
        if(texObj.Has("index"))
          mat.clearcoatRoughnessTexture = addTexture(texObj.Get("index").GetNumberAsInt(), false);
        parseTextureTransformFromValue(texObj, mat.clearcoatRoughnessUvTransform, mat.clearcoatRoughnessTexCoord);
      }
    }

    // KHR_materials_emissive_strength
    if(auto it = gltfMat.extensions.find("KHR_materials_emissive_strength"); it != gltfMat.extensions.end())
    {
      if(it->second.Has("emissiveStrength"))
        mat.emissiveStrength = static_cast<float>(it->second.Get("emissiveStrength").GetNumberAsDouble());
    }

    // KHR_materials_pbrSpecularGlossiness (legacy, replaces metallic-roughness)
    if(auto it = gltfMat.extensions.find("KHR_materials_pbrSpecularGlossiness"); it != gltfMat.extensions.end())
    {
      mat.usePbrSpecularGlossiness = 1;
      if(it->second.Has("diffuseFactor"))
      {
        const auto& arr = it->second.Get("diffuseFactor");
        if(arr.ArrayLen() >= 4)
        {
          mat.pbrSgDiffuseFactor = glm::vec4(static_cast<float>(arr.Get(0).GetNumberAsDouble()),
                                             static_cast<float>(arr.Get(1).GetNumberAsDouble()),
                                             static_cast<float>(arr.Get(2).GetNumberAsDouble()),
                                             static_cast<float>(arr.Get(3).GetNumberAsDouble()));
        }
      }
      if(it->second.Has("specularFactor"))
      {
        const auto& arr = it->second.Get("specularFactor");
        if(arr.ArrayLen() >= 3)
        {
          mat.pbrSgSpecularFactor = glm::vec3(static_cast<float>(arr.Get(0).GetNumberAsDouble()),
                                              static_cast<float>(arr.Get(1).GetNumberAsDouble()),
                                              static_cast<float>(arr.Get(2).GetNumberAsDouble()));
        }
      }
      if(it->second.Has("glossinessFactor"))
        mat.pbrSgGlossinessFactor = static_cast<float>(it->second.Get("glossinessFactor").GetNumberAsDouble());
      if(it->second.Has("diffuseTexture"))
      {
        const auto& texObj = it->second.Get("diffuseTexture");
        if(texObj.Has("index"))
          mat.pbrSgDiffuseTexture = addTexture(texObj.Get("index").GetNumberAsInt(), true);
      }
      if(it->second.Has("specularGlossinessTexture"))
      {
        const auto& texObj = it->second.Get("specularGlossinessTexture");
        if(texObj.Has("index"))
          mat.pbrSgSpecularGlossinessTexture = addTexture(texObj.Get("index").GetNumberAsInt(), true);
      }
    }

    mat.maxBounces = 3;  // default for glTF imports

    // All PBR texture slots (sRGB for color maps, linear for data maps)
    mat.baseColorTexture         = addTexture(pbr.baseColorTexture.index, true);
    mat.metallicRoughnessTexture = addTexture(pbr.metallicRoughnessTexture.index, false);
    mat.normalTexture            = addTexture(gltfMat.normalTexture.index, false);
    mat.emissiveTexture          = addTexture(gltfMat.emissiveTexture.index, true);
    mat.occlusionTexture         = addTexture(gltfMat.occlusionTexture.index, false);

    // KHR_texture_transform: per-slot UV transforms and texCoord overrides
    {
      glm::mat3x2 xform;
      int         tc;
      parseTextureTransform(pbr.baseColorTexture, xform, tc);
      mat.baseColorUvTransform = xform;
      mat.baseColorTexCoord    = tc;

      parseTextureTransform(pbr.metallicRoughnessTexture, xform, tc);
      mat.metallicRoughnessUvTransform = xform;
      mat.metallicRoughnessTexCoord    = tc;

      parseTextureTransform(gltfMat.normalTexture, xform, tc);
      mat.normalUvTransform = xform;
      mat.normalTexCoord    = tc;

      parseTextureTransform(gltfMat.emissiveTexture, xform, tc);
      mat.emissiveUvTransform = xform;
      mat.emissiveTexCoord    = tc;

      parseTextureTransform(gltfMat.occlusionTexture, xform, tc);
      mat.occlusionUvTransform = xform;
      mat.occlusionTexCoord    = tc;
    }

    mesh.materials.emplace_back(mat);
    mesh.matNames.emplace_back(gltfMat.name.empty() ? "Material_" + std::to_string(mesh.matNames.size()) : gltfMat.name);
  }

  if(mesh.materials.empty())
  {
    Material defaultMat;
    defaultMat.baseColor = glm::vec3(0.7f, 0.7f, 0.7f);
    defaultMat.roughness = 0.5f;
    defaultMat.metallic  = 0.0f;
    mesh.materials.emplace_back(defaultMat);
    mesh.matNames.emplace_back("Default");
  }

  // --- Scene graph traversal: flatten all nodes into a single mesh ---
  // Each node may carry a transform (TRS or matrix) and reference a mesh.
  // Multiple nodes can reference the same mesh (instancing).
  // We walk the tree, accumulate world transforms, and bake them into vertex data.

  // Process a single glTF mesh, applying worldTransform to all vertices
  auto processMesh = [&](int meshIndex, const glm::mat4& worldTransform) {
    const auto&     gltfMesh      = model.meshes[meshIndex];
    const glm::mat3 normalMatrix  = glm::inverseTranspose(glm::mat3(worldTransform));
    const glm::mat3 tangentMatrix = glm::mat3(worldTransform);
    const float     detSign       = (glm::determinant(glm::mat3(worldTransform)) < 0.0f) ? -1.0f : 1.0f;

    for(const auto& primitive : gltfMesh.primitives)
    {
      if(primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != -1)
        continue;

      auto posIt = primitive.attributes.find("POSITION");
      if(posIt == primitive.attributes.end())
        continue;

      const uint32_t vertexOffset = static_cast<uint32_t>(mesh.vertices.size());
      const size_t   indexStart   = mesh.indices.size();

      // Positions
      const int    posAccessor = posIt->second;
      const int    vertexCount = accessorCount(model, posAccessor);
      const float* posData     = getAccessorData<float>(model, posAccessor);

      // Normals (may be null)
      const float* nrmData = nullptr;
      auto         nrmIt   = primitive.attributes.find("NORMAL");
      if(nrmIt != primitive.attributes.end())
        nrmData = getAccessorData<float>(model, nrmIt->second);

      // Texture coordinates (may be null)
      const float* uvData = nullptr;
      auto         uvIt   = primitive.attributes.find("TEXCOORD_0");
      if(uvIt != primitive.attributes.end())
        uvData = getAccessorData<float>(model, uvIt->second);

      const float* uv1Data = nullptr;
      auto         uv1It   = primitive.attributes.find("TEXCOORD_1");
      if(uv1It != primitive.attributes.end())
        uv1Data = getAccessorData<float>(model, uv1It->second);

      // Tangents (may be null — glTF stores float4: xyz = tangent, w = handedness)
      const float* tanData = nullptr;
      auto         tanIt   = primitive.attributes.find("TANGENT");
      if(tanIt != primitive.attributes.end())
        tanData = getAccessorData<float>(model, tanIt->second);

      // Build vertices with world transform baked in
      mesh.vertices.reserve(mesh.vertices.size() + vertexCount);
      for(int v = 0; v < vertexCount; ++v)
      {
        Vertex vert{};

        glm::vec3 localPos(posData[v * 3 + 0], posData[v * 3 + 1], posData[v * 3 + 2]);
        vert.pos = glm::vec3(worldTransform * glm::vec4(localPos, 1.0f));

        if(nrmData)
        {
          glm::vec3 localNrm(nrmData[v * 3 + 0], nrmData[v * 3 + 1], nrmData[v * 3 + 2]);
          vert.nrm = glm::normalize(normalMatrix * localNrm);
        }
        if(uvData)
          vert.texCoord = glm::vec2(uvData[v * 2 + 0], uvData[v * 2 + 1]);
        if(uv1Data)
          vert.texCoord1 = glm::vec2(uv1Data[v * 2 + 0], uv1Data[v * 2 + 1]);
        if(tanData)
        {
          glm::vec3 localTan(tanData[v * 4 + 0], tanData[v * 4 + 1], tanData[v * 4 + 2]);
          float     handedness = tanData[v * 4 + 3];
          vert.tangent         = glm::vec4(glm::normalize(tangentMatrix * localTan), handedness * detSign);
        }

        mesh.vertices.push_back(vert);
      }

      bool hasTangents = (tanData != nullptr);

      // Indices
      if(primitive.indices >= 0)
      {
        const auto& accessor = model.accessors[primitive.indices];
        const int   idxCount = static_cast<int>(accessor.count);
        mesh.indices.reserve(mesh.indices.size() + idxCount);

        const auto& bufferView = model.bufferViews[accessor.bufferView];
        const auto& buffer     = model.buffers[bufferView.buffer];
        const auto* base       = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

        if(accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
        {
          const auto* idx = reinterpret_cast<const uint16_t*>(base);
          for(int i = 0; i < idxCount; ++i)
            mesh.indices.push_back(vertexOffset + idx[i]);
        }
        else if(accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
        {
          const auto* idx = reinterpret_cast<const uint32_t*>(base);
          for(int i = 0; i < idxCount; ++i)
            mesh.indices.push_back(vertexOffset + idx[i]);
        }
        else if(accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        {
          const auto* idx = reinterpret_cast<const uint8_t*>(base);
          for(int i = 0; i < idxCount; ++i)
            mesh.indices.push_back(vertexOffset + idx[i]);
        }
      }
      else
      {
        // Non-indexed geometry: generate sequential indices
        mesh.indices.reserve(mesh.indices.size() + vertexCount);
        for(int i = 0; i < vertexCount; ++i)
          mesh.indices.push_back(vertexOffset + i);
      }

      // Mirroring transforms flip triangle winding; reverse to keep front-face consistent
      if(detSign < 0.0f)
      {
        for(size_t i = indexStart; i < mesh.indices.size(); i += 3)
          std::swap(mesh.indices[i + 1], mesh.indices[i + 2]);
      }

      // Generate flat normals when none were provided (already in world space since positions are)
      if(!nrmData)
        generateFlatNormals(mesh.vertices, mesh.indices, indexStart);

      // Generate tangents from UVs when the glTF primitive doesn't provide them
      if(!hasTangents)
        computeTangents(mesh.vertices, mesh.indices, indexStart);

      // Per-face material index
      uint32_t matIdx    = (primitive.material >= 0) ? static_cast<uint32_t>(primitive.material) : 0u;
      size_t   faceCount = (mesh.indices.size() - indexStart) / 3;
      mesh.matIndices.insert(mesh.matIndices.end(), faceCount, matIdx);
    }
  };

  // Recursive node traversal — accumulates parent-to-child transforms
  std::function<void(int, const glm::mat4&)> traverseNode = [&](int nodeIndex, const glm::mat4& parentTransform) {
    const auto& node           = model.nodes[nodeIndex];
    glm::mat4   worldTransform = parentTransform * getNodeTransform(node);

    if(node.mesh >= 0)
      processMesh(node.mesh, worldTransform);

    for(int child : node.children)
      traverseNode(child, worldTransform);
  };

  // Start from the default scene (or scene 0), traversing all root nodes
  int sceneIndex = (model.defaultScene >= 0) ? model.defaultScene : 0;
  if(sceneIndex < static_cast<int>(model.scenes.size()))
  {
    for(int rootNode : model.scenes[sceneIndex].nodes)
      traverseNode(rootNode, glm::mat4(1.0f));
  }
  else
  {
    // No scene defined: fall back to iterating all meshes without transforms
    for(int i = 0; i < static_cast<int>(model.meshes.size()); i++)
      processMesh(i, glm::mat4(1.0f));
  }

  // Clamp out-of-range material indices
  for(auto& mi : mesh.matIndices)
  {
    if(mi >= mesh.materials.size())
      mi = 0;
  }

  if(!mesh.isValid())
  {
    LOGE("Invalid glTF file %s", filenameStr.c_str());
    return false;
  }

  return true;
}

}  // namespace vk_gaussian_splatting
