/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#define TINYOBJLOADER_IMPLEMENTATION
#include "obj_loader.h"
#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

bool ObjLoader::load(const std::filesystem::path& filename, Mesh& mesh)
{
  mesh = {};

  tinyobj::ObjReader reader;
  reader.ParseFromFile(filename.string());
  if(!reader.Valid())
  {
    LOGE("Cannot load %s: %s", filename.string().c_str(), reader.Error().c_str());
    return false;
  }

  mesh.path = filename.string();

  // Collecting the material in the scene — convert OBJ Phong to PBR metallic-roughness
  for(const auto& material : reader.GetMaterials())
  {
    Material m;

    glm::vec3 diffuse  = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
    glm::vec3 specular = glm::vec3(material.specular[0], material.specular[1], material.specular[2]);
    glm::vec3 transmit = glm::vec3(material.transmittance[0], material.transmittance[1], material.transmittance[2]);

    m.baseColor = diffuse;
    m.emissive  = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);

    // Phong shininess → GGX roughness approximation
    float shininess = std::max(material.shininess, 1.0f);
    m.roughness     = std::sqrt(2.0f / (shininess + 2.0f));

    // Heuristic: high specular with low diffuse → metallic
    float specLum = glm::length(specular);
    float diffLum = glm::length(diffuse);
    m.metallic    = (specLum > 0.5f && diffLum < 0.1f) ? 1.0f : 0.0f;

    m.ior     = (material.ior > 0.0f) ? material.ior : 1.5f;
    m.opacity = material.dissolve;

    // Transmission from OBJ transmittance
    float transmitLen = glm::length(transmit);
    m.transmission    = (transmitLen > 0.001f) ? 1.0f : 0.0f;

    // Map OBJ texture names to PBR texture slots
    // sRGB for color maps (diffuse, emissive), linear for data maps (normal, specular)
    auto addTex = [&](const std::string& name, bool sRGB) -> int {
      if(name.empty())
        return -1;
      mesh.textures.push_back(name);
      mesh.textureSRGB.push_back(sRGB);
      return static_cast<int>(mesh.textures.size()) - 1;
    };
    m.baseColorTexture         = addTex(material.diffuse_texname, true);
    m.normalTexture            = addTex(material.bump_texname, false);
    m.metallicRoughnessTexture = addTex(material.specular_texname, false);
    m.emissiveTexture          = addTex(material.emissive_texname, true);

    mesh.materials.emplace_back(m);
    mesh.matNames.emplace_back(material.name);
  }

  // If there were none, add a default
  if(mesh.materials.empty())
  {
    Material defaultMat;
    defaultMat.baseColor = glm::vec3(0.7f, 0.7f, 0.7f);
    defaultMat.roughness = 0.5f;
    defaultMat.metallic  = 0.0f;
    mesh.materials.emplace_back(defaultMat);
    mesh.matNames.emplace_back("Default");
  }

  const tinyobj::attrib_t& attrib = reader.GetAttrib();

  // storage to generate some normal vectors if needed
  std::vector<uint8_t>   visited;
  std::vector<glm::vec3> normals;

  for(const auto& shape : reader.GetShapes())
  {
    mesh.vertices.reserve(shape.mesh.indices.size() + mesh.vertices.size());
    mesh.indices.reserve(shape.mesh.indices.size() + mesh.indices.size());
    mesh.matIndices.insert(mesh.matIndices.end(), shape.mesh.material_ids.begin(), shape.mesh.material_ids.end());

    // If we do not have normal vectors we generate some
    if(true)  //attrib.normals.empty())
    {
      // Compute per vertex normal when no normal were provided.
      // no smoothing groups or crease angle. avarages per face
      // normals to compute per vertex normals
      visited.resize(visited.size() + shape.mesh.indices.size(), 0);
      normals.resize(normals.size() + shape.mesh.indices.size());

      // iterate over the faces
      for(size_t i = 0; i < shape.mesh.indices.size(); i += 3)
      {
        const auto& index0 = shape.mesh.indices[i + 0].vertex_index;
        const auto& index1 = shape.mesh.indices[i + 1].vertex_index;
        const auto& index2 = shape.mesh.indices[i + 2].vertex_index;
        const auto  vp0 =
            glm::vec3(attrib.vertices[3 * index0], attrib.vertices[3 * index0 + 1], attrib.vertices[3 * index0 + 2]);
        const auto vp1 =
            glm::vec3(attrib.vertices[3 * index1], attrib.vertices[3 * index1 + 1], attrib.vertices[3 * index1 + 2]);
        const auto vp2 =
            glm::vec3(attrib.vertices[3 * index2], attrib.vertices[3 * index2 + 1], attrib.vertices[3 * index2 + 2]);

        glm::vec3 n = glm::normalize(glm::cross((vp1 - vp0), (vp2 - vp0)));

        glm::vec3& nrm0 = normals[index0];
        glm::vec3& nrm1 = normals[index1];
        glm::vec3& nrm2 = normals[index2];

        // dispatch the normal to each of the face vertex
        if(visited[index0])
        {
          nrm0 = glm::mix(nrm0, n, 0.5);
        }
        else
        {
          nrm0            = n;
          visited[index0] = 1;
        }
        if(visited[index1])
        {
          nrm1 = glm::mix(nrm1, n, 0.5);
        }
        else
        {
          nrm1            = n;
          visited[index1] = 1;
        }
        if(visited[index2])
        {
          nrm2 = glm::mix(nrm2, n, 0.5);
        }
        else
        {
          nrm2            = n;
          visited[index2] = 1;
        }
      }
    }

    // prepare output so that all attributes uses
    // same primary index for rendering
    for(const auto& index : shape.mesh.indices)
    {
      Vertex       vertex = {};
      const float* vp     = &attrib.vertices[3 * index.vertex_index];
      vertex.pos          = {*(vp + 0), *(vp + 1), *(vp + 2)};

      if(!attrib.normals.empty() && index.normal_index >= 0)
      {
        const float* np = &attrib.normals[3 * index.normal_index];
        vertex.nrm      = {*(np + 0), *(np + 1), *(np + 2)};
      }
      else
      {
        vertex.nrm = normals[index.vertex_index];
      }

      if(!attrib.texcoords.empty() && index.texcoord_index >= 0)
      {
        const float* tp = &attrib.texcoords[2 * index.texcoord_index + 0];
        vertex.texCoord = {*tp, 1.0f - *(tp + 1)};
      }
      mesh.vertices.push_back(vertex);
      mesh.indices.push_back(static_cast<uint32_t>(mesh.indices.size()));
    }
  }

  // Fixing material indices
  for(auto& mi : mesh.matIndices)
  {
    if(mi >= mesh.materials.size())
      mi = 0;
  }

  // Generate tangent vectors from UVs (needed for normal mapping)
  computeTangents(mesh.vertices, mesh.indices);

  if(!mesh.isValid())
  {
    LOGE("Invalid Obj file %s \n", filename.c_str());
    return false;
  }

  return true;
}

}  // namespace vk_gaussian_splatting