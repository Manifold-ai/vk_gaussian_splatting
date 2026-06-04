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

#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>

#include "shaderio.h"

using Material = shaderio::Material;
using Vertex   = shaderio::Vertex;

namespace vk_gaussian_splatting {

struct Mesh
{
  std::string              path;        // source file path or display name
  std::vector<Vertex>      vertices;    // triangle vertices
  std::vector<uint32_t>    indices;     // triangle vertex indices
  std::vector<uint32_t>    matIndices;  // per face material indices
  std::vector<Material>    materials;   // materials
  std::vector<std::string> matNames;    // material names
  std::vector<std::string> textures;    // texture name/key per local index
  std::vector<bool> textureSRGB;  // true = sRGB color texture, false = linear data texture (parallel to textures[])

  // Embedded image data for GLB textures (parallel to textures[]; empty for external files)
  std::vector<std::vector<uint8_t>> embeddedImages;

  uint32_t nbVertices() const { return static_cast<uint32_t>(vertices.size()); }
  uint32_t nbIndices() const { return static_cast<uint32_t>(indices.size()); }
  bool     isValid() const { return !vertices.empty() && !indices.empty(); }
};

// Compute tangent vectors from triangle edges and UVs (Lengyel's method).
// Accumulates per-vertex tangents/bitangents, then orthogonalizes and stores
// the result as float4(T.xyz, handedness) in each vertex.
// indexStart lets you limit the computation to a subset of the index buffer.
inline void computeTangents(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, size_t indexStart = 0)
{
  std::vector<glm::vec3> tan1(vertices.size(), glm::vec3(0.0f));
  std::vector<glm::vec3> tan2(vertices.size(), glm::vec3(0.0f));

  for(size_t i = indexStart; i + 2 < indices.size(); i += 3)
  {
    uint32_t i0 = indices[i + 0];
    uint32_t i1 = indices[i + 1];
    uint32_t i2 = indices[i + 2];

    const glm::vec3& p0 = vertices[i0].pos;
    const glm::vec3& p1 = vertices[i1].pos;
    const glm::vec3& p2 = vertices[i2].pos;

    const glm::vec2& uv0 = vertices[i0].texCoord;
    const glm::vec2& uv1 = vertices[i1].texCoord;
    const glm::vec2& uv2 = vertices[i2].texCoord;

    glm::vec3 edge1    = p1 - p0;
    glm::vec3 edge2    = p2 - p0;
    glm::vec2 deltaUV1 = uv1 - uv0;
    glm::vec2 deltaUV2 = uv2 - uv0;

    float det = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;
    if(std::abs(det) < 1e-8f)
      continue;

    float     invDet = 1.0f / det;
    glm::vec3 sdir   = (deltaUV2.y * edge1 - deltaUV1.y * edge2) * invDet;
    glm::vec3 tdir   = (deltaUV1.x * edge2 - deltaUV2.x * edge1) * invDet;

    tan1[i0] += sdir;
    tan1[i1] += sdir;
    tan1[i2] += sdir;
    tan2[i0] += tdir;
    tan2[i1] += tdir;
    tan2[i2] += tdir;
  }

  // Gram-Schmidt orthogonalize and compute handedness
  for(size_t v = 0; v < vertices.size(); ++v)
  {
    const glm::vec3& n = vertices[v].nrm;
    const glm::vec3& t = tan1[v];

    if(glm::length(t) < 1e-8f)
    {
      vertices[v].tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);  // fallback
      continue;
    }

    // Orthogonalize: T' = normalize(T - N * dot(N, T))
    glm::vec3 tangent = glm::normalize(t - n * glm::dot(n, t));

    // Handedness: sign of dot(cross(N, T), B)
    float w = (glm::dot(glm::cross(n, t), tan2[v]) < 0.0f) ? -1.0f : 1.0f;

    vertices[v].tangent = glm::vec4(tangent, w);
  }
}

}  // namespace vk_gaussian_splatting
