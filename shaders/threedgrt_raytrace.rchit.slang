/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"
#include "threedgrt_payload.h.slang"
#if RTX_HAS_MESHES
#include "mesh_hit_packing.h.slang"
#endif
#if WIREFRAME
#include "nvshaders/wireframe.h.slang"
#endif

// clang-format off
// Always bind assets (no conditional compilation)
// This shader only runs when mesh TLAS is hit, so meshCount > 0 is implicit
[[vk::binding(BINDING_ASSETS, 0)]] ConstantBuffer<SceneAssets> assets;
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
// clang-format on

[shader("closesthit")]
void main(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
#if RTX_HAS_MESHES
  const MeshDesc meshDesc = assets.meshes[InstanceID()];

  if(meshDesc.show == 0)
    return;  // Hidden mesh: leave payload at miss state

  // Indices of the triangle
  int3 tri = ((int3*)meshDesc.indexAddress)[PrimitiveIndex()];

  // Vertex of the triangle
  Vertex v0 = meshDesc.vertexAddress[tri.x];
  Vertex v1 = meshDesc.vertexAddress[tri.y];
  Vertex v2 = meshDesc.vertexAddress[tri.z];

  const float3 barycentrics =
      float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

  // Computing the coordinates of the hit position
  const float3 pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;

  // Computing the normal at hit position
  const float3 nrm      = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  float3       worldNrm = normalize(mul(WorldToObject4x3(), nrm).xyz);
  if(HitKind() == HIT_KIND_TRIANGLE_BACK_FACE)
    worldNrm = -worldNrm;

  // Interpolate texture coordinates
  const float2 uv  = v0.texCoord  * barycentrics.x + v1.texCoord  * barycentrics.y + v2.texCoord  * barycentrics.z;
  const float2 uv1 = v0.texCoord1 * barycentrics.x + v1.texCoord1 * barycentrics.y + v2.texCoord1 * barycentrics.z;

  // Interpolate tangent (direction only, preserve handedness from first vertex)
  const float3 tanDir     = v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z;
  float3       worldTanDir = normalize(mul(WorldToObject4x3(), tanDir).xyz);

  MeshHitData hit;
  hit.hitT       = RayTCurrent();
  hit.worldPos   = float3(mul(float4(pos, 1.0), ObjectToWorld4x3()));
  hit.worldNrm   = worldNrm;
  hit.uv0        = uv;
  hit.tangentDir = worldTanDir;
  hit.tangentW   = v0.tangent.w;
  hit.uv1        = uv1;

  packMeshHitToPayload(payload.dist, payload.id, hit,
                       int(InstanceID()),
                       int(meshDesc.materialIndexAddress[PrimitiveIndex()]));

  // Mesh wireframe: compute edge overlay using ray differentials from raygen
#if WIREFRAME
  const float3 triVerts[3] = { HitTriangleVertexPosition(0), HitTriangleVertexPosition(1), HitTriangleVertexPosition(2) };
  const float3 derivative  = computeDerivativeHighp(WorldToObject4x3(), WorldRayOrigin(), ObjectRayOrigin(),
                                                    triVerts, payload.differentialX.get().value,
                                                    payload.differentialY.get().value, barycentrics);
  const float alpha = processWireframe(wireframeDefaultSettings(), barycentrics, derivative);
  if(alpha > 0.0)
  {
    payload.wireframeDist.set(hit.hitT);
    payload.wireframeAlpha.set(alpha);
  }
#endif
#endif
}
