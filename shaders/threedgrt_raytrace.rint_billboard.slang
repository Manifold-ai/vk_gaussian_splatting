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
#include "threedgs_particle_storage.h.slang"
#include "threedgrt.h.slang"
#include "threedgrt_custom_attributes.h.slang"
#include "threedgrt_payload.h.slang"

#include "wireframe_extended.h.slang"

// clang-format off
[[vk::binding(BINDING_ASSETS, 0)]] ConstantBuffer<SceneAssets> assets;
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
// clang-format on

SplatSetDesc* getRtxSplatSetDescriptors()
{
  return (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;
}

bool particleBillboardHitInstance(in float3 modelRayOrigin,
                                  in float3 modelRayDir,
                                  in float3 planeNormal,
                                  in float3 center,
                                  in float3 canonicalRayOrigin,
                                  in float3 canonicalRayDirection,
                                  in float  minHitDistance,
                                  in float  maxHitDistance,
                                  in float  maxSquaredExtent,
                                  out float hitDistance)
{
  hitDistance = 0.0;

  // Ray–plane intersection in model space (same formula as computeBillboardDist)
  float denom = dot(modelRayDir, planeNormal);
  if(abs(denom) < 1e-8)
    return false;

  hitDistance = dot(center - modelRayOrigin, planeNormal) / denom;
  if(hitDistance <= minHitDistance || hitDistance >= maxHitDistance)
    return false;

  // Extent check: perpendicular distance from ray to particle center in canonical space,
  // matching particleDensityHitInstance's acceptance criterion exactly.
  return particleRayMinSquaredDistance(canonicalRayOrigin, normalize(canonicalRayDirection)) < maxSquaredExtent;
}

[shader("intersection")]
void main()
{
  const float3 canonicalRayOrigin    = ObjectRayOrigin();
  const float3 canonicalRayDirection = ObjectRayDirection();

  const uint   descIdx = InstanceID();
  SplatSetDesc desc    = getRtxSplatSetDescriptors()[descIdx];

#if WIREFRAME || DISABLE_OPACITY_GAUSSIAN
  const float  aabbSize = 1.0;
  const float3 aabbMin  = float3(-aabbSize);
  const float3 aabbMax  = float3(aabbSize);

  float aabbHitT;
  if(!rayAABBIntersection(canonicalRayOrigin, canonicalRayDirection, aabbMin, aabbMax, RayTMin(), RayTCurrent(), aabbHitT))
    return;
#endif

  // Camera view direction in model space — used directly as billboard plane normal,
  // exactly matching computeBillboardDist() in the any-hit shader.
  const float3 cameraViewDirWorld = normalize(mul(float4(0, 0, 1, 0), frameInfo.viewInverse).xyz);
  const float3 planeNormal        = normalize(mul(cameraViewDirWorld, desc.transformRotScaleInverse));

  // Particle center in model space = translation of the TLAS instance transform
  const float4x3 otw    = ObjectToWorld4x3();
  const float3   center = otw[3];

  // maxSquaredExtent = 1.0 (1σ): tighter than the ellipsoid shader's 9.0 (3σ), but yields
  // better PSNR vs. mesh-shader ground truth and faster rendering in billboard mode.
  // The precise 2D cutoff is handled by threedgsProcessHit (mahal > 8, matching raster quad).
  float billboardT    = PAYLOAD_INF_DISTANCE;
  bool  hitBillboard  = particleBillboardHitInstance(WorldRayOrigin(), WorldRayDirection(), planeNormal, center,
                                                     canonicalRayOrigin, canonicalRayDirection,
                                                     RayTMin(), RayTCurrent(), 1.0, billboardT);

#if !(WIREFRAME || DISABLE_OPACITY_GAUSSIAN)
  if(!hitBillboard)
    return;
#endif

#if RTX_BILLBOARD_FRUSTUM_CULLING
  {
    float4 worldPos = mul(float4(center, 1.0), desc.transform);
    float4 viewPos  = mul(worldPos, frameInfo.viewMatrix);
    float4 clipPos  = mul(viewPos, frameInfo.projectionMatrix);
    float3 ndcPos   = clipPos.xyz / clipPos.w;
    const float clip = 1.0f + frameInfo.frustumDilation;
    if(abs(ndcPos.x) > clip || abs(ndcPos.y) > clip || ndcPos.z < -frameInfo.frustumDilation || ndcPos.z > 1.0)
      return;
  }
#endif

#if !WIREFRAME
  BuiltInTriangleIntersectionAttributes attr;
  attr.barycentrics = float2(0.0, 0.0);
  ReportHit(billboardT, 0, attr);

#else
  CustomIntersectionAttributes attr;
  attr.aabbHitT     = aabbHitT;
  attr.barycentrics = float2(0.5, 0.5);

  uint hitKind = hitBillboard ? (1 << 4) : 0;

  aabbFaceBarycentrics(canonicalRayOrigin, canonicalRayDirection, aabbHitT, aabbSize, hitKind, attr.barycentrics);

  const float reportT = hitBillboard ? billboardT : aabbHitT;

  ReportHit(reportT, hitKind, attr);

#endif
}

// Ray-AABB intersection using slab method (same as in threedgrt_raytrace.rint.slang)
bool rayAABBIntersection(in float3 rayOrigin,
                         in float3 rayDirection,
                         in float3 aabbMin,
                         in float3 aabbMax,
                         in float  minHitDistance,
                         in float  maxHitDistance,
                         out float hitDistance)
{
  const double3 invDir         = 1.0 / rayDirection;
  const double3 t0             = (aabbMin - rayOrigin) * invDir;
  const double3 t1             = (aabbMax - rayOrigin) * invDir;
  const double3 tmin           = min(t0, t1);
  const double3 tmax           = max(t0, t1);
  const double  tNearUnclamped = max(max(tmin.x, tmin.y), tmin.z);
  const double  tNear          = max(tNearUnclamped, double(minHitDistance));
  const double  tFar           = min(min(tmax.x, tmax.y), min(tmax.z, double(maxHitDistance)));

  if(tNearUnclamped < 0.0)
  {
    hitDistance = maxHitDistance;
    return false;
  }

  hitDistance = float(tNear);
  return (tNear <= tFar);
}
