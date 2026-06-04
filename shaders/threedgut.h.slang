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

#ifndef _GUT_H_
#define _GUT_H_

// 3DGUT hit processing functions for the rasterization fragment shader.
// These wrap the 3DGRT ellipsoidal kernel (from threedgrt.h.slang) but take
// pre-fetched particle attributes as parameters — the mesh shader fetches
// once per particle, and the fragment shader evaluates per fragment.
// Only opacity is returned; color is handled by the mesh shader and blended
// by the Vulkan alpha-blending stage.

#include "threedgrt.h.slang"

bool threedgutProcessHit(in FrameInfo frameInfo,
                         in float3    modelRayOrigin,
                         in float3    modelRayDirection,
                         in int       splatId,
                         in float4    splatColor,
                         in float3    particlePosition,
                         in float3    particleScale,
                         in float3x3  particleInvRotation,
                         out float    opacity)

{
  const float particleDensity = splatColor.a;
  if(particleDensity <= frameInfo.alphaCullThreshold)
#pragma warning(disable: 41018)
    return false;
#pragma warning(default: 41018)

  float3 particleRayOrigin;
  float3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation,
                        particleRayOrigin, particleRayDirection);

  const float rayDist = particleRayMinSquaredDistance(particleRayOrigin, particleRayDirection);

  const float maxResponse = particleRayMaxKernelResponse<KERNEL_DEGREE>(rayDist);
  const float alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  // Kept as float: alpha is already a float quantized at 1/255 granularity, so a
  // double-precision comparison brings no benefit and would otherwise force the
  // module to declare the Float64 SPIR-V capability (broken on devices without
  // VkPhysicalDeviceFeatures::shaderFloat64, e.g. Intel Arc).
  const float minParticleAlpha = 1.0f / 255.0f;
  const bool  acceptHit        = (alpha > minParticleAlpha) && (maxResponse > KERNEL_MIN_RESPONSE);

  if(acceptHit)
  {
#if DISABLE_OPACITY_GAUSSIAN
    opacity = 1.0;
#else
    opacity = alpha;
#endif
  }

  return acceptHit;
}

// Version with normal output for surface generation.
bool threedgutProcessHitWithNormal(in FrameInfo frameInfo,
                                   in float3    modelRayOrigin,
                                   in float3    modelRayDirection,
                                   in int       splatId,
                                   in float4    splatColor,
                                   in float3    particlePosition,
                                   in float3    particleScale,
                                   in float3x3  particleInvRotation,
                                   in float4x4  splatSetTransform,  // For world space normal transform
                                   out float    opacity,
                                   out float3   normalWorld)  // Output: normal in world space

{
  const float particleDensity = splatColor.a;
  if(particleDensity <= frameInfo.alphaCullThreshold)
#pragma warning(disable: 41018)
    return false;
#pragma warning(default: 41018)

  float3 particleRayOrigin;
  float3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation,
                        particleRayOrigin, particleRayDirection);

  const float rayDist = particleRayMinSquaredDistance(particleRayOrigin, particleRayDirection);

  const float maxResponse = particleRayMaxKernelResponse<KERNEL_DEGREE>(rayDist);
  const float alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  // Kept as float: see comment in threedgutProcessHit above.
  const float minParticleAlpha = 1.0f / 255.0f;
  const bool  acceptHit        = (alpha > minParticleAlpha) && (maxResponse > KERNEL_MIN_RESPONSE);

  if(acceptHit)
  {
#if DISABLE_OPACITY_GAUSSIAN
    opacity = 1.0;
#else
    opacity = alpha;
#endif

    // Compute normal using max density plane approximation
    // Create a ThreedgsParticle struct for the normal computation
    ThreedgsParticle particle;
    particle.id          = splatId;
    particle.position    = particlePosition;
    particle.scale       = particleScale;
    particle.invRotation = particleInvRotation;
    // color not needed for normal computation

#if USE_MAX_DENSITY_PLANE_NORMAL
    float3 normalModel = computeEllipsoidNormalMaxDensityPlane(frameInfo, modelRayOrigin, modelRayDirection, particle);
#else
    float3 normalModel =
        computeEllipsoidNormal(frameInfo, modelRayOrigin, modelRayDirection, particleRayOrigin, particleRayDirection, particle);
#endif

    // Transform normal from model space to world space
    // For normals: use transpose(inverse(RS)) but for rotation-scale matrix this is simplified
    // Extract upper-left 3x3 of splatSetTransform for the rotation-scale part
    float3x3 modelToWorldRS = float3x3(splatSetTransform[0].xyz, splatSetTransform[1].xyz, splatSetTransform[2].xyz);
    // For normals, we need inverse-transpose: transpose(inverse(RS))
    // Since modelToWorldRS is RS, inverse(RS) = S^-1 * R^T, and transpose of that = R * S^-1
    // However, for uniform scale or rotation-only, we can just use the matrix directly
    // For correctness with non-uniform scale, we'd need proper inverse-transpose
    // But the normal is already computed in model space accounting for scale
    normalWorld = normalize(mul(normalModel, modelToWorldRS));
  }

  return acceptHit;
}

#endif
