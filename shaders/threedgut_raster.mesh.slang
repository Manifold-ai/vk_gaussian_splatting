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

#include "shaderio.h"
#include "threedgs.h.slang"
#include "threedgs_particle_storage.h.slang"
#include "threedgut_projector.h.slang"

// Parallel Processing : Each global invocation (thread) processes one splat.
// Batch Processing : The workgroup can process up to RASTER_MESH_WORKGROUP_SIZE splats(outputQuadCount)

static const int MAX_VERTICES   = 4 * RASTER_MESH_WORKGROUP_SIZE;
static const int MAX_PRIMITIVES = 2 * RASTER_MESH_WORKGROUP_SIZE;

// clang-format off

// Splat attributes passed to the fragment shader.
//
// ROOT CAUSE (Slang issue shader-slang/slang#7019): Slang decorates per-primitive mesh *outputs*
// with the required PerPrimitiveEXT decoration, but has no way to emit the matching PerPrimitiveEXT
// on the fragment-shader *inputs*. NVIDIA tolerates the resulting interface mismatch (it links
// per-primitive vars by location at PSO creation), but every non-NVIDIA vendor (AMD, Intel, ...)
// reads the per-primitive data at per-vertex rate -> garbage/missing attributes. The Vulkan
// validation layers now also flag this (KhronosGroup/Vulkan-ValidationLayers#8398). It is NOT a
// driver bug dropping outputs.
//
// WORKAROUND (gated by MESH_SHADER_PER_PRIMITIVE_ATTRIBS, see shaderio.h): on non-NVIDIA vendors, emit the
// attributes per-vertex (duplicated across the 4 quad vertices) with nointerpolation. NVIDIA keeps
// the per-primitive path (one write per primitive, less output bandwidth), matching the integration
// reference. The fragment shader reads them purely by location/semantic, so the same frag works for
// both layouts.
// TODO: when Slang #7019 lands (PerPrimitiveEXT on frag inputs), drop this gate and use per-primitive
// everywhere (also fixes the validation warning on NVIDIA/AMD).
#if !MESH_SHADER_PER_PRIMITIVE_ATTRIBS
struct PrimitiveOutput
{
};
struct VertexOutput
{
  float4 position : SV_Position;
  [[vk::location(0)]] nointerpolation uint splatId : SPLATID;
  [[vk::location(1)]] nointerpolation float4 splatCol : COLOR0;
  [[vk::location(2)]] nointerpolation float3 splatPosition : POSITION0;
  [[vk::location(3)]] nointerpolation float3 splatScale : SCALE0;
  [[vk::location(4)]] nointerpolation float3 splatInvRotation0 : ROTATION0;
  [[vk::location(5)]] nointerpolation float3 splatInvRotation1 : ROTATION1;
  [[vk::location(6)]] nointerpolation float3 splatInvRotation2 : ROTATION2;
};
#else
struct PrimitiveOutput
{
  [[vk::location(0)]] nointerpolation uint splatId : SPLATID;
  [[vk::location(1)]] nointerpolation float4 splatCol : COLOR0;
  [[vk::location(2)]] nointerpolation float3 splatPosition : POSITION0;
  [[vk::location(3)]] nointerpolation float3 splatScale : SCALE0;
  [[vk::location(4)]] nointerpolation float3 splatInvRotation0 : ROTATION0;  // matrix row 0
  [[vk::location(5)]] nointerpolation float3 splatInvRotation1 : ROTATION1;  // matrix row 1
  [[vk::location(6)]] nointerpolation float3 splatInvRotation2 : ROTATION2;  // matrix row 2
};
struct VertexOutput
{
  float4 position : SV_Position;
};
#endif

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(BINDING_ASSETS, 0)]] ConstantBuffer<SceneAssets> assets;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;
// to get the actual number of splats (after culling if any)
[[vk::binding(BINDING_INDIRECT_BUFFER, 0)]] RWStructuredBuffer<IndirectParams> indirectBuffer;

// clang-format on

// used when quad need to be discard
void emitDegeneratedQuad(uint localIndex, out OutputVertices<VertexOutput, MAX_VERTICES> verts)
{
  [unroll]
  for(uint i = 0; i < 4; ++i)
  {
    verts[localIndex * 4 + i].position          = float4(0.0, 0.0, 2.0, 1.0);
#if !MESH_SHADER_PER_PRIMITIVE_ATTRIBS
    verts[localIndex * 4 + i].splatId           = 0;
    verts[localIndex * 4 + i].splatCol          = float4(0.0);
    verts[localIndex * 4 + i].splatPosition     = float3(0.0);
    verts[localIndex * 4 + i].splatScale        = float3(0.0);
    verts[localIndex * 4 + i].splatInvRotation0 = float3(0.0);
    verts[localIndex * 4 + i].splatInvRotation1 = float3(0.0);
    verts[localIndex * 4 + i].splatInvRotation2 = float3(0.0);
#endif
  }
}

[numthreads(RASTER_MESH_WORKGROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
[shader("mesh")]
void main(
  uint3 groupThreadID : SV_GroupThreadID,
  uint3 groupID : SV_GroupID,
  uint3 dispatchThreadID : SV_DispatchThreadID,
  out OutputVertices<VertexOutput, MAX_VERTICES> verts,
  out OutputPrimitives<PrimitiveOutput, MAX_PRIMITIVES> prims,
  out OutputIndices<uint3, MAX_PRIMITIVES> triangles)
{
  const uint localIndex = groupThreadID.x;
  // Mesh tasks are dispatched as a 2D grid (X = MESH_SHADER_MAX_GROUPS_X = device
  // maxMeshWorkGroupCount[0]) so very large scenes stay under the per-dimension limit. Recover the
  // linear workgroup index, then the per-thread global splat slot. With groupID.y == 0 (groupCountY
  // == 1) this is identical to a plain 1D dispatch (linearGroup == groupID.x).
  const uint     linearGroup = groupID.y * MESH_SHADER_MAX_GROUPS_X + groupID.x;
  const uint32_t baseIndex   = linearGroup * RASTER_MESH_WORKGROUP_SIZE + localIndex;
#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST && USE_GPU_DIST_SORT
  const uint splatCount = indirectBuffer[0].instanceCount;
#else
  const uint splatCount = frameInfo.splatCount;
#endif
  const uint groupBase       = linearGroup * RASTER_MESH_WORKGROUP_SIZE;
  const uint outputQuadCount = groupBase < splatCount ? min(RASTER_MESH_WORKGROUP_SIZE, splatCount - groupBase) : 0;

  if(localIndex == 0)
  {
    // set the number of vertices and primitives to put out just once for the complete workgroup
    SetMeshOutputCounts(outputQuadCount * 4, outputQuadCount * 2);
  }

  if(baseIndex < splatCount)
  {
    // Access sorted indices via bindless device address
    // Explicit 64-bit byte offset to avoid 32-bit overflow at ~1G visible splats
    uint64_t       byteOffset    = uint64_t(baseIndex) * uint64_t(sizeof(uint32_t));
    const uint32_t globalSplatID = *(uint32_t*)(assets.splatSortingIndicesAddress + byteOffset);

    // Resolve global splat ID to correct descriptor and local index
    SplatSetDesc splatSetDesc;
    uint splatIndex;
    resolveGlobalSplatID(assets, globalSplatID, splatSetDesc, splatIndex);
    
    // emit primitives (triangles) as soon as possible
    triangles[localIndex * 2 + 0] = uint3(0, 2, 1) + localIndex * 4;
    triangles[localIndex * 2 + 1] = uint3(2, 0, 3) + localIndex * 4;
    
    // Fetch particle minimal data 
    // Fetches are done as early as possible.
    // Moving those after culling statments will reduce the performance
    float4       splatColor  = fetchColor(splatSetDesc, splatIndex);  // contains opacity as .a
    const float3 splatCenter = fetchCenter(splatSetDesc, splatIndex);
    const float3 splatScale  = exp(fetchScale(splatSetDesc, splatIndex));
    // INRIA quaternions are scalar first: w in splatRotation.x and xyz in splatRotation.yzw)
    // our internal representation (hence splatRotation), uses xyz, w. vec4ToQuat does convertion
    const float3x3 splatRotation = quatToMat3(normalize(vec4toQuat(fetchRotation(splatSetDesc, splatIndex))));
    const float3x3 splatInvRotation = transpose(splatRotation);

#if MESH_SHADER_PER_PRIMITIVE_ATTRIBS
    // NVIDIA: emit per-primitive data as early as possible (perf). The non-NVIDIA path emits the same
    // values per-vertex at the end of the shader (see header / Slang #7019).
    prims[localIndex * 2 + 0].splatId           = globalSplatID;
    prims[localIndex * 2 + 1].splatId           = globalSplatID;
    prims[localIndex * 2 + 0].splatPosition     = splatCenter;
    prims[localIndex * 2 + 1].splatPosition     = splatCenter;
    prims[localIndex * 2 + 0].splatScale        = splatScale;
    prims[localIndex * 2 + 1].splatScale        = splatScale;
    prims[localIndex * 2 + 0].splatInvRotation0 = splatInvRotation[0];
    prims[localIndex * 2 + 1].splatInvRotation0 = splatInvRotation[0];
    prims[localIndex * 2 + 0].splatInvRotation1 = splatInvRotation[1];
    prims[localIndex * 2 + 1].splatInvRotation1 = splatInvRotation[1];
    prims[localIndex * 2 + 0].splatInvRotation2 = splatInvRotation[2];
    prims[localIndex * 2 + 1].splatInvRotation2 = splatInvRotation[2];
#endif

    // fetch radiance from SH coefs > degree 0, we work in model coordinates
    const float3 worldViewDir = normalize(splatCenter - mul(float4(frameInfo.cameraPosition, 1.0), splatSetDesc.transformInverse).xyz);

#if SHOW_SH_ONLY == 1
    splatColor.rgb = float3(0.5);
#endif
    splatColor.rgb += fetchViewDependentRadiance(splatSetDesc, frameInfo.shDegree, splatIndex, worldViewDir);

    // alpha based culling
    if(splatColor.a < frameInfo.alphaCullThreshold)
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

#if MESH_SHADER_PER_PRIMITIVE_ATTRIBS && MS_ANTIALIASING == 0
    // emit per-primitive color as early as possible for perf reasons (MipSplatting compensates the
    // alpha later, so its color is emitted after the extent computation instead)
    prims[localIndex * 2 + 0].splatCol = splatColor;
    prims[localIndex * 2 + 1].splatCol = splatColor;
#endif

#if CAMERA_TYPE == CAMERA_PINHOLE
    CameraModelParameters sensorModel = initPerfectPinholeCamera(frameInfo.nearFar, frameInfo.viewport, frameInfo.focal);
#else
    CameraModelParameters sensorModel = initPerfectFisheyeCamera(frameInfo.viewport, frameInfo.focal);
#endif

    // camera pose in world space, used by projector to compute the ray.
    SensorState sensorState = initGlobalShutterSensorState(frameInfo.viewTrans, frameInfo.viewQuat);

    int2 resolution = int2(int(frameInfo.viewport.x), int(frameInfo.viewport.y));

    float3 particleSensorRay;
    float2 particleProjCenter;
    float3 particleProjCovariance;

    if(!threedgutParticleProjection(resolution, sensorModel, splatSetDesc.transform, sensorState, splatCenter, splatScale,
                                    splatRotation, particleProjCenter, particleProjCovariance))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

// Method from paper used for cuda raster
#if EXTENT_METHOD == EXTENT_CONIC
    float2  extent;
    float4  conicOpacity;
    float maxConicOpacityPower;

    if(!threedgutProjectedExtentConicOpacity(particleProjCovariance, splatColor.a, extent, conicOpacity, maxConicOpacityPower))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

#if MS_ANTIALIASING == 1
    splatColor.a = conicOpacity.a;
#endif

// Method optimized for rasterisation pipelines
#elif EXTENT_METHOD == EXTENT_EIGEN

    float2 basisVector1, basisVector2;

    if(!threedgsProjectedExtentBasis(particleProjCovariance, 3.33, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }
#endif

#if MESH_SHADER_PER_PRIMITIVE_ATTRIBS && MS_ANTIALIASING == 1
    // emit per-primitive color after threedgutProjectedExtentConicOpacity / threedgsProjectedExtentBasis
    // modified splatColor.a for MipSplatting alpha compensation
    prims[localIndex * 2 + 0].splatCol = splatColor;
    prims[localIndex * 2 + 1].splatCol = splatColor;
#endif

    // Convert projected particle center from pixel coordinates to NDC [-1, 1]
    // particleProjCenter is in pixel coordinates, need to convert to NDC
    const float2 ndcXY = (particleProjCenter / frameInfo.viewport) * 2.0 - 1.0;

    const float4 worldSplatCenter = mul(float4(splatCenter, 1.0), splatSetDesc.transform);
    const float4 viewSplatCenter  = mul(worldSplatCenter, frameInfo.viewMatrix);
    // TODO: projection matrix is not correct with fisheye, this is a coarse aprox to compute depth
    const float4 clipSplatCenter = mul(viewSplatCenter, frameInfo.projectionMatrixJittered);

    const float3 ndcCenter = float3(ndcXY, clipSplatCenter.z / clipSplatCenter.w);
#if EXTENT_METHOD == EXTENT_CONIC
    const float2 ndcExtent = extent * frameInfo.basisViewport * 2.0;
#endif

    /////////////////////////////
    // emiting quad vertices

    const float2 vertexPositions[4] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};

    [unroll]
    for(uint i = 0; i < 4; ++i)
    {
      const float2 vertPos = vertexPositions[i].xy;

#if EXTENT_METHOD == EXTENT_CONIC

      const float2 ndcOffset = vertPos * ndcExtent;

#elif EXTENT_METHOD == EXTENT_EIGEN

      const float2 ndcOffset = float2(vertPos.x * basisVector1 + vertPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                             * frameInfo.inverseFocalAdjustment;

#endif
      const float4 quadPos = float4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

      verts[localIndex * 4 + i].position          = quadPos;
#if !MESH_SHADER_PER_PRIMITIVE_ATTRIBS
      // Non-NVIDIA: emit the varyings per-vertex (duplicated); see header / Slang #7019
      verts[localIndex * 4 + i].splatId           = globalSplatID;
      verts[localIndex * 4 + i].splatCol          = splatColor;
      verts[localIndex * 4 + i].splatPosition     = splatCenter;
      verts[localIndex * 4 + i].splatScale        = splatScale;
      verts[localIndex * 4 + i].splatInvRotation0 = splatInvRotation[0];
      verts[localIndex * 4 + i].splatInvRotation1 = splatInvRotation[1];
      verts[localIndex * 4 + i].splatInvRotation2 = splatInvRotation[2];
#endif
    }
  }
}