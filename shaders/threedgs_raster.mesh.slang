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

/*
* Some mathematical formulations and comments have been directly retained from
* https://github.com/mkkellogg/GaussianSplats3D. Original source code  
* licence hereafter.
* ----------------------------------
* The MIT License (MIT)
* 
* Copyright (c) 2023 Mark Kellogg
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "shaderio.h"
#include "threedgs.h.slang"  // provides ThreedgsParticle + computeEllipsoidNormalMaxDensityPlane
#include "threedgs_particle_storage.h.slang"

#if NEED_SURFACE_INFO && QUANTIZE_NORMALS
#include "octahedral_normal.h.slang"
#endif

// Parallel Processing : Each global invocation (thread) processes one splat.
// Batch Processing : The workgroup can process up to RASTER_MESH_WORKGROUP_SIZE splats(outputQuadCount)

static const int MAX_VERTICES   = 4 * RASTER_MESH_WORKGROUP_SIZE;
static const int MAX_PRIMITIVES = 2 * RASTER_MESH_WORKGROUP_SIZE;

// clang-format off

// Splat attributes passed to the fragment shader.
//
// ROOT CAUSE (Slang issue shader-slang/slang#7019): Slang decorates per-primitive mesh *outputs*
// with the required PerPrimitiveEXT decoration, but has no way to emit the matching PerPrimitiveEXT
// on the fragment-shader *inputs* (verified by disassembling our SPIR-V: mesh out = PerPrimitiveEXT,
// frag in = Flat only). NVIDIA tolerates this interface mismatch (it links per-primitive vars by
// location at PSO creation), but every non-NVIDIA vendor (AMD, Intel, ...) reads the per-primitive
// data at per-vertex rate, which yields garbage/missing attributes. The Vulkan validation layers now
// also flag this mismatch (KhronosGroup/Vulkan-ValidationLayers#8398). It is NOT a driver bug
// dropping outputs.
//
// WORKAROUND (gated by MESH_SHADER_PER_PRIMITIVE_ATTRIBS, see shaderio.h): on non-NVIDIA vendors, emit the
// flat attributes (color, splat id, normal) per-vertex (duplicated across the 4 quad vertices) with
// nointerpolation. NVIDIA keeps the per-primitive path (less output bandwidth), matching the
// integration reference. The per-vertex quad corner (outFragPos) and position stay per-vertex in both
// cases; the fragment shader reads everything by location/semantic.
// TODO: when Slang #7019 lands (PerPrimitiveEXT on frag inputs), drop this gate and use per-primitive
// everywhere (also fixes the validation warning on AMD/Intel).
#if !MESH_SHADER_PER_PRIMITIVE_ATTRIBS
struct PrimitiveOutput
{
};
struct VertexOutput
{
  [[vk::location(0)]] nointerpolation float4 outSplatCol : COLOR0;
#if !USE_BARYCENTRIC
  [[vk::location(1)]] float2 outFragPos : TEXCOORD0;
#endif
  [[vk::location(2)]] float4 position : SV_Position;
#if STOCHASTIC_SPLAT || NEED_SURFACE_INFO
  [[vk::location(3)]] nointerpolation uint outSplatId : SPLATID;
#endif
#if NEED_SURFACE_INFO
#if QUANTIZE_NORMALS
  // Octahedral encoding: unit normal packed into 32-bit uint (16 bits per component)
  [[vk::location(4)]] nointerpolation uint outSplatNormalPacked : NORMAL0;
#else
  [[vk::location(4)]] nointerpolation float3 outSplatNormal : NORMAL0;
#endif
#endif
};
#else
// Per primitive output
struct PrimitiveOutput
{
  [[vk::location(0)]] nointerpolation float4 outSplatCol : COLOR0;
#if STOCHASTIC_SPLAT || NEED_SURFACE_INFO
  [[vk::location(3)]] nointerpolation uint outSplatId : SPLATID;
#endif
#if NEED_SURFACE_INFO
#if QUANTIZE_NORMALS
  // Octahedral encoding: unit normal packed into 32-bit uint (16 bits per component)
  [[vk::location(4)]] nointerpolation uint outSplatNormalPacked : NORMAL0;
#else
  [[vk::location(4)]] nointerpolation float3 outSplatNormal : NORMAL0;
#endif
#endif
};

// Per vertex output
struct VertexOutput
{
#if !USE_BARYCENTRIC
  [[vk::location(1)]] float2 outFragPos : TEXCOORD0;
#endif
  [[vk::location(2)]] float4 position : SV_Position;
};
#endif

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]  ConstantBuffer<FrameInfo> frameInfo;
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
    verts[localIndex * 4 + i].position   = float4(0.0, 0.0, 2.0, 1.0);
#if !MESH_SHADER_PER_PRIMITIVE_ATTRIBS
    verts[localIndex * 4 + i].outSplatCol = float4(0.0);
#endif
  }
}

[numthreads(RASTER_MESH_WORKGROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
[shader("mesh")]
void main(uint3                                             groupThreadID: SV_GroupThreadID,
          uint3                                             groupID: SV_GroupID,
          uint3                                             dispatchThreadID: SV_DispatchThreadID,
          OutputIndices<uint3, MAX_PRIMITIVES>              tris,
          OutputVertices<VertexOutput, MAX_VERTICES>        verts,
          OutputPrimitives<PrimitiveOutput, MAX_PRIMITIVES> prims)
{
  const uint localIndex = groupThreadID.x;
  // Mesh tasks are dispatched as a 2D grid (X = MESH_SHADER_MAX_GROUPS_X = device
  // maxMeshWorkGroupCount[0]) so very large scenes stay under the per-dimension limit. Recover the
  // linear workgroup index, then the per-thread global splat slot. With groupID.y == 0 (groupCountY
  // == 1) this is identical to a plain 1D dispatch (linearGroup == groupID.x).
  const uint     linearGroup = groupID.y * MESH_SHADER_MAX_GROUPS_X + groupID.x;
  const uint32_t baseIndex   = linearGroup * RASTER_MESH_WORKGROUP_SIZE + localIndex;

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST && USE_GPU_DIST_SORT
  // Distance compute (GPU radix / stochastic) fills instanceCount after culling
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
    uint         splatIndex;
    resolveGlobalSplatID(assets, globalSplatID, splatSetDesc, splatIndex);

    // emit primitives (triangles) as soon as possible
    tris[localIndex * 2 + 0] = uint3(0, 2, 1) + localIndex * 4;
    tris[localIndex * 2 + 1] = uint3(2, 0, 3) + localIndex * 4;

#if MESH_SHADER_PER_PRIMITIVE_ATTRIBS && (STOCHASTIC_SPLAT || NEED_SURFACE_INFO)
    // NVIDIA: emit the splat ID per-primitive as soon as possible. The non-NVIDIA path emits it
    // per-vertex at the end (see header / Slang #7019).
    prims[localIndex * 2 + 0].outSplatId = globalSplatID;
    prims[localIndex * 2 + 1].outSplatId = globalSplatID;
#endif

    // Fetches are done as needed to allow early exit (culling)
    float4 splatColor = fetchColor(splatSetDesc, splatIndex);

    // alpha based culling
    if(splatColor.a < frameInfo.alphaCullThreshold)
    {
      // Early return to discard splat
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

    const float3 splatCenter = fetchCenter(splatSetDesc, splatIndex);
    
    // Use transform from descriptor, not push constants
    const float4x4 modelViewMatrix = mul(splatSetDesc.transform, frameInfo.viewMatrix);

    // Row-major: vector * matrix
    const float4 viewCenter = mul(float4(splatCenter, 1.0), modelViewMatrix);
    const float4 clipCenter = mul(viewCenter, frameInfo.projectionMatrixJittered);

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_RASTER
    const float clip = (1.0 + frameInfo.frustumDilation) * clipCenter.w;
    if(abs(clipCenter.x) > clip || abs(clipCenter.y) > clip
       || clipCenter.z < (0.0 - frameInfo.frustumDilation) * clipCenter.w || clipCenter.z > clipCenter.w)
    {
      // Early return to discard splat
      emitDegeneratedQuad(localIndex, verts);
      return;
    }
#endif

    // the vertices of the quad
    const float2 positions[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0) };

#if SHOW_SH_ONLY == 1
    splatColor.rgb = float3(0.5, 0.5, 0.5);
#endif

#if NEED_SURFACE_INFO
#if QUANTIZE_NORMALS
    uint splatNormalPacked = 0;
#else
    float3 splatNormal = float3(0.0);
#endif
    {
      // Create particle struct for normal computation (position, scale, rotation only)
      ThreedgsParticle particle = ThreedgsParticle(splatSetDesc, splatIndex, true);

      // Camera position in model space (inverse transform)
      const float3 modelCameraPos = mul(float4(frameInfo.cameraPosition, 1.0), splatSetDesc.transformInverse).xyz;
      const float3 modelRayDir    = normalize(splatCenter - modelCameraPos);

      // Use the robust function that handles degenerate (flat) particles
      float3 normalModel = computeEllipsoidNormalMaxDensityPlane(frameInfo, modelCameraPos, modelRayDir, particle);

      // Transform normal to world space
      // For normals, we use transpose(inverse(M)), but for pure rotation this is just the rotation part
      float3 normalWorld = normalize(mul(float4(normalModel, 0.0), splatSetDesc.transform).xyz);
#if QUANTIZE_NORMALS
      // Pack normal using octahedral encoding (32-bit)
      splatNormalPacked = encodeNormalOctahedral(normalWorld);
#else
      splatNormal = normalWorld;
#endif

#if MESH_SHADER_PER_PRIMITIVE_ATTRIBS
      // NVIDIA: emit the normal per-primitive (integration reference)
#if QUANTIZE_NORMALS
      prims[localIndex * 2 + 0].outSplatNormalPacked = splatNormalPacked;
      prims[localIndex * 2 + 1].outSplatNormalPacked = splatNormalPacked;
#else
      prims[localIndex * 2 + 0].outSplatNormal = splatNormal;
      prims[localIndex * 2 + 1].outSplatNormal = splatNormal;
#endif
#endif
    }
#endif

    // fetch radiance from SH coefs > degree 0
    // const float3 worldViewDir = normalize(splatCenter - frameInfo.cameraPosition);
    // Row-major: vector * matrix (camera position transformed by inverse model matrix)
    const float3 worldViewDir =
        normalize(splatCenter - mul(float4(frameInfo.cameraPosition, 1.0), splatSetDesc.transformInverse).xyz);

    splatColor.rgb += fetchViewDependentRadiance(splatSetDesc, frameInfo.shDegree, splatIndex, worldViewDir);

#if MESH_SHADER_PER_PRIMITIVE_ATTRIBS && MS_ANTIALIASING == 0
    // emit per primitive color as early as possible for perf reasons,
    // only for original 3DGS, see later on for MipSplatting
    prims[localIndex * 2 + 0].outSplatCol = splatColor;
    prims[localIndex * 2 + 1].outSplatCol = splatColor;
#endif

    const float3x3 cov3Dm = fetchCovariance(splatSetDesc, splatIndex);

    // Computes the projected covariance
    const float3 cov2Dv = threedgsCovarianceProjection(cov3Dm, viewCenter, frameInfo.focal, modelViewMatrix);

    // computes the basis vectors of the extent of the projected covariance
    // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
    float2 basisVector1, basisVector2;
    if(!threedgsProjectedExtentBasis(cov2Dv, sqrt8, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
    {
      // Early return to discard splat
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

#if MESH_SHADER_PER_PRIMITIVE_ATTRIBS && MS_ANTIALIASING == 1
    // emit the color with alpha compensation after threedgsProjectedExtentBasis modified splatColor.a
    prims[localIndex * 2 + 0].outSplatCol = splatColor;
    prims[localIndex * 2 + 1].outSplatCol = splatColor;
#endif

    /////////////////////////////
    // emiting quad vertices

    const float3 ndcCenter = clipCenter.xyz / clipCenter.w;

    [unroll]
    for(uint i = 0; i < 4; ++i)
    {
      const float2 fragPos = positions[i].xy;

      const float2 ndcOffset = float2(fragPos.x * basisVector1 + fragPos.y * basisVector2) * frameInfo.basisViewport
                               * 2.0 * frameInfo.inverseFocalAdjustment;

      const float4 quadPos = float4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

      verts[localIndex * 4 + i].position = quadPos;
#if !USE_BARYCENTRIC
      verts[localIndex * 4 + i].outFragPos = fragPos.xy * sqrt8;
#endif
#if !MESH_SHADER_PER_PRIMITIVE_ATTRIBS
      // Non-NVIDIA: emit the flat attributes per-vertex (duplicated); see header / Slang #7019
      verts[localIndex * 4 + i].outSplatCol = splatColor;
#if STOCHASTIC_SPLAT || NEED_SURFACE_INFO
      verts[localIndex * 4 + i].outSplatId = globalSplatID;
#endif
#if NEED_SURFACE_INFO
#if QUANTIZE_NORMALS
      verts[localIndex * 4 + i].outSplatNormalPacked = splatNormalPacked;
#else
      verts[localIndex * 4 + i].outSplatNormal = splatNormal;
#endif
#endif
#endif
    }
  }
}
