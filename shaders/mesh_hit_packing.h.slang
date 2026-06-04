/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MESH_HIT_PACKING_H_SLANG
#define MESH_HIT_PACKING_H_SLANG

#include "octahedral_normal.h.slang"

// Mesh hit data transferred between closest-hit and raygen shaders via the payload.
// Two layouts are supported:
//   Full precision:  15 float slots (original layout)
//   Quantized:       11 float slots (octahedral normal + fp16 tangent, fp32 UVs)
// Controlled by the RTX_QUANTIZE_MESH_PAYLOAD compile-time macro.

struct MeshHitData
{
  float  hitT;
  float3 worldPos;
  float3 worldNrm;
  float2 uv0;
  float3 tangentDir;
  float  tangentW;
  float2 uv1;
};

#if RTX_QUANTIZE_MESH_PAYLOAD
static const int MESH_PAYLOAD_SLOTS = 11;
#else
static const int MESH_PAYLOAD_SLOTS = 15;
#endif

// Pack two floats into a single uint using fp16 representation
uint packHalf2(float2 v)
{
  return f32tof16(v.x) | (f32tof16(v.y) << 16);
}

// Unpack a uint back into two floats from fp16 representation
float2 unpackHalf2(uint packed)
{
  return float2(f16tof32(packed & 0xFFFF), f16tof32(packed >> 16));
}

// Pack mesh hit data into payload arrays.
// dist[] must have at least MESH_PAYLOAD_SLOTS entries, id[] at least 2.
void packMeshHitToPayload(inout float dist[], inout int id[], in MeshHitData hit, int objId, int matId)
{
  id[0] = objId;
  id[1] = matId;

#if RTX_QUANTIZE_MESH_PAYLOAD
  // Quantized layout: 11 float slots
  // [0]     hitT              (fp32)
  // [1..3]  worldPos          (fp32)
  // [4]     normal            (octahedral uint, 16 bits per component)
  // [5..6]  uv0               (fp32)
  // [7]     tangentDir.xy     (2x fp16)
  // [8]     tangentDir.z+tanW (2x fp16)
  // [9..10] uv1               (fp32)
  dist[0]  = hit.hitT;
  dist[1]  = hit.worldPos.x;
  dist[2]  = hit.worldPos.y;
  dist[3]  = hit.worldPos.z;
  dist[4]  = asfloat(encodeNormalOctahedral(hit.worldNrm));
  dist[5]  = hit.uv0.x;
  dist[6]  = hit.uv0.y;
  dist[7]  = asfloat(packHalf2(hit.tangentDir.xy));
  dist[8]  = asfloat(packHalf2(float2(hit.tangentDir.z, hit.tangentW)));
  dist[9]  = hit.uv1.x;
  dist[10] = hit.uv1.y;
#else
  // Full precision layout: 15 float slots
  dist[0]  = hit.hitT;
  dist[1]  = hit.worldPos.x;
  dist[2]  = hit.worldPos.y;
  dist[3]  = hit.worldPos.z;
  dist[4]  = hit.worldNrm.x;
  dist[5]  = hit.worldNrm.y;
  dist[6]  = hit.worldNrm.z;
  dist[7]  = hit.uv0.x;
  dist[8]  = hit.uv0.y;
  dist[9]  = hit.tangentDir.x;
  dist[10] = hit.tangentDir.y;
  dist[11] = hit.tangentDir.z;
  dist[12] = hit.tangentW;
  dist[13] = hit.uv1.x;
  dist[14] = hit.uv1.y;
#endif
}

// Unpack mesh hit data from payload arrays.
MeshHitData unpackMeshHitFromPayload(float dist[], int id[], out int objId, out int matId)
{
  objId = id[0];
  matId = id[1];

  MeshHitData hit;

#if RTX_QUANTIZE_MESH_PAYLOAD
  hit.hitT       = dist[0];
  hit.worldPos   = float3(dist[1], dist[2], dist[3]);
  hit.worldNrm   = decodeNormalOctahedral(asuint(dist[4]));
  hit.uv0        = float2(dist[5], dist[6]);
  hit.tangentDir = float3(unpackHalf2(asuint(dist[7])), unpackHalf2(asuint(dist[8])).x);
  hit.tangentW   = unpackHalf2(asuint(dist[8])).y;
  hit.uv1        = float2(dist[9], dist[10]);
#else
  hit.hitT       = dist[0];
  hit.worldPos   = float3(dist[1], dist[2], dist[3]);
  hit.worldNrm   = float3(dist[4], dist[5], dist[6]);
  hit.uv0        = float2(dist[7], dist[8]);
  hit.tangentDir = float3(dist[9], dist[10], dist[11]);
  hit.tangentW   = dist[12];
  hit.uv1        = float2(dist[13], dist[14]);
#endif

  return hit;
}

#endif  // MESH_HIT_PACKING_H_SLANG
