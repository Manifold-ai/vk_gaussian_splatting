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

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(BINDING_ASSETS, 0)]] ConstantBuffer<SceneAssets> assets;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

struct VertexInput
{
  [[vk::location(0)]] float3 position : POSITION;
  [[vk::location(1)]] float3 normal : NORMAL;
  [[vk::location(2)]] float2 texCoord : TEXCOORD0;
  [[vk::location(3)]] float2 texCoord1 : TEXCOORD1;
  [[vk::location(4)]] float4 tangent : TANGENT;
};

struct VertexOutput
{
  [[vk::location(1)]] float3 worldPos : POSITION_WS;
  [[vk::location(2)]] float3 worldNrm : NORMAL_WS;
  [[vk::location(3)]] float3 viewDir : VIEW_DIR;
  [[vk::location(4)]] float2 texCoord : TEXCOORD0;
  [[vk::location(5)]] float2 texCoord1 : TEXCOORD1;
  [[vk::location(6)]] float4 worldTangent : TANGENT_WS;
  float4 position : SV_Position;
};
// clang-format on

[shader("vertex")]
VertexOutput main(VertexInput input)
{
  VertexOutput output;

  // Get mesh descriptor from bindless assets
  MeshDesc meshDesc = assets.meshes[pcRaster.objIndex];

  float3 origin = mul(float4(0, 0, 0, 1), frameInfo.viewInverse).xyz;

  output.worldPos  = mul(float4(input.position, 1.0), meshDesc.transform).xyz;
  output.viewDir   = output.worldPos - origin;
  output.texCoord  = input.texCoord;
  output.texCoord1 = input.texCoord1;

  // For normals, we need transpose(inverse(RS))
  // We store inverse(RS), then transpose() it here (fast operation)
  output.worldNrm = normalize(mul(input.normal, transpose(meshDesc.transformRotScaleInverse)));

  // Transform tangent direction to world space (same transform as normals for direction vectors)
  float3 worldT = normalize(mul(input.tangent.xyz, transpose(meshDesc.transformRotScaleInverse)));
  output.worldTangent = float4(worldT, input.tangent.w);

  output.position = mul(mul(float4(output.worldPos, 1.0), frameInfo.viewMatrix), frameInfo.projectionMatrixJittered);

  return output;
}
