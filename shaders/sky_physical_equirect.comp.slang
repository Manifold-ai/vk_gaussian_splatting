/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshaders/sky_functions.h.slang"

struct SkyEquirectPushConstants
{
  SkyPhysicalParameters skyParams;
};

// clang-format off
[[vk::push_constant]]  ConstantBuffer<SkyEquirectPushConstants> pushConst;
[[vk::binding(0)]]     RWTexture2D<float4>                      outImage;
// clang-format on

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
  uint2 imageSize;
  outImage.GetDimensions(imageSize.x, imageSize.y);

  if(tid.x >= imageSize.x || tid.y >= imageSize.y)
    return;

  float2 uv = (float2(tid.xy) + 0.5F) / float2(imageSize);

  // Equirectangular mapping: (u,v) -> (theta, phi) -> world direction
  // Azimuth convention must match directionToEquirectUV: u=0.5 → +Z, u=0 → -Z
  float theta = (uv.x - 0.5F) * M_TWO_PI;  // azimuth [-pi, pi]
  float phi   = (1.0F - uv.y) * M_PI;  // zenith   [0, pi] (top = 0)

  float sinPhi = sin(phi);
  float3 dir   = float3(sinPhi * sin(theta), cos(phi), sinPhi * cos(theta));

  float3 skyColor = evalPhysicalSky(pushConst.skyParams, dir);
  outImage[int2(tid.xy)] = float4(skyColor, 1.0F);
}
