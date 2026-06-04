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

#include <string>
#include <unordered_set>

#include <nvvk/context.hpp>

#include "shaderio.h"  // GPU_VENDOR_* macros (shared with shaders via the GPU_VENDOR macro)

namespace vk_gaussian_splatting {

// Runtime hardware capability registry.
//
// Populated once from main.cpp right after nvvk::Context::init() succeeds, by
// inspecting the final lists of enabled instance and device extensions. The
// resulting booleans let the rest of the application guard optional code paths
// with simple readable checks, e.g.:
//
//   if (isSupported.raytracing) { ... }
//   if (isSupported.DLSS)       { ... }
//
// DLSS has two independent signals: the Vulkan extensions (covered by the
// instance/device extension cache below) and the NGX runtime probe done by
// DlssDenoiser. The latter is reported via setDlssRuntimeAvailable() once it
// is known.
class HardwareSupport
{
public:
  // Cache the enabled extension names from the live Vulkan context and derive
  // the composite capability flags. Must be called once after a successful
  // call to nvvk::Context::init().
  void initFromVulkanContext(const nvvk::Context& ctx);

#if defined(USE_DLSS)
  // Bridge from DlssDenoiser once NGX has probed DLSS-RR availability.
  void setDlssRuntimeAvailable(bool available) { DLSS = available; }
#endif

  // Generic queries against the cached extension lists.
  bool hasInstanceExtension(const char* name) const;
  bool hasDeviceExtension(const char* name) const;

  // Composite/named capability flags. These mirror the optional extensions
  // requested in src/main.cpp; they are true iff the matching extension
  // survived nvvk::Context's filtering and was actually enabled on the device.
  bool raytracing = false;  // KHR ray tracing pipeline + acceleration structure + deferred host operations + shaderFloat64
  bool meshShader                   = false;  // VK_EXT_mesh_shader
  bool fragmentShadingRate          = false;  // VK_KHR_fragment_shading_rate
  bool fragmentShaderBarycentric    = false;  // VK_KHR_fragment_shader_barycentric
  bool fragmentShaderInterlock      = false;  // VK_EXT_fragment_shader_interlock
  bool rayTracingPositionFetch      = false;  // VK_KHR_ray_tracing_position_fetch
  bool rayTracingInvocationReorder  = false;  // VK_NV_ray_tracing_invocation_reorder
  bool rayTracingLinearSweptSpheres = false;  // VK_NV_ray_tracing_linear_swept_spheres
  bool shaderClock                  = false;  // VK_KHR_shader_clock
  bool memoryBudget                 = false;  // VK_EXT_memory_budget
  bool dynamicRendering             = false;  // VK_KHR_dynamic_rendering
  bool pushDescriptor               = false;  // VK_KHR_push_descriptor

  // Vulkan 1.0 core feature flags (queried directly from the physical device,
  // enabled via enableAllFeatures = true). Not part of the extension cache.
  bool shaderFloat64 = false;  // VkPhysicalDeviceFeatures::shaderFloat64

  // Vulkan 1.4 core feature flags (line rasterization features were promoted
  // from VK_KHR/EXT_line_rasterization to core in 1.4).
  bool smoothLines = false;  // VkPhysicalDeviceVulkan14Features::smoothLines

  // GPU vendor of the selected physical device, derived from VkPhysicalDeviceProperties::vendorID.
  // Holds one of the GPU_VENDOR_* values (shaderio.h). Used to gate vendor-specific code paths
  // (e.g. Intel iGPU workarounds) and exported to shaders as the GPU_VENDOR compile macro.
  uint32_t gpuVendor = GPU_VENDOR_OTHER;

  // Mesh-shader workgroup dispatch limits, from VkPhysicalDeviceMeshShaderPropertiesEXT
  // (defaulted to the Vulkan-guaranteed minima when mesh shaders are unsupported or report 0).
  // The splat raster mesh shaders are always dispatched as a 2D grid of width maxMeshWorkGroupCountX
  // with the workgroup id linearized in-shader, so scenes whose workgroup count exceeds the
  // per-dimension X limit simply wrap onto more Y rows (no vendor-specific path). maxMeshWorkGroupCountX
  // is exported to shaders as the MESH_SHADER_MAX_GROUPS_X macro. The Y and total limits are used to
  // detect scenes too large to dispatch at all (see GaussianSplatting::isMeshTaskDispatchOverflow).
  uint32_t maxMeshWorkGroupCountX     = 65535;    // maxMeshWorkGroupCount[0]
  uint32_t maxMeshWorkGroupCountY     = 65535;    // maxMeshWorkGroupCount[1]
  uint32_t maxMeshWorkGroupTotalCount = 4194304;  // maxMeshWorkGroupTotalCount (2^22)

  // Whether the mesh-shader raster path may emit per-primitive (PerPrimitiveEXT) attributes consumed
  // by the fragment shader. Only true on NVIDIA: Slang cannot emit the matching PerPrimitiveEXT
  // decoration on fragment inputs (Slang issue shader-slang/slang#7019), so on every other vendor the
  // mesh/frag shaders must fall back to per-vertex (duplicated, nointerpolation) attributes. Exported
  // to shaders as the MESH_SHADER_PER_PRIMITIVE_ATTRIBS compile macro.
  bool meshShaderPerPrimitiveAttribute = false;

#if defined(USE_DLSS)
  bool DLSS = false;  // NGX DLSS Ray Reconstruction usable at runtime
#else
  static constexpr bool DLSS = false;
#endif

private:
  std::unordered_set<std::string> m_instanceExtensions;
  std::unordered_set<std::string> m_deviceExtensions;
};

// Single global instance, populated from main.cpp.
extern HardwareSupport isSupported;

}  // namespace vk_gaussian_splatting
