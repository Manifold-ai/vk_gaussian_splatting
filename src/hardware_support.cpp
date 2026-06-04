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

#include "hardware_support.h"

#include <volk.h>

namespace vk_gaussian_splatting {

HardwareSupport isSupported{};

void HardwareSupport::initFromVulkanContext(const nvvk::Context& ctx)
{
  // Cache the names of all instance extensions that were passed to
  // vkCreateInstance. The instance list in ContextInitInfo is the one the
  // context actually used to create the instance, since validation only
  // filters at the device level.
  m_instanceExtensions.clear();
  for(const char* name : ctx.contextInfo.instanceExtensions)
  {
    if(name != nullptr)
    {
      m_instanceExtensions.emplace(name);
    }
  }

  // Cache the names of all device extensions that survived
  // nvvk::Context::filterAvailableExtensions and are therefore actually
  // enabled on the VkDevice.
  m_deviceExtensions.clear();
  for(const auto& ext : ctx.contextInfo.deviceExtensions)
  {
    if(ext.extensionName != nullptr)
    {
      m_deviceExtensions.emplace(ext.extensionName);
    }
  }

  // Cache the Vulkan 1.0 core feature flags we care about. enableAllFeatures
  // = true in main.cpp turns on every feature the physical device exposes, so
  // this reflects what is actually usable from shaders. We pull this directly
  // from the live context rather than re-querying vkGetPhysicalDeviceFeatures
  // because nvvk::Context may have masked some features.
  shaderFloat64 = (ctx.getPhysicalDeviceFeatures().shaderFloat64 == VK_TRUE);

  // Identify the GPU vendor from the physical device's PCI vendor ID. Used to gate
  // vendor-specific paths (Intel iGPU workarounds) on host and in shaders (GPU_VENDOR).
  {
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    vkGetPhysicalDeviceProperties(ctx.getPhysicalDevice(), &physicalDeviceProperties);
    switch(physicalDeviceProperties.vendorID)
    {
      case 0x10DE:  // NVIDIA
        gpuVendor = GPU_VENDOR_NVIDIA;
        break;
      case 0x1002:  // AMD (ATI)
      case 0x1022:  // AMD
        gpuVendor = GPU_VENDOR_AMD;
        break;
      case 0x8086:  // Intel
        gpuVendor = GPU_VENDOR_INTEL;
        break;
      default:
        gpuVendor = GPU_VENDOR_OTHER;
        break;
    }

    // Per-primitive mesh->fragment attributes only link correctly on NVIDIA (see Slang #7019 and the
    // field comment in hardware_support.h); every other vendor must use the per-vertex fallback.
    meshShaderPerPrimitiveAttribute = (gpuVendor == GPU_VENDOR_NVIDIA);
  }

  // Smooth line rasterization is a Vulkan 1.4 core feature (line rasterization was promoted from
  // VK_KHR/EXT_line_rasterization to core in 1.4), enabled via enableAllFeatures.
  smoothLines = (ctx.getPhysicalDeviceFeatures14().smoothLines == VK_TRUE);

  // Composite ray tracing flag: true only if the trio actually needed to
  // build acceleration structures and dispatch ray tracing pipelines is
  // present, AND shaderFloat64 is available. The RTX shaders (raygen,
  // intersection, anyhit, particle_as_build, etc.) use double precision for
  // transmittance accumulation and AABB intersection, and pipeline creation
  // crashes on devices without shaderFloat64 even when the KHR ray tracing
  // trio is otherwise reported supported (e.g. Intel Arc Pro). Optional RT
  // extras (position fetch, SER, LSS) are reported through their own flags
  // below.
  raytracing = hasDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
               && hasDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
               && hasDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) && shaderFloat64;

  meshShader                   = hasDeviceExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME);
  fragmentShadingRate          = hasDeviceExtension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
  fragmentShaderBarycentric    = hasDeviceExtension(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
  fragmentShaderInterlock      = hasDeviceExtension(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
  rayTracingPositionFetch      = hasDeviceExtension(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME);
  rayTracingInvocationReorder  = hasDeviceExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
  rayTracingLinearSweptSpheres = hasDeviceExtension(VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME);
  shaderClock                  = hasDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
  memoryBudget                 = hasDeviceExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  dynamicRendering             = hasDeviceExtension(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
  pushDescriptor               = hasDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  // Mesh-shader per-dimension workgroup limit. The splat raster path always dispatches a 2D grid of
  // width maxMeshWorkGroupCountX (linearized in-shader), so this caps the X extent; scenes larger
  // than the limit simply wrap onto additional Y rows. Defaults to the Vulkan-guaranteed minimum of
  // 65535 when mesh shaders are unsupported or the property reports 0.
  if(meshShader)
  {
    VkPhysicalDeviceMeshShaderPropertiesEXT meshProps{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &meshProps};
    vkGetPhysicalDeviceProperties2(ctx.getPhysicalDevice(), &props2);
    if(meshProps.maxMeshWorkGroupCount[0] > 0)
    {
      maxMeshWorkGroupCountX = meshProps.maxMeshWorkGroupCount[0];
    }
    if(meshProps.maxMeshWorkGroupCount[1] > 0)
    {
      maxMeshWorkGroupCountY = meshProps.maxMeshWorkGroupCount[1];
    }
    if(meshProps.maxMeshWorkGroupTotalCount > 0)
    {
      maxMeshWorkGroupTotalCount = meshProps.maxMeshWorkGroupTotalCount;
    }
  }
}

bool HardwareSupport::hasInstanceExtension(const char* name) const
{
  if(name == nullptr)
    return false;
  return m_instanceExtensions.find(name) != m_instanceExtensions.end();
}

bool HardwareSupport::hasDeviceExtension(const char* name) const
{
  if(name == nullptr)
    return false;
  return m_deviceExtensions.find(name) != m_deviceExtensions.end();
}

}  // namespace vk_gaussian_splatting
