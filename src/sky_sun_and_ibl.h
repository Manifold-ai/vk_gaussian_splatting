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

#pragma once

#include <filesystem>
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvslang/slang.hpp>
#include <nvshaders/sky_io.h.slang>

#include "shaderio.h"

namespace vk_gaussian_splatting {

//--------------------------------------------------------------------------------------------------
// SkySunAndIBL: Manages environment lighting — procedural physical sky or HDR IBL image.
//
// In eSky mode, owns a compute pipeline that bakes the physical sky model into an equirect image.
// In eHDR mode, loads an HDR file via nvvk::HdrIbl and exposes its texture + importance sampling
// acceleration structure. Both modes share the same equirect texture binding point.
//--------------------------------------------------------------------------------------------------
class SkySunAndIBL
{
public:
  struct Resources
  {
    VkDevice                 device;
    nvvk::ResourceAllocator* alloc;
    nvvk::SamplerPool*       samplerPool;
    nvvk::StagingUploader*   uploader;
    nvslang::SlangCompiler*  slangCompiler;
    nvvk::ProfilerGpuTimer*  profiler;
  };

  void init(const Resources& res);
  void deinit();

  // Re-bake equirect texture if dirty and in eSky mode. Returns true if a bake occurred.
  bool bake(VkCommandBuffer cmd);

  // Load an HDR environment map and switch to eHDR mode
  bool loadHdrEnvironment(VkCommandBuffer cmd, const std::filesystem::path& hdrFile);

  // Environment mode
  shaderio::EnvironmentMode mode() const { return m_mode; }
  void                      setMode(shaderio::EnvironmentMode mode);

  // Lighting contribution toggle (independent of mode)
  bool isEnabled() const { return m_enabled; }
  void setEnabled(bool enabled);

  bool isDirty() const { return m_dirty; }
  void setDirty() { m_dirty = true; }

  // Sky parameters (eSky mode)
  shaderio::SkyPhysicalParameters&       skyParams() { return m_skyParams; }
  const shaderio::SkyPhysicalParameters& skyParams() const { return m_skyParams; }

  // IBL parameters (eHDR mode)
  float                        iblIntensity() const { return m_iblIntensity; }
  void                         setIblIntensity(float v) { m_iblIntensity = v; }
  const glm::vec3&             iblRotation() const { return m_iblRotation; }
  void                         setIblRotation(const glm::vec3& r) { m_iblRotation = r; }
  const std::filesystem::path& iblFilePath() const { return m_iblFilePath; }
  float                        iblIntegral() const { return m_hdrIbl.isValid() ? m_hdrIbl.getIntegral() : 1.0f; }
  bool                         isHdrLoaded() const { return m_hdrIbl.isValid(); }

  // Resolution of the equirectangular texture (editable for eSky, read-only for eHDR)
  const glm::ivec2& resolution() const { return m_resolution; }
  void              setResolution(glm::ivec2 res);

  // True after setResolution() or mode change recreated the image; caller must update external descriptors.
  bool imageViewChanged() const { return m_imageViewDirty; }
  void clearImageViewChanged() { m_imageViewDirty = false; }

  // Accessors for descriptor set binding in the RT pipeline
  VkImageView getImageView();
  VkSampler   getSampler() const;
  VkImage     getImage();

  // HDR importance sampling acceleration buffer (valid only in eHDR mode)
  nvvk::Buffer getEnvAccelBuffer() const { return m_hdrIbl.getEnvAccel(); }
  bool         hasEnvAccel() const { return m_hdrIbl.isValid(); }

private:
  // Sky bake infrastructure
  void createSkyImage();
  void destroySkyImage();
  void updateSkyImageDescriptor();
  void createSkySampler();
  void createSkyPipeline();

  Resources m_res{};

  // Environment state
  shaderio::EnvironmentMode m_mode    = shaderio::EnvironmentMode::eNone;
  bool                      m_enabled = true;  // Lighting contribution
  bool                      m_dirty   = true;

  // Sky parameters (eSky mode)
  bool                            m_skyImageReady = false;
  shaderio::SkyPhysicalParameters m_skyParams{};
  shaderio::SkyPhysicalParameters m_prevSkyParams{};

  // IBL parameters (eHDR mode)
  float                 m_iblIntensity = 1.0f;
  glm::vec3             m_iblRotation{0.0f};
  std::filesystem::path m_iblFilePath;

  // Equirect resolution (user-adjustable for eSky)
  glm::ivec2 m_resolution{2048, 1024};
  bool       m_needsResize    = false;
  bool       m_imageViewDirty = false;

  // Equirectangular sky texture (eSky mode)
  nvvk::Image m_skyImage{};
  VkImageView m_skyImageView = VK_NULL_HANDLE;
  VkSampler   m_skySampler   = VK_NULL_HANDLE;

  // HDR IBL (eHDR mode)
  nvvk::HdrIbl m_hdrIbl;

  // Compute pipeline for equirect sky bake
  VkShaderModule        m_shaderModule        = VK_NULL_HANDLE;
  VkPipeline            m_pipeline            = VK_NULL_HANDLE;
  VkPipelineLayout      m_pipelineLayout      = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool      m_descriptorPool      = VK_NULL_HANDLE;
  VkDescriptorSet       m_descriptorSet       = VK_NULL_HANDLE;
};

}  // namespace vk_gaussian_splatting
