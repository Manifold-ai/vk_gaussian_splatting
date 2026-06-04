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

#include "sky_sun_and_ibl.h"

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

void SkySunAndIBL::init(const Resources& res)
{
  m_res = res;

  m_skyParams.sunGlowIntensity = 2.5f;

  createSkyImage();
  createSkySampler();
  createSkyPipeline();

  // Initialize HdrIbl (does not load anything yet)
  m_hdrIbl.init(m_res.alloc, m_res.samplerPool);
}

void SkySunAndIBL::deinit()
{
  // Sky pipeline resources
  if(m_pipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(m_res.device, m_pipeline, nullptr);
  if(m_pipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(m_res.device, m_pipelineLayout, nullptr);
  if(m_descriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(m_res.device, m_descriptorPool, nullptr);
  if(m_descriptorSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(m_res.device, m_descriptorSetLayout, nullptr);
  if(m_shaderModule != VK_NULL_HANDLE)
    vkDestroyShaderModule(m_res.device, m_shaderModule, nullptr);
  if(m_skySampler != VK_NULL_HANDLE)
    vkDestroySampler(m_res.device, m_skySampler, nullptr);
  if(m_skyImage.image != VK_NULL_HANDLE)
    m_res.alloc->destroyImage(m_skyImage);

  m_pipeline            = VK_NULL_HANDLE;
  m_pipelineLayout      = VK_NULL_HANDLE;
  m_descriptorPool      = VK_NULL_HANDLE;
  m_descriptorSetLayout = VK_NULL_HANDLE;
  m_descriptorSet       = VK_NULL_HANDLE;
  m_shaderModule        = VK_NULL_HANDLE;
  m_skySampler          = VK_NULL_HANDLE;
  m_skyImageView        = VK_NULL_HANDLE;
  m_skyImage            = {};
  m_skyImageReady       = false;

  // HDR IBL resources
  m_hdrIbl.deinit();
}

void SkySunAndIBL::setMode(shaderio::EnvironmentMode mode)
{
  if(mode == m_mode)
    return;
  m_mode           = mode;
  m_dirty          = true;
  m_imageViewDirty = true;
}

void SkySunAndIBL::setEnabled(bool enabled)
{
  m_enabled = enabled;
}

void SkySunAndIBL::setResolution(glm::ivec2 res)
{
  res = glm::clamp(res, glm::ivec2(64, 32), glm::ivec2(8192, 4096));
  if(res == m_resolution)
    return;
  m_resolution  = res;
  m_needsResize = true;
  m_dirty       = true;
}

bool SkySunAndIBL::bake(VkCommandBuffer cmd)
{
  // Ensure sky image has a valid layout even when not in eSky mode
  if(!m_skyImageReady)
  {
    nvvk::cmdImageMemoryBarrier(cmd, {.image     = m_skyImage.image,
                                      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    m_skyImageReady = true;
  }

  if(m_mode != shaderio::EnvironmentMode::eSky || !m_dirty)
    return false;

  if(m_pipeline == VK_NULL_HANDLE)
    return false;

  if(m_needsResize)
  {
    vkDeviceWaitIdle(m_res.device);
    destroySkyImage();
    createSkyImage();
    updateSkyImageDescriptor();
    m_needsResize    = false;
    m_skyImageReady  = false;
    m_imageViewDirty = true;
  }

  NVVK_DBG_SCOPE(cmd);
  auto timerSection = m_res.profiler->cmdFrameSection(cmd, "SkyBake");

  nvvk::cmdImageMemoryBarrier(cmd, {.image     = m_skyImage.image,
                                    .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL});

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::SkyPhysicalParameters), &m_skyParams);

  uint32_t w = static_cast<uint32_t>(m_resolution.x);
  uint32_t h = static_cast<uint32_t>(m_resolution.y);
  vkCmdDispatch(cmd, (w + 15) / 16, (h + 15) / 16, 1);

  nvvk::cmdImageMemoryBarrier(cmd, {.image     = m_skyImage.image,
                                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});

  m_dirty         = false;
  m_skyImageReady = true;
  m_prevSkyParams = m_skyParams;

  return true;
}

bool SkySunAndIBL::loadHdrEnvironment(VkCommandBuffer cmd, const std::filesystem::path& hdrFile)
{
  m_hdrIbl.destroyEnvironment();
  m_hdrIbl.loadEnvironment(cmd, *m_res.uploader, hdrFile, false);
  m_res.uploader->cmdUploadAppended(cmd);

  if(!m_hdrIbl.isValid())
  {
    LOGE("SkySunAndIBL: failed to load HDR environment: %s\n", hdrFile.string().c_str());
    return false;
  }

  m_iblFilePath = hdrFile;

  VkExtent2D hdrSize = m_hdrIbl.getHdrImageSize();
  m_resolution       = glm::ivec2(hdrSize.width, hdrSize.height);

  m_mode           = shaderio::EnvironmentMode::eHDR;
  m_imageViewDirty = true;

  return true;
}

VkImageView SkySunAndIBL::getImageView()
{
  if(m_mode == shaderio::EnvironmentMode::eHDR && m_hdrIbl.isValid())
    return m_hdrIbl.getHdrImage().descriptor.imageView;
  return m_skyImageView;
}

VkSampler SkySunAndIBL::getSampler() const
{
  return m_skySampler;
}

VkImage SkySunAndIBL::getImage()
{
  if(m_mode == shaderio::EnvironmentMode::eHDR && m_hdrIbl.isValid())
    return m_hdrIbl.getHdrImage().image;
  return m_skyImage.image;
}

void SkySunAndIBL::createSkyImage()
{
  uint32_t w = static_cast<uint32_t>(m_resolution.x);
  uint32_t h = static_cast<uint32_t>(m_resolution.y);

  VkImageCreateInfo imageInfo{
      .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType   = VK_IMAGE_TYPE_2D,
      .format      = VK_FORMAT_R32G32B32A32_SFLOAT,
      .extent      = {w, h, 1},
      .mipLevels   = 1,
      .arrayLayers = 1,
      .samples     = VK_SAMPLE_COUNT_1_BIT,
      .tiling      = VK_IMAGE_TILING_OPTIMAL,
      .usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VkImageViewCreateInfo viewInfo{
      .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType         = VK_IMAGE_VIEW_TYPE_2D,
      .format           = VK_FORMAT_R32G32B32A32_SFLOAT,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };

  NVVK_CHECK(m_res.alloc->createImage(m_skyImage, imageInfo, viewInfo));
  m_skyImageView = m_skyImage.descriptor.imageView;
  NVVK_DBG_NAME(m_skyImage.image);
}

void SkySunAndIBL::destroySkyImage()
{
  if(m_skyImage.image != VK_NULL_HANDLE)
    m_res.alloc->destroyImage(m_skyImage);
  m_skyImageView = VK_NULL_HANDLE;
  m_skyImage     = {};
}

void SkySunAndIBL::updateSkyImageDescriptor()
{
  VkDescriptorImageInfo imageDescriptor{
      .imageView   = m_skyImageView,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
  };
  VkWriteDescriptorSet write{
      .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet          = m_descriptorSet,
      .dstBinding      = 0,
      .descriptorCount = 1,
      .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo      = &imageDescriptor,
  };
  vkUpdateDescriptorSets(m_res.device, 1, &write, 0, nullptr);
}

void SkySunAndIBL::createSkySampler()
{
  VkSamplerCreateInfo samplerInfo{
      .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter    = VK_FILTER_LINEAR,
      .minFilter    = VK_FILTER_LINEAR,
      .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  };
  NVVK_CHECK(vkCreateSampler(m_res.device, &samplerInfo, nullptr, &m_skySampler));
  NVVK_DBG_NAME(m_skySampler);
}

void SkySunAndIBL::createSkyPipeline()
{
  if(!m_res.slangCompiler->compileFile("sky_physical_equirect.comp.slang"))
  {
    LOGE("SkySunAndIBL: failed to compile sky_physical_equirect.comp.slang\n");
    return;
  }

  VkShaderModuleCreateInfo moduleInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = m_res.slangCompiler->getSpirvSize(),
      .pCode    = m_res.slangCompiler->getSpirv(),
  };
  NVVK_CHECK(vkCreateShaderModule(m_res.device, &moduleInfo, nullptr, &m_shaderModule));
  NVVK_DBG_NAME(m_shaderModule);

  VkDescriptorSetLayoutBinding layoutBinding{
      .binding         = 0,
      .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
  };

  VkDescriptorSetLayoutCreateInfo dslInfo{
      .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings    = &layoutBinding,
  };
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_res.device, &dslInfo, nullptr, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  VkPushConstantRange pushRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset     = 0,
      .size       = sizeof(shaderio::SkyPhysicalParameters),
  };

  VkPipelineLayoutCreateInfo plInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_res.device, &plInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  VkDescriptorPoolSize       poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
  VkDescriptorPoolCreateInfo poolInfo{
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = 1,
      .pPoolSizes    = &poolSize,
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_res.device, &poolInfo, nullptr, &m_descriptorPool));
  NVVK_DBG_NAME(m_descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo{
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_descriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_res.device, &allocInfo, &m_descriptorSet));
  NVVK_DBG_NAME(m_descriptorSet);

  VkDescriptorImageInfo imageDescriptor{
      .imageView   = m_skyImageView,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
  };
  VkWriteDescriptorSet write{
      .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet          = m_descriptorSet,
      .dstBinding      = 0,
      .descriptorCount = 1,
      .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo      = &imageDescriptor,
  };
  vkUpdateDescriptorSets(m_res.device, 1, &write, 0, nullptr);

  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = m_shaderModule,
              .pName  = "main",
          },
      .layout = m_pipelineLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(m_res.device, {}, 1, &pipelineInfo, nullptr, &m_pipeline));
  NVVK_DBG_NAME(m_pipeline);
}

}  // namespace vk_gaussian_splatting
