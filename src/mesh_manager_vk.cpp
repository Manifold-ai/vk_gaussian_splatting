/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mesh_manager_vk.h"
#include "gltf_loader.h"
#include "obj_loader.h"
#include "utilities.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <algorithm>  // for std::find
#include <fmt/format.h>

#include <nvvk/debug_util.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

namespace vk_gaussian_splatting {

// =============================================================================
// MeshVk Buffer Management
// =============================================================================

void MeshVk::initBuffers(nvvk::ResourceAllocator* alloc, nvvk::StagingUploader* uploader)
{
  gpuVertexCount = nbVertices();
  gpuIndexCount  = nbIndices();

  // Allocate GPU buffers
  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  NVVK_CHECK(alloc->createBuffer(vertexBuffer, vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(vertexBuffer.buffer);

  NVVK_CHECK(alloc->createBuffer(indexBuffer, indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(indexBuffer.buffer);

  NVVK_CHECK(alloc->createBuffer(materialsBuffer, materials.size() * sizeof(Material),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(materialsBuffer.buffer);

  NVVK_CHECK(alloc->createBuffer(matIndexBuffer, matIndices.size() * sizeof(uint32_t),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(matIndexBuffer.buffer);

  // Update needShading flags before uploading materials to GPU
  for(auto& mat : materials)
  {
    shaderio::updateMaterialNeedsShading(mat);
  }

  // Upload data to GPU
  NVVK_CHECK(uploader->appendBuffer(vertexBuffer, 0, std::span(vertices)));
  NVVK_CHECK(uploader->appendBuffer(indexBuffer, 0, std::span(indices)));
  NVVK_CHECK(uploader->appendBuffer(materialsBuffer, 0, std::span(materials)));
  NVVK_CHECK(uploader->appendBuffer(matIndexBuffer, 0, std::span(matIndices)));

  // Note: uploader->cmdUploadAppended() and staging release handled by caller
}

void MeshVk::deinitBuffers(nvvk::ResourceAllocator* alloc)
{
  // Destroy all buffers
  alloc->destroyBuffer(vertexBuffer);
  alloc->destroyBuffer(indexBuffer);
  alloc->destroyBuffer(materialsBuffer);
  alloc->destroyBuffer(matIndexBuffer);

  // Reset to empty state
  vertexBuffer    = {};
  indexBuffer     = {};
  materialsBuffer = {};
  matIndexBuffer  = {};
}

// =============================================================================
// MeshManagerVk Implementation
// =============================================================================

// High-level API: Load from file
std::shared_ptr<MeshVk> MeshManagerVk::loadModel(const std::filesystem::path& filename)
{
  LOGI("Loading File:  %s \n", filename.string().c_str());

  auto meshVk = std::make_shared<MeshVk>();

  const std::string ext    = filename.extension().string();
  bool              loaded = false;
  if(ext == ".gltf" || ext == ".glb")
  {
    GltfLoader loader;
    loaded = loader.load(filename, *meshVk);
  }
  else
  {
    ObjLoader loader;
    loaded = loader.load(filename, *meshVk);
  }

  if(!loaded)
    return nullptr;

  // OBJ stores color factors in sRGB space — convert to linear for shading.
  // glTF factors are already linear per spec, so skip the conversion.
  bool isOBJ = (ext == ".obj");
  if(isOBJ)
  {
    for(auto& m : meshVk->materials)
    {
      m.baseColor = glm::pow(m.baseColor, glm::vec3(2.2f));
      m.emissive  = glm::pow(m.emissive, glm::vec3(2.2f));
    }
  }

  // Load textures referenced by materials and remap local indices to global
  if(!meshVk->textures.empty())
  {
    const auto basePath = filename.parent_path();

    // Build local-to-global index map
    std::vector<int> localToGlobal(meshVk->textures.size(), -1);
    for(size_t i = 0; i < meshVk->textures.size(); i++)
    {
      // Pass embedded image bytes if available (GLB embedded textures)
      const auto& embedded = (i < meshVk->embeddedImages.size()) ? meshVk->embeddedImages[i] : std::vector<uint8_t>{};
      bool        sRGB     = (i < meshVk->textureSRGB.size()) ? meshVk->textureSRGB[i] : false;
      auto        tex      = loadTexture(meshVk->textures[i], basePath, sRGB, embedded);
      if(tex)
      {
        localToGlobal[i] = static_cast<int>(tex->globalIndex);
        meshVk->loadedTextures.push_back(tex);
      }
    }

    // Free embedded image data — no longer needed after GPU upload
    meshVk->embeddedImages.clear();
    meshVk->embeddedImages.shrink_to_fit();

    // Remap all material texture IDs from loader-local to global descriptor index
    auto remap = [&](int& id) {
      if(id >= 0 && id < static_cast<int>(localToGlobal.size()))
        id = localToGlobal[id];
      else
        id = -1;
    };
    for(auto& mat : meshVk->materials)
    {
      remap(mat.baseColorTexture);
      remap(mat.metallicRoughnessTexture);
      remap(mat.normalTexture);
      remap(mat.emissiveTexture);
      remap(mat.occlusionTexture);
      remap(mat.specularTexture);
      remap(mat.specularColorTexture);
      remap(mat.clearcoatTexture);
      remap(mat.clearcoatRoughnessTexture);
      remap(mat.pbrSgDiffuseTexture);
      remap(mat.pbrSgSpecularGlossinessTexture);
    }
  }

  meshVk->path = filename.string();
  registerMesh(meshVk);

  // Create default instance at origin
  m_lastCreatedInstance = createInstance(meshVk);

  return meshVk;
}

// ---------------------------------------------------------------------------
// Texture loading with deduplication
// ---------------------------------------------------------------------------

std::shared_ptr<MeshTexture> MeshManagerVk::loadTexture(const std::filesystem::path& texturePath,
                                                        const std::filesystem::path& basePath,
                                                        bool                         sRGB,
                                                        const std::vector<uint8_t>&  embeddedData)
{
  // Build deduplication key (includes sRGB flag — same image as color vs data needs separate GPU resources)
  std::string key;
  if(texturePath.string().find("<embedded>:") == 0)
  {
    key = basePath.string() + "/" + texturePath.string();
  }
  else
  {
    auto            resolved = basePath / texturePath;
    std::error_code ec;
    auto            canonical = std::filesystem::weakly_canonical(resolved, ec);
    key                       = ec ? resolved.string() : canonical.string();
  }
  if(sRGB)
    key += ":srgb";

  // Check cache
  auto it = m_textureCache.find(key);
  if(it != m_textureCache.end())
  {
    if(auto existing = it->second.lock())
      return existing;
    m_textureCache.erase(it);  // expired
  }

  // Load pixel data
  int            width = 0, height = 0, channels = 0;
  unsigned char* pixels = nullptr;

  if(!embeddedData.empty())
  {
    pixels = stbi_load_from_memory(embeddedData.data(), static_cast<int>(embeddedData.size()), &width, &height, &channels, 4);
  }
  else if(texturePath.string().find("<embedded>:") != 0)
  {
    auto resolved = basePath / texturePath;
    pixels        = stbi_load(resolved.string().c_str(), &width, &height, &channels, 4);
  }

  if(!pixels)
  {
    LOGW("Failed to load texture: %s", key.c_str());
    return nullptr;
  }

  // Create VkImage and upload
  auto meshTex  = std::make_shared<MeshTexture>();
  meshTex->path = key;

  VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imgInfo.imageType   = VK_IMAGE_TYPE_2D;
  imgInfo.format      = sRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
  imgInfo.extent      = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  imgInfo.mipLevels   = 1;
  imgInfo.arrayLayers = 1;
  imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
  imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
  imgInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  NVVK_CHECK(m_alloc->createImage(meshTex->image, imgInfo, DEFAULT_VkImageViewCreateInfo));

  // Upload pixel data
  size_t          dataSize = static_cast<size_t>(width) * height * 4;
  VkCommandBuffer cmd      = m_app->createTempCmdBuffer();
  NVVK_CHECK(m_uploader->appendImage(meshTex->image, std::span<uint8_t>(pixels, dataSize), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
  m_uploader->cmdUploadAppended(cmd);
  m_app->submitAndWaitTempCmdBuffer(cmd);
  m_uploader->releaseStaging();

  stbi_image_free(pixels);

  // Attach sampler to descriptor
  meshTex->image.descriptor.sampler = *m_sampler;

  // Assign global index and store
  meshTex->globalIndex = static_cast<uint32_t>(m_meshTextures.size());
  m_meshTextures.push_back(meshTex);
  m_textureCache[key] = meshTex;
  m_meshTexturesDirty = true;

  LOGI("Loaded mesh texture [%u]: %s (%dx%d)\n", meshTex->globalIndex, texturePath.string().c_str(), width, height);

  return meshTex;
}

void MeshManagerVk::pruneExpiredTextures()
{
  // Remove expired weak_ptrs from cache
  for(auto it = m_textureCache.begin(); it != m_textureCache.end();)
  {
    if(it->second.expired())
      it = m_textureCache.erase(it);
    else
      ++it;
  }

  // Build old → new index mapping while compacting the texture array
  std::unordered_map<uint32_t, uint32_t>    oldToNew;
  std::vector<std::shared_ptr<MeshTexture>> surviving;

  for(auto& tex : m_meshTextures)
  {
    if(tex.use_count() > 1)  // >1 means a mesh still holds a reference
    {
      uint32_t oldIdx  = tex->globalIndex;
      tex->globalIndex = static_cast<uint32_t>(surviving.size());
      oldToNew[oldIdx] = tex->globalIndex;
      surviving.push_back(tex);
    }
    else
    {
      m_alloc->destroyImage(tex->image);
    }
  }

  if(surviving.size() == m_meshTextures.size())
    return;  // nothing changed

  m_meshTextures      = std::move(surviving);
  m_meshTexturesDirty = true;

  // Remap material texture IDs in all live meshes
  auto remap = [&](int& id) {
    if(id < 0)
      return;
    auto it = oldToNew.find(static_cast<uint32_t>(id));
    id      = (it != oldToNew.end()) ? static_cast<int>(it->second) : -1;
  };

  for(auto& mesh : meshes)
  {
    if(!mesh)
      continue;

    // Also prune the mesh's own loadedTextures (remove expired shared_ptrs)
    mesh->loadedTextures.erase(std::remove_if(mesh->loadedTextures.begin(), mesh->loadedTextures.end(),
                                              [](const auto& t) { return !t; }),
                               mesh->loadedTextures.end());

    for(auto& mat : mesh->materials)
    {
      remap(mat.baseColorTexture);
      remap(mat.metallicRoughnessTexture);
      remap(mat.normalTexture);
      remap(mat.emissiveTexture);
      remap(mat.occlusionTexture);
      remap(mat.specularTexture);
      remap(mat.specularColorTexture);
      remap(mat.clearcoatTexture);
      remap(mat.clearcoatRoughnessTexture);
      remap(mat.pbrSgDiffuseTexture);
      remap(mat.pbrSgSpecularGlossinessTexture);
    }
    mesh->flags |= MeshVk::Flags::eMaterialsChanged;
  }
  pendingRequests |= Request::eUpdateMaterials;
}

std::shared_ptr<MeshVk> MeshManagerVk::registerMesh(std::shared_ptr<MeshVk> meshVk)
{
  // Generate default material names if not provided
  if(meshVk->matNames.empty())
  {
    meshVk->matNames.resize(meshVk->materials.size());
    for(size_t i = 0; i < meshVk->materials.size(); ++i)
      meshVk->matNames[i] = "material_" + std::to_string(i);
  }

  // Set flag: needs GPU upload (buffers NOT allocated yet)
  meshVk->flags |= MeshVk::Flags::eNew;

  // Set index and store mesh
  meshVk->index = meshes.size();
  meshes.push_back(meshVk);

  // Request GPU sync (deferred to processVramUpdates)
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildBLAS;

  LOGD("registerMesh: Registered mesh '%s' (meshes.size=%zu)\n", meshVk->path.c_str(), meshes.size());

  return meshVk;
}

// Create instance of existing mesh (returns shared_ptr to instance)
std::shared_ptr<MeshInstanceVk> MeshManagerVk::createInstance(std::shared_ptr<MeshVk> mesh, const glm::mat4& transform, MeshType type)
{
  if(!mesh)
  {
    LOGE("createInstance: Null mesh pointer\n");
    return nullptr;
  }

  auto instance                      = std::make_shared<MeshInstanceVk>();
  instance->mesh                     = mesh;  // Store mesh shared_ptr
  instance->type                     = type;
  instance->transform                = transform;
  instance->transformInverse         = glm::inverse(transform);
  instance->transformRotScaleInverse = glm::inverse(glm::mat3(transform));  // Extract and invert rotation-scale part

  // Generate display name (only for user objects, not light proxies)
  if(type == MeshType::eObject)
  {
    std::filesystem::path filepath(mesh->path);
    std::string           filename = filepath.filename().string();

    // Handle empty path (e.g., procedural meshes)
    if(filename.empty() && !mesh->path.empty())
      filename = mesh->path;  // Use whatever path string we have
    else if(filename.empty())
      filename = "Procedural";

    instance->name = fmt::format("Model {} - {}", m_nextInstanceNumber, truncateFilename(filename));
    ++m_nextInstanceNumber;
  }
  else
  {
    // Internal mesh types (light proxies, etc.) don't get numbered
    instance->name = mesh->path.empty() ? "Internal" : mesh->path;
  }

  // Set flag: needs descriptor entry
  instance->flags |= MeshInstanceVk::Flags::eNew;

  // Set index and store instance
  instance->index = instances.size();
  instances.push_back(instance);     // Add to vector
  m_lastCreatedInstance = instance;  // Store for UI convenience

  // Request GPU sync (deferred to processVramUpdates)
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildTLAS;

  LOGD("createInstance: Created instance '%s' (instances.size=%zu)\n", instance->name.c_str(), instances.size());

  return instance;
}

std::shared_ptr<MeshInstanceVk> MeshManagerVk::registerInstance(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
  {
    LOGE("registerInstance: Null instance pointer\n");
    return nullptr;
  }

  // Set index and add to vector
  instance->index = instances.size();

  // Mark as new (needs descriptor entry)
  instance->flags |= MeshInstanceVk::Flags::eNew;

  instances.push_back(instance);
  m_lastCreatedInstance = instance;

  // Request GPU sync (deferred to processVramUpdates)
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildTLAS;

  LOGD("registerInstance: Registered instance '%s' (instances.size=%zu)\n", instance->name.c_str(), instances.size());

  return instance;
}

std::shared_ptr<MeshInstanceVk> MeshManagerVk::duplicateInstance(std::shared_ptr<MeshInstanceVk> sourceInstance)
{
  if(!sourceInstance || !sourceInstance->mesh)
  {
    LOGE("duplicateInstance: Invalid source instance\n");
    return nullptr;
  }

  // Create new instance as a copy of the source (copy all fields via copy constructor)
  auto newInstance = std::make_shared<MeshInstanceVk>(*sourceInstance);

  // Generate NEW display name (don't copy source name) - only for user objects
  if(sourceInstance->type == MeshType::eObject)
  {
    std::filesystem::path filepath(sourceInstance->mesh->path);
    std::string           filename = filepath.filename().string();
    if(filename.empty())
      filename = sourceInstance->name;  // Fallback to source name

    newInstance->name = fmt::format("Model {} - {}", m_nextInstanceNumber, truncateFilename(filename));
    ++m_nextInstanceNumber;
  }
  // else: Keep copied name for internal instances

  // Register the new instance (this will set index and add to vector)
  // Note: registerInstance may reallocate vector, so sourceInstance reference could be invalidated after this
  newInstance = registerInstance(newInstance);

  if(!newInstance)
  {
    LOGE("duplicateInstance: Failed to register new instance\n");
    return nullptr;
  }

  LOGD("duplicateInstance: Duplicated instance (source='%s' -> new='%s')\n", sourceInstance->name.c_str(),
       newInstance->name.c_str());

  return newInstance;
}

void MeshManagerVk::updateMeshDescriptionBuffer()
{
  // Rebuild objectDescriptions from instances set
  objectDescriptions.clear();

  for(const auto& instance : instances)
  {
    if(!instance || !instance->mesh)
      continue;  // Skip invalid instances

    shaderio::MeshDesc desc{};

    // Geometry addresses (from mesh)
    desc.vertexAddress        = (shaderio::Vertex*)instance->mesh->vertexBuffer.address;
    desc.indexAddress         = (uint32_t*)instance->mesh->indexBuffer.address;
    desc.materialAddress      = (shaderio::Material*)instance->mesh->materialsBuffer.address;
    desc.materialIndexAddress = (uint32_t*)instance->mesh->matIndexBuffer.address;

    // Visibility
    desc.show = instance->show ? 1 : 0;

    // Instance transform
    desc.transform                = instance->transform;
    desc.transformInverse         = instance->transformInverse;
    desc.transformRotScaleInverse = instance->transformRotScaleInverse;
    desc.prevTransform            = instance->prevTransform;

    objectDescriptions.push_back(desc);
  }

  // Save old buffer to destroy after new one is ready
  nvvk::Buffer oldBuffer   = objectDescriptionsBuffer;
  objectDescriptionsBuffer = {};

  if(objectDescriptions.empty())
  {
    // Destroy old buffer if we're going to empty state
    if(oldBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(oldBuffer);
    return;
  }

  // Create buffer
  NVVK_CHECK(m_alloc->createBuffer(objectDescriptionsBuffer, objectDescriptions.size() * sizeof(shaderio::MeshDesc),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
  NVVK_DBG_NAME(objectDescriptionsBuffer.buffer);

  // Upload buffer
  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(objectDescriptionsBuffer, 0, std::span(objectDescriptions)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();

  // Destroy old buffer AFTER new one is submitted
  if(oldBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(oldBuffer);
}

void MeshManagerVk::uploadMaterialsBufferInternal(std::shared_ptr<MeshVk> mesh)
{
  if(!mesh)
    return;

  // Note: This is an internal method called by processVramUpdates()
  // Assumes materialsBuffer already exists (created by initBuffers)

  // Update needShading flags before uploading materials to GPU
  for(auto& mat : mesh->materials)
  {
    shaderio::updateMaterialNeedsShading(mat);
  }

  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(mesh->materialsBuffer, 0, std::span(mesh->materials)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();
}

nvvk::AccelerationStructureGeometryInfo MeshManagerVk::rtxCreateMeshVkKHR(const MeshVk& model)
{
  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
  triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = model.vertexBuffer.address;
  triangles.vertexStride             = sizeof(Vertex);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = model.indexBuffer.address;
  triangles.transformData            = {};  // Identity
  triangles.maxVertex                = model.gpuVertexCount - 1;

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR geometry{};
  geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
  geometry.geometry.triangles = triangles;

  // The entire array will be used to build the BLAS.
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
  rangeInfo.firstVertex     = 0;
  rangeInfo.primitiveCount  = model.gpuIndexCount / 3;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.transformOffset = 0;

  return nvvk::AccelerationStructureGeometryInfo{.geometry = geometry, .rangeInfo = rangeInfo};
}

void MeshManagerVk::rtxInitAccelerationStructures()
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  // Mesh BLAS - each obj mesh is stored in a BLAS
  if(!meshes.empty())
  {
    std::vector<nvvk::AccelerationStructureGeometryInfo> asGeoInfo;
    asGeoInfo.reserve(meshes.size());

    for(const auto& mesh : meshes)
    {
      asGeoInfo.emplace_back(rtxCreateMeshVkKHR(*mesh));
    }
    // build the blas set
    NVVK_CHECK(rtAccelerationStructures.blasSubmitBuildAndWait(
        asGeoInfo, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR
                       | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR));

    // Statistics
    LOGD("%s%s\n", nvutils::ScopedTimer::indent().c_str(), rtAccelerationStructures.blasBuildStatistics.toString().c_str());
  }

  // Mesh TLAS - one entry/node per instance
  if(!instances.empty())
  {
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(instances.size());
    uint32_t descriptorIndex = 0;  // Track descriptor array index

    for(const auto& instance : instances)
    {
      if(!instance || !instance->mesh)
        continue;

      size_t meshIndex = instance->mesh->index;
      if(meshIndex >= rtAccelerationStructures.blasSet.size())
        continue;

      VkAccelerationStructureInstanceKHR asInst{};
      asInst.transform           = nvvk::toTransformMatrixKHR(instance->transform);  // Position of the instance
      asInst.instanceCustomIndex = descriptorIndex;                                  // Index in descriptor array
      asInst.accelerationStructureReference = rtAccelerationStructures.blasSet[meshIndex].address;
      asInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      // Use instance type for mask (MeshType enum values match RTX masks)
      asInst.mask                                   = static_cast<uint32_t>(instance->type);
      asInst.instanceShaderBindingTableRecordOffset = meshSbtRecordOffset;
      tlasInstances.emplace_back(asInst);
      descriptorIndex++;
    }
    // then build the dynamic TLAS, add allow update flag so we can update mesh matrices and use tlasUpade
    NVVK_CHECK(rtAccelerationStructures.tlasSubmitBuildAndWait(tlasInstances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                                  | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
  }

  // [MESH-BLACK-DIAG] confirm the mesh BLAS/TLAS were built — hybrid RTX mesh shading depends on this.
  LOGW("[MESH-BLACK-DIAG] rtxInitAccelerationStructures: meshes=%zu instances=%zu blasSet=%zu tlas=%s\n", meshes.size(),
       instances.size(), rtAccelerationStructures.blasSet.size(),
       rtAccelerationStructures.tlas.accel ? "built" : "NULL");
}

void MeshManagerVk::rtxUpdateTopLevelAccelerationStructure()
{
  // Prepare TLAS for Incrusted meshes, different from the splat one.
  if(!instances.empty())
  {
    // TODO could be a class member to prevent reallocation
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(instances.size());
    uint32_t descriptorIndex = 0;  // Track descriptor array index

    for(const auto& instance : instances)
    {
      if(!instance || !instance->mesh)
        continue;

      size_t meshIndex = instance->mesh->index;
      if(meshIndex >= rtAccelerationStructures.blasSet.size())
        continue;

      VkAccelerationStructureInstanceKHR asInst{};
      asInst.transform           = nvvk::toTransformMatrixKHR(instance->transform);  // Position of the instance
      asInst.instanceCustomIndex = descriptorIndex;                                  // Index in descriptor array
      asInst.accelerationStructureReference = rtAccelerationStructures.blasSet[meshIndex].address;
      asInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      // Use instance type for mask (MeshType enum values match RTX masks)
      asInst.mask                                   = static_cast<uint32_t>(instance->type);
      asInst.instanceShaderBindingTableRecordOffset = meshSbtRecordOffset;
      tlasInstances.emplace_back(asInst);
      descriptorIndex++;
    }

    // Check if instance count changed - if so, rebuild TLAS from scratch
    if(tlasInstances.size() != rtAccelerationStructures.tlasSize)
    {
      LOGD("Instance count changed (%zu -> %zu), rebuilding TLAS\n", rtAccelerationStructures.tlasSize, tlasInstances.size());

      // Wait for GPU to finish using old TLAS resources before destroying them
      vkDeviceWaitIdle(m_app->getDevice());

      // Manually destroy old TLAS resources (but keep BLAS)
      if(rtAccelerationStructures.tlas.accel)
        m_alloc->destroyAcceleration(rtAccelerationStructures.tlas);
      if(rtAccelerationStructures.tlasInstancesBuffer.buffer)
        m_alloc->destroyLargeBuffer(rtAccelerationStructures.tlasInstancesBuffer);
      if(rtAccelerationStructures.tlasScratchBuffer.buffer)
        m_alloc->destroyLargeBuffer(rtAccelerationStructures.tlasScratchBuffer);

      rtAccelerationStructures.tlas                = {};
      rtAccelerationStructures.tlasInstancesBuffer = {};
      rtAccelerationStructures.tlasScratchBuffer   = {};
      rtAccelerationStructures.tlasBuildData       = {};
      rtAccelerationStructures.tlasSize            = 0;

      // Rebuild with new instance count
      NVVK_CHECK(rtAccelerationStructures.tlasSubmitBuildAndWait(tlasInstances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                                    | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
    }
    else
    {
      // Just update existing TLAS (transforms only)
      rtAccelerationStructures.tlasSubmitUpdateAndWait(tlasInstances);
    }
  }
}

void MeshManagerVk::deleteInstance(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
  {
    LOGE("deleteInstance: Null instance pointer\n");
    return;
  }

  // Set Delete flag (deferred deletion in processVramUpdates)
  instance->flags |= MeshInstanceVk::Flags::eDelete;
  pendingRequests |= Request::eProcessDeletions;

  LOGD("deleteInstance: Marked instance for deletion ('%s')\n", instance->name.c_str());
}

void MeshManagerVk::deleteInstanceOnly(std::shared_ptr<MeshInstanceVk> instance)
{
  // Delete instance without destroying the mesh (used for light proxies where meshes are persistent)
  // Note: Now uses deferred mechanism like deleteInstance() - processVramUpdates handles both
  if(!instance)
  {
    LOGE("deleteInstanceOnly: Null instance pointer\n");
    return;
  }

  // Set Delete flag (deferred deletion in processVramUpdates)
  instance->flags |= MeshInstanceVk::Flags::eDelete;
  pendingRequests |= Request::eProcessDeletions;

  LOGD("deleteInstanceOnly: Marked instance for deletion ('%s')\n", instance->name.c_str());
}

// =============================================================================
// DEFERRED UPDATE API - Methods that set flags and requests
// =============================================================================

void MeshManagerVk::deleteMesh(std::shared_ptr<MeshVk> mesh)
{
  if(!mesh)
    return;

  mesh->flags |= MeshVk::Flags::eDelete;
  pendingRequests |= Request::eProcessDeletions;
}

void MeshManagerVk::updateInstanceTransform(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
    return;

  // Caller has already modified instance->transform in RAM
  // Just set flag and request GPU update
  instance->transformDirtyCount = 2;  // Two descriptor uploads needed for DLSS motion vectors
  instance->flags |= MeshInstanceVk::Flags::eTransformChanged;
  pendingRequests |= Request::eUpdateTransformsOnly;
}

void MeshManagerVk::updateInstanceMaterial(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
    return;

  instance->flags |= MeshInstanceVk::Flags::eMaterialChanged;
  pendingRequests |= Request::eUpdateDescriptors;  // May need descriptor rebuild
}

void MeshManagerVk::updateMeshMaterials(std::shared_ptr<MeshVk> mesh)
{
  if(!mesh)
    return;

  // Materials already modified directly in RAM by caller
  // Just mark for GPU upload
  mesh->flags |= MeshVk::Flags::eMaterialsChanged;
  pendingRequests |= Request::eUpdateMaterials;
}

void MeshManagerVk::setVisibilityDirty()
{
  pendingRequests |= Request::eUpdateDescriptors;
}

// =============================================================================
// VRAM SYNC - Process all deferred updates
// =============================================================================

void MeshManagerVk::processVramUpdates(bool processRtx)
{
  bool instanceCountChanged   = false;
  bool descriptorsNeedRebuild = false;

  // =========================================================================
  // Phase 1 - Remove from GPU + delete from RAM using shift-left compaction
  // =========================================================================

  if(static_cast<uint32_t>(pendingRequests & Request::eProcessDeletions))
  {
    // Step 1.1: Delete instances (shift-left compaction)
    {
      size_t originalSize = instances.size();
      size_t shiftLeft    = 0;

      for(size_t i = 0; i < originalSize; i++)
      {
        if(instances[i]->isMarkedForDeletion())
        {
          // Instance will be destroyed when shared_ptr is released
          shiftLeft++;
          instanceCountChanged = true;
        }
        else
        {
          // Keep instance - shift it left if needed
          instances[i - shiftLeft]        = instances[i];
          instances[i - shiftLeft]->index = i - shiftLeft;
        }
      }

      instances.resize(originalSize - shiftLeft);

      if(shiftLeft > 0)
        LOGD("Deleted %zu mesh instances\n", shiftLeft);
    }

    // Step 1.2: Delete meshes (only if no instances reference them)
    {
      size_t originalSize = meshes.size();
      size_t shiftLeft    = 0;

      for(size_t i = 0; i < originalSize; i++)
      {
        if(meshes[i]->isMarkedForDeletion())
        {
          // Verify no instances still reference this mesh
          bool hasReferences = false;
          for(const auto& inst : instances)
          {
            if(inst->mesh == meshes[i])
            {
              hasReferences = true;
              break;
            }
          }

          if(hasReferences)
          {
            // Clear delete flag - still in use
            meshes[i]->flags &= ~MeshVk::Flags::eDelete;
            // Keep this mesh
            meshes[i - shiftLeft]        = meshes[i];
            meshes[i - shiftLeft]->index = i - shiftLeft;
          }
          else
          {
            // Safe to delete
            vkDeviceWaitIdle(m_app->getDevice());
            meshes[i]->deinitBuffers(m_alloc);
            shiftLeft++;
          }
        }
        else
        {
          meshes[i - shiftLeft]        = meshes[i];
          meshes[i - shiftLeft]->index = i - shiftLeft;
        }
      }

      meshes.resize(originalSize - shiftLeft);

      if(shiftLeft > 0)
        LOGD("Deleted %zu meshes\n", shiftLeft);
    }

    if(instanceCountChanged)
    {
      descriptorsNeedRebuild = true;
      pendingRequests |= Request::eRebuildTLAS;
    }

    // Free GPU textures no longer referenced by any mesh
    pruneExpiredTextures();

    pendingRequests &= ~Request::eProcessDeletions;
  }

  // =========================================================================
  // PHASE 2: UPDATES (RAM → GPU sync)
  // =========================================================================

  // Process New meshes (allocate buffers + upload geometry)
  bool needsUpload = false;
  for(const auto& mesh : meshes)
  {
    if(static_cast<uint32_t>(mesh->flags & MeshVk::Flags::eNew))
    {
      // Allocate buffers and append to upload queue
      mesh->initBuffers(m_alloc, m_uploader);
      needsUpload = true;

      // Clear RAM copies after upload is queued (optional optimization)
      mesh->vertices.clear();
      mesh->vertices.shrink_to_fit();
      mesh->indices.clear();
      mesh->indices.shrink_to_fit();
      mesh->matIndices.clear();
      mesh->matIndices.shrink_to_fit();

      descriptorsNeedRebuild = true;
      pendingRequests |= Request::eRebuildBLAS;
      mesh->flags &= ~MeshVk::Flags::eNew;  // Clear flag
    }
  }

  // Execute all uploads in a single command buffer
  if(needsUpload)
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_uploader->cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_uploader->releaseStaging();
  }

  // Process New instances (add to descriptors)
  for(const auto& instance : instances)
  {
    if(static_cast<uint32_t>(instance->flags & MeshInstanceVk::Flags::eNew))
    {
      // Mesh geometry already uploaded above or in previous frame
      // Just needs descriptor entry
      instance->prevTransform = instance->transform;  // No object motion on first frame
      instanceCountChanged    = true;
      descriptorsNeedRebuild  = true;
      instance->flags &= ~MeshInstanceVk::Flags::eNew;  // Clear flag
    }
  }

  // Process material changes
  if(static_cast<uint32_t>(pendingRequests & Request::eUpdateMaterials))
  {
    for(const auto& mesh : meshes)
    {
      if(static_cast<uint32_t>(mesh->flags & MeshVk::Flags::eMaterialsChanged))
      {
        uploadMaterialsBufferInternal(mesh);
        mesh->flags &= ~MeshVk::Flags::eMaterialsChanged;
      }
    }

    // Update descriptor buffer so raster pipeline sees the new material data
    descriptorsNeedRebuild = true;

    pendingRequests &= ~Request::eUpdateMaterials;
  }

  // =========================================================================
  // PHASE 3: RTX ACCELERATION STRUCTURES (BLAS first, then TLAS)
  // =========================================================================

  // Only process RTX if in RTX pipeline mode
  // In raster mode, defer RTX builds until pipeline switch
  if(processRtx)
  {
    // IMPORTANT: Rebuild BLAS BEFORE any TLAS operations!
    // BLAS must exist before TLAS can reference them

    // Rebuild BLAS if needed
    if(static_cast<uint32_t>(pendingRequests & Request::eRebuildBLAS))
    {
      rtxDeinitAccelerationStructures();  // Clean up old structures first
      rtxInitAccelerationStructures();    // Rebuilds both BLAS and TLAS
      pendingRequests &= ~Request::eRebuildBLAS;
      pendingRequests &= ~Request::eRebuildTLAS;           // Already rebuilt
      pendingRequests &= ~Request::eUpdateTransformsOnly;  // Already applied in TLAS build
    }
    // Rebuild TLAS if needed (and BLAS wasn't rebuilt)
    else if(static_cast<uint32_t>(pendingRequests & Request::eRebuildTLAS))
    {
      rtxDeinitAccelerationStructures();  // Clean up old structures first
      rtxInitAccelerationStructures();    // Rebuilds both BLAS and TLAS
      pendingRequests &= ~Request::eRebuildTLAS;
      pendingRequests &= ~Request::eUpdateTransformsOnly;  // Already applied in TLAS build
    }
    // Update TLAS only (fast path for transform changes)
    else if(static_cast<uint32_t>(pendingRequests & Request::eUpdateTransformsOnly))
    {
      // Update TLAS with new transforms (for RTX pipeline - no rebuild, just update)
      rtxUpdateTopLevelAccelerationStructure();

      // Transform changes also need descriptor updates
      // (eTransformChanged instance flags are cleared in PHASE 4 post-upload)
      pendingRequests |= Request::eUpdateDescriptors;
      pendingRequests &= ~Request::eUpdateTransformsOnly;
    }
  }
  else
  {
    // Raster mode: Keep BLAS/TLAS rebuild flags for deferred processing
    // When switching to RTX pipeline, these accumulated flags will trigger rebuild
    // Do NOT clear RebuildBLAS or RebuildTLAS flags

    // However, transform changes MUST update descriptor buffers immediately
    // (raster shaders read MeshDesc transform matrices from descriptor buffer)
    if(static_cast<uint32_t>(pendingRequests & Request::eUpdateTransformsOnly))
    {
      // Trigger descriptor buffer update (needed for raster pipeline)
      // (eTransformChanged instance flags are cleared in PHASE 4 post-upload)
      pendingRequests |= Request::eUpdateDescriptors;

      // Clear UpdateTransformsOnly (we've handled it by updating descriptors)
      // When switching to RTX, RebuildBLAS/RebuildTLAS flags will trigger full rebuild
      pendingRequests &= ~Request::eUpdateTransformsOnly;
    }
  }

  // transformDirtyCount > 0 triggers additional descriptor uploads for DLSS object motion vectors
  for(const auto& instance : instances)
  {
    if(instance && instance->transformDirtyCount > 0)
    {
      pendingRequests |= Request::eUpdateDescriptors;
      break;
    }
  }

  // =========================================================================
  // PHASE 4: DESCRIPTORS (After all RTX structures are ready)
  // =========================================================================

  // Rebuild descriptors if needed (must be after BLAS/TLAS so addresses are valid)
  if(descriptorsNeedRebuild || static_cast<uint32_t>(pendingRequests & Request::eUpdateDescriptors))
  {
    updateMeshDescriptionBuffer();
    pendingRequests &= ~Request::eUpdateDescriptors;

    // DLSS object motion vectors: decrement counter after each descriptor upload.
    // Count 2 → 1: uploaded old prevTransform, sync CPU prevTransform = transform for next round.
    // Count 1 → 0: uploaded prevTransform == transform (zero object motion), clear flag.
    for(const auto& instance : instances)
    {
      if(!instance || instance->transformDirtyCount <= 0)
        continue;
      instance->transformDirtyCount--;
      if(instance->transformDirtyCount == 1)
        instance->prevTransform = instance->transform;
      if(instance->transformDirtyCount <= 0)
        instance->flags &= ~MeshInstanceVk::Flags::eTransformChanged;
    }

    // Keep pendingRequests alive if any instance still needs descriptor uploads,
    // so AssetManagerVk::processVramUpdates calls us again next frame.
    for(const auto& instance : instances)
    {
      if(instance && instance->transformDirtyCount > 0)
      {
        pendingRequests |= Request::eUpdateDescriptors;
        break;
      }
    }
  }
}

}  // namespace vk_gaussian_splatting
