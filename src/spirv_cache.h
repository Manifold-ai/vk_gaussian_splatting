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

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vk_gaussian_splatting {

// Persistent on-disk cache for compiled SPIR-V. The Slang source->SPIR-V front-end is the
// dominant cold-start cost (~9s for the RTX/DLSS pipeline); this lets a warm start skip it.
//
// A cached entry is reused ONLY when the shader identity, its full macro set, ALL shader
// source content, and the compiler options are byte-identical to when it was produced.
// That is the "reload only when necessary" mechanism: change any macro, edit any shader
// source, or change a compiler option, and the key changes -> miss -> recompile; nothing
// else triggers a rebuild.
//
// SPIR-V is GPU-portable (unlike VkPipelineCache), so a cache dir may be shared across
// machines/GPUs — device differences that affect the SPIR-V are already encoded in the
// macro set (e.g. GPU_VENDOR, RTX_USE_AABBS, MESH_SHADER_*).
//
// All operations are best-effort and never throw: any I/O problem disables the cache
// (load->miss, store->no-op) so compilation always proceeds.
class SpirvCache
{
public:
  // Bump when the on-disk format changes (last-resort invalidation independent of the key).
  static constexpr uint32_t CACHE_VERSION = 1;

  // cacheDir    : where <key>.spv files live (created if missing).
  // sourceDirs  : shader search roots (getShaderDirs()). The contents of every .slang/.h
  //               file found under them form the source stamp, so editing any shader or
  //               shared header (incl. nvshaders/) invalidates the cache.
  // optionsDesc : a string describing the fixed compiler target/options/capabilities and
  //               build type; folded into every key so an option/build change invalidates.
  // On any failure the cache disables itself (enabled()==false).
  void init(const std::filesystem::path&             cacheDir,
            const std::vector<std::filesystem::path>& sourceDirs,
            const std::string&                        optionsDesc);

  bool enabled() const { return m_enabled; }

  // Cached SPIR-V words on hit (validated magic + size), std::nullopt on miss/corrupt/disabled.
  std::optional<std::vector<uint32_t>> load(const std::string&                                      shaderName,
                                            const std::vector<std::pair<std::string, std::string>>& macros) const;

  // Atomically stores SPIR-V (size in BYTES). No-op if disabled or the blob is not valid SPIR-V.
  void store(const std::string&                                      shaderName,
             const std::vector<std::pair<std::string, std::string>>& macros,
             const uint32_t*                                         spirv,
             size_t                                                  sizeInBytes) const;

private:
  std::filesystem::path keyPath(const std::string&                                      shaderName,
                                const std::vector<std::pair<std::string, std::string>>& macros) const;

  bool                  m_enabled      = false;
  std::filesystem::path m_cacheDir;
  uint64_t              m_sourceStamp  = 0;  // FNV-1a over all shader-source file contents
  uint64_t              m_optionsStamp = 0;  // FNV-1a over optionsDesc
};

}  // namespace vk_gaussian_splatting
