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

#include "spirv_cache.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>

#include <nvutils/logger.hpp>

#if defined(_WIN32)
#include <process.h>
#define VKGS_GETPID() _getpid()
#else
#include <unistd.h>
#define VKGS_GETPID() ::getpid()
#endif

namespace vk_gaussian_splatting {

namespace {
constexpr uint64_t kFnvOffset  = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime   = 1099511628211ULL;
constexpr uint32_t kSpirvMagic = 0x07230203u;

inline void fnvAppend(uint64_t& h, const void* data, size_t n)
{
  const auto* p = static_cast<const unsigned char*>(data);
  for(size_t i = 0; i < n; ++i)
  {
    h ^= p[i];
    h *= kFnvPrime;
  }
}

inline bool endsWith(const std::string& s, const char* suffix)
{
  const size_t n = std::char_traits<char>::length(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}
}  // namespace

void SpirvCache::init(const std::filesystem::path&              cacheDir,
                      const std::vector<std::filesystem::path>& sourceDirs,
                      const std::string&                        optionsDesc)
{
  m_enabled = false;
  try
  {
    // 1) Source stamp: hash the CONTENTS of every shader source file found under the search
    //    roots (getShaderDirs()), which include src/shaders AND the copied nvshaders/ tree
    //    that shaders #include. A global stamp is used because Slang does not expose per-shader
    //    include dependencies; any edit to any .slang/.h busts the whole cache (conservative,
    //    correct). Iterated in sorted path order for determinism.
    std::vector<std::filesystem::path> files;
    for(const auto& dir : sourceDirs)
    {
      std::error_code ec;
      if(!std::filesystem::exists(dir, ec) || ec)
        continue;
      for(auto it = std::filesystem::recursive_directory_iterator(
              dir, std::filesystem::directory_options::skip_permission_denied, ec);
          !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec))
      {
        std::error_code fec;
        if(!it->is_regular_file(fec) || fec)
          continue;
        const std::string name = it->path().filename().string();
        if(endsWith(name, ".slang") || endsWith(name, ".h"))  // .h.slang ends with .slang
          files.push_back(it->path());
      }
    }
    std::sort(files.begin(), files.end());

    uint64_t stamp = kFnvOffset;
    for(const auto& f : files)
    {
      const std::string p = f.generic_string();
      fnvAppend(stamp, p.data(), p.size());  // include path identity, not just content
      std::ifstream in(f, std::ios::binary);
      if(!in)
        continue;
      char buf[65536];
      while(in.read(buf, sizeof(buf)) || in.gcount())
        fnvAppend(stamp, buf, static_cast<size_t>(in.gcount()));
    }
    m_sourceStamp = stamp;

    // 2) Options stamp
    uint64_t opt = kFnvOffset;
    fnvAppend(opt, optionsDesc.data(), optionsDesc.size());
    m_optionsStamp = opt;

    // 3) Ensure the cache directory is usable
    std::error_code ec;
    std::filesystem::create_directories(cacheDir, ec);
    if(!std::filesystem::exists(cacheDir, ec) || ec)
    {
      LOGW("SpirvCache: cannot use cache dir '%s' - disabling disk cache.\n", cacheDir.string().c_str());
      return;
    }
    m_cacheDir = cacheDir;
    m_enabled  = true;
    LOGI("SpirvCache: enabled at '%s' (source %016llx, options %016llx)\n", m_cacheDir.string().c_str(),
         (unsigned long long)m_sourceStamp, (unsigned long long)m_optionsStamp);
  }
  catch(const std::exception& e)
  {
    LOGW("SpirvCache: init failed (%s) - disabling disk cache.\n", e.what());
    m_enabled = false;
  }
  catch(...)
  {
    m_enabled = false;
  }
}

std::filesystem::path SpirvCache::keyPath(const std::string&                                      shaderName,
                                          const std::vector<std::pair<std::string, std::string>>& macros) const
{
  uint64_t       h   = kFnvOffset;
  const uint32_t ver = CACHE_VERSION;
  fnvAppend(h, &ver, sizeof(ver));
  fnvAppend(h, shaderName.data(), shaderName.size());
  const char sep = '\0';
  fnvAppend(h, &sep, 1);

  // Sort the macro set so the key is insensitive to build order.
  std::vector<std::pair<std::string, std::string>> sorted(macros);
  std::sort(sorted.begin(), sorted.end());
  for(const auto& m : sorted)
  {
    fnvAppend(h, m.first.data(), m.first.size());
    const char eq = '=';
    fnvAppend(h, &eq, 1);
    fnvAppend(h, m.second.data(), m.second.size());
    const char nl = '\n';
    fnvAppend(h, &nl, 1);
  }
  fnvAppend(h, &m_sourceStamp, sizeof(m_sourceStamp));
  fnvAppend(h, &m_optionsStamp, sizeof(m_optionsStamp));

  char name[24];
  std::snprintf(name, sizeof(name), "%016llx.spv", (unsigned long long)h);
  return m_cacheDir / name;
}

std::optional<std::vector<uint32_t>> SpirvCache::load(
    const std::string& shaderName, const std::vector<std::pair<std::string, std::string>>& macros) const
{
  if(!m_enabled)
    return std::nullopt;
  try
  {
    const std::filesystem::path p = keyPath(shaderName, macros);
    std::ifstream               in(p, std::ios::binary | std::ios::ate);
    if(!in)
      return std::nullopt;
    const std::streamsize bytes = in.tellg();
    if(bytes <= 0 || (bytes % 4) != 0)
      return std::nullopt;
    in.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    if(!in.read(reinterpret_cast<char*>(words.data()), bytes))
      return std::nullopt;
    if(words.empty() || words[0] != kSpirvMagic)  // reject non-SPIR-V / truncated
      return std::nullopt;
    return words;
  }
  catch(...)
  {
    return std::nullopt;
  }
}

void SpirvCache::store(const std::string&                                      shaderName,
                       const std::vector<std::pair<std::string, std::string>>& macros,
                       const uint32_t*                                         spirv,
                       size_t                                                  sizeInBytes) const
{
  if(!m_enabled || spirv == nullptr || sizeInBytes < 4 || (sizeInBytes % 4) != 0)
    return;
  if(spirv[0] != kSpirvMagic)  // never cache a non-SPIR-V blob
    return;
  try
  {
    const std::filesystem::path finalPath = keyPath(shaderName, macros);

    // Atomic write: unique temp file in the same dir + rename, so a concurrent reader (the
    // benchmark spawns many processes) never observes a half-written .spv.
    static std::atomic<uint64_t> s_counter{0};
    std::filesystem::path        tmp = finalPath;
    tmp += ".tmp." + std::to_string((long long)VKGS_GETPID()) + "." + std::to_string(s_counter.fetch_add(1));
    {
      std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
      if(!out)
        return;
      out.write(reinterpret_cast<const char*>(spirv), static_cast<std::streamsize>(sizeInBytes));
      if(!out)
      {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return;
      }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, finalPath, ec);  // atomic within the directory
    if(ec)
      std::filesystem::remove(tmp, ec);  // lost race / error: identical bytes, harmless
  }
  catch(...)
  {
    // Never let caching break compilation.
  }
}

}  // namespace vk_gaussian_splatting
