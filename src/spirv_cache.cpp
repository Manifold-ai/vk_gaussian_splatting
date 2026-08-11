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

// (A) Wipe-on-generation-change. The "generation signature" folds only the parts shared by
//     EVERY key — CACHE_VERSION, the source stamp, and the options stamp — but not the
//     per-shader name or per-render macro set. When it differs from the value persisted in
//     <cacheDir>/cache.stamp, every existing .spv was produced by a dead generation (a shader
//     edit, compiler-option change, or format bump), so the whole cache is cleared and the new
//     signature written. Deletion happens ONLY when a prior stamp exists and differs (a dir we
//     previously owned); a first run just records the baseline and deletes nothing, so aiming
//     --spirvCacheDir at a dir holding unrelated .spv files is safe. Rendering settings
//     (DLSS/pipeline/temporal/...) are MACROS, so they do NOT enter this signature and never
//     trigger a wipe — new macro combos just add coexisting entries, which (B) bounds instead.
void wipeIfGenerationChanged(const std::filesystem::path& cacheDir, uint32_t version, uint64_t sourceStamp, uint64_t optionsStamp)
{
  uint64_t sig = kFnvOffset;
  fnvAppend(sig, &version, sizeof(version));
  fnvAppend(sig, &sourceStamp, sizeof(sourceStamp));
  fnvAppend(sig, &optionsStamp, sizeof(optionsStamp));
  char sigStr[24];
  std::snprintf(sigStr, sizeof(sigStr), "%016llx", (unsigned long long)sig);

  const std::filesystem::path stampPath = cacheDir / "cache.stamp";
  std::string                 prev;
  {
    std::ifstream sin(stampPath);
    if(sin)
      std::getline(sin, prev);
  }
  if(prev == sigStr)
    return;  // same generation -> keep all entries

  // Only wipe when a PRIOR stamp exists and differs — i.e. a directory THIS cache previously
  // owned whose generation genuinely changed. With no prior stamp (first run on a fresh or a
  // user-supplied dir, or a dir populated by a pre-eviction build) we delete NOTHING and just
  // record the baseline signature. This keeps `--spirvCacheDir <dir-with-unrelated-.spv>` from
  // destroying the user's files, and lets a valid pre-upgrade cache survive (stale entries are
  // then bounded by (B) LRU, and real generation changes still wipe once the baseline is set).
  const bool hadPriorStamp = !prev.empty();
  uint32_t   cleared       = 0;
  if(hadPriorStamp)
  {
    std::error_code ec;
    for(auto it = std::filesystem::directory_iterator(cacheDir, ec);
        !ec && it != std::filesystem::directory_iterator(); it.increment(ec))
    {
      std::error_code fec;
      if(!it->is_regular_file(fec) || fec)  // files only — never a subdir/symlink named *.spv
        continue;
      const std::string n = it->path().filename().string();
      if(endsWith(n, ".spv") || n.find(".spv.tmp.") != std::string::npos)  // final blobs + stale temps
      {
        std::error_code rec;
        if(std::filesystem::remove(it->path(), rec))
          ++cleared;
      }
    }
  }

  // Persist the new signature via a per-process-unique temp + rename (same pattern as store())
  // so concurrent starters never share or clobber one temp; a losing rename drops its own temp.
  static std::atomic<uint64_t> s_stampCounter{0};
  std::filesystem::path        tmp = stampPath;
  tmp += ".tmp." + std::to_string((long long)VKGS_GETPID()) + "." + std::to_string(s_stampCounter.fetch_add(1));
  {
    std::ofstream sout(tmp, std::ios::trunc);
    sout << sigStr << "\n";
  }
  std::error_code rec;
  std::filesystem::rename(tmp, stampPath, rec);
  if(rec)
    std::filesystem::remove(tmp, rec);
  if(cleared > 0)
    LOGI("SpirvCache: generation changed - cleared %u stale entries\n", cleared);
}

// (B) Size-cap LRU. Sums the .spv entries; if over maxBytes, deletes them oldest-mtime-first
//     until under cap. load() bumps an entry's mtime on every hit, so "oldest mtime" means
//     "least recently used" — frequently-read shaders survive, only stale macro-set variants
//     are evicted. cache.stamp and *.tmp are excluded from the size accounting.
void enforceSizeCap(const std::filesystem::path& cacheDir, size_t maxBytes)
{
  if(maxBytes == 0)
    return;  // unlimited
  struct Ent
  {
    std::filesystem::path           path;
    std::uintmax_t                  size;
    std::filesystem::file_time_type mtime;
  };
  std::vector<Ent> ents;
  std::uintmax_t   total = 0;
  std::error_code  ec;
  for(auto it = std::filesystem::directory_iterator(cacheDir, ec);
      !ec && it != std::filesystem::directory_iterator(); it.increment(ec))
  {
    const std::string n = it->path().filename().string();
    if(!endsWith(n, ".spv"))  // only final .spv blobs (skip cache.stamp / *.tmp)
      continue;
    std::error_code      sec, tec;
    const std::uintmax_t sz = std::filesystem::file_size(it->path(), sec);
    const auto           mt = std::filesystem::last_write_time(it->path(), tec);
    if(sec || tec)
      continue;
    ents.push_back({it->path(), sz, mt});
    total += sz;
  }
  if(total <= maxBytes)
    return;
  std::sort(ents.begin(), ents.end(), [](const Ent& a, const Ent& b) { return a.mtime < b.mtime; });  // oldest first
  uint32_t       evicted = 0;
  std::uintmax_t freed   = 0;
  for(const auto& e : ents)
  {
    if(total <= maxBytes)
      break;
    std::error_code rec;
    if(std::filesystem::remove(e.path, rec))
    {
      total -= e.size;
      freed += e.size;
      ++evicted;
    }
  }
  if(evicted > 0)
    LOGI("SpirvCache: over cap - evicted %u LRU entries (freed %.1f MB)\n", evicted, (double)freed / (1024.0 * 1024.0));
}
}  // namespace

void SpirvCache::init(const std::filesystem::path&              cacheDir,
                      const std::vector<std::filesystem::path>& sourceDirs,
                      const std::string&                        optionsDesc,
                      size_t                                    maxBytes)
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

    // Bound the cache before first use. (A) drops entries left over from a previous generation
    // (shader/compiler change); (B) caps total size, evicting least-recently-used entries.
    // Both are best-effort — a cleanup failure must never disable the cache (see catch below).
    wipeIfGenerationChanged(m_cacheDir, CACHE_VERSION, m_sourceStamp, m_optionsStamp);  // (A)
    enforceSizeCap(m_cacheDir, maxBytes);                                               // (B)

    m_enabled = true;
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

  auto attempt = [&]() -> std::optional<std::vector<uint32_t>> {
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
      // LRU bookkeeping: mark this entry as just-used so the size-cap eviction keeps hot shaders
      // and only drops genuinely-unused macro-set variants. Best-effort (ignore read-only dirs).
      std::error_code tec;
      std::filesystem::last_write_time(p, std::filesystem::file_time_type::clock::now(), tec);
      return words;
    }
    catch(...)
    {
      return std::nullopt;
    }
  };

  std::optional<std::vector<uint32_t>> result = attempt();
  if(result)
    ++m_hits;  // served from disk (skips slang compilation)
  else
    ++m_misses;  // caller will compile + store()
  return result;
}

void SpirvCache::logStats(const char* context)
{
  if(m_enabled)
    LOGI("SpirvCache: %s: %u loaded from cache, %u compiled\n", context, m_hits, m_misses);
  m_hits   = 0;
  m_misses = 0;
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
