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

#include "utilities_ui.h"

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

namespace vk_gaussian_splatting {

bool beginCollapsibleGroup(const char* label, bool defaultOpen)
{
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
  if(defaultOpen)
    flags |= ImGuiTreeNodeFlags_DefaultOpen;
  bool open = ImGui::TreeNodeEx(label, flags);
  if(open)
    ImGui::Indent(ImGui::GetStyle().IndentSpacing * 2.0f);
  return open;
}

void endCollapsibleGroup(bool open)
{
  if(open)
  {
    ImGui::Unindent(ImGui::GetStyle().IndentSpacing * 2.0f);
    ImGui::TreePop();
  }
}

}  // namespace vk_gaussian_splatting
