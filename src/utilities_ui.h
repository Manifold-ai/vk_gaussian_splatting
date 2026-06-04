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

#ifndef _UTILITIES_UI_H_
#define _UTILITIES_UI_H_

namespace vk_gaussian_splatting {

// Collapsible group with bordered frame and tree-node header.
// Returns true if the group is open and content should be drawn.
// Always call endCollapsibleGroup() after beginCollapsibleGroup().
bool beginCollapsibleGroup(const char* label, bool defaultOpen = false);
void endCollapsibleGroup(bool open);

}  // namespace vk_gaussian_splatting

#endif  // _UTILITIES_UI_H_
