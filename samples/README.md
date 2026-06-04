# Samples

This folder contains sample `.vkgs` project files and a build script that
downloads the external assets they depend on.

## Prerequisites

The script requires a **bash** shell with the following tools available:

- `curl`
- `unzip`
- `git` (for sparse-checkout of repository assets)

On Windows, **Git Bash** provides all of these out of the box.

## Quick start

From within the `samples/` directory, run:

```bash
./build.sh
```

This downloads all assets into a `data/` subfolder next to the script.
The `data/` folder is git-ignored and will not be committed.

You can then open the `.vkgs` project files in the application by using
**File > Open Project** or by dragging and dropping them onto the 
vk_gaussian_splatting viewer window.

## Custom destination

You can pass a destination folder to assemble the samples elsewhere:

```bash
./build.sh /path/to/destination
```

When a destination is provided the script will:

1. Create the destination folder.
2. Copy all `.vkgs` project files into it.
3. Download assets into `<destination>/data/`.

## What gets downloaded

| Type | Source | Local path |
|------|--------|------------|
| File | [horse.obj](https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/horse.obj) | `data/horse.obj` |
| Google Drive | [20K-Photo-103Mspats-4x2KM-Andrii_Shramko_Poland-JG.ply](https://drive.google.com/file/d/1_Gvh6eRjUY1RU4-GCzROFHUeoAj7G-xT/view) | `data/teleportour/20K-Photo-103Mspats-4x2KM-Andrii_Shramko_Poland-JG.ply` |
| Google Drive | [Winter-Garden-view2.ply](https://drive.google.com/file/d/1UN79ygHHIsM8AHG1FkzI88gnkAG78tNt/view) | `data/teleportour/Winter-Garden-view2.ply` |
| Google Drive | [License Agreement](https://drive.google.com/file/d/12TLqEsp-LZBkg8YwddpwpC3i5e5mFChB/view) | `data/teleportour/License_Agreement.md` |
| Zip | [flowers_1.zip](http://developer.download.nvidia.com/ProGraphics/nvpro-samples/flowers_1.zip) | `data/flowers_1/` |
| Git (sparse) | [glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) — `Models/ABeautifulGame`, `Models/Corset`, `Models/FlightHelmet`, `Models/DiffuseTransmissionTeacup` | `data/glTF-Sample-Assets/` |

## Re-running the script

The script is idempotent: files and folders that already exist are skipped.
Git sparse-checkout entries are updated if their checkout paths change in the
script, so adding new folders to an existing repo entry and re-running will
fetch the new content.
