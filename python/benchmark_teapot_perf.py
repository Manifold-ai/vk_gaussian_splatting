#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# benchmark_teapot_perf.py
#
# Headless performance benchmark for ~/teapot+scene.vkgs, driven through the
# `vkgs` Python layer (Scene.load -> render_scene -> HeadlessRunner). Each
# config renders one camera and reports the app's per-stage GPU/CPU timers
# (parsed from the benchmark log by vkgs.runner) plus wall-clock time, and
# saves the final image, so you get *both* an image output set and metrics.
#
# Default base (enforced on every config, matching the scene file):
#   pipeline            = 2 (pure 3DGRT / RTX)
#   rtx_trace_strategy  = 0 (All pass / FULL_ANYHIT)
#   gs_shadow_mask      = True   (+ force_surfel = True)
#   max_passes          = 200
#   temporal_sampling   = DISABLED (forced)
#   dlss_enabled        = True, dlss_size_mode = 1 (DLSS-RR Optimal)
#
# Sweeps (each varies ONE axis off the default base):
#   A dlss_tier   : DLSS off / Min(0) / Optimal(1) / Max(2)          [warmup spp]
#   B maxpass     : max_passes in {50, 100, 200, 400}                [warmup spp]
#   C temporal    : DLSS OFF, temporal ENABLED, count=spp in         (dlss off)
#                   {100,200,300,400,500}  (each 100 -> convergence step)
#   D dlss_spp    : DLSS Optimal on, temporal off, spp in 1..15      (DLSS
#                   history warm-up / accumulation speed)
#
# Requires: a built `vk_gaussian_splatting` binary (RTX GPU + DLSS runtime
# .so for DLSS configs) reachable via $VKGS_BIN or --exe, and the scene's
# assets (teapot.obj, splat.ply) co-located next to the .vkgs.
#
# Usage:
#   VKGS_BIN=/path/to/_bin/Release/vk_gaussian_splatting \
#     python benchmark_teapot_perf.py --scene ~/teapot+scene.vkgs --out ~/perf_out
#   python benchmark_teapot_perf.py --sweeps A,B --warmup-spp 16 --list
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import copy
import csv
import os
import shutil
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

from vkgs import Pipeline, Scene, TemporalSamplingMode, render_scene

# DLSS size modes (src/gaussian_splatting_ui.cpp: "Min"/"Optimal"/"Max"; -1 =
# disabled, expressed here via dlss_enabled=False).
DLSS_MIN, DLSS_OPTIMAL, DLSS_MAX = 0, 1, 2
ALL_PASS = 0  # RtxTraceStrategy.FULL_ANYHIT

# The camera position rendered from the scene's preset list (teapot+scene.vkgs
# ships exactly one preset -> index 0).
CAMERA = 0

# Timer stages worth surfacing in the printed table (full set still goes to
# the CSV). "Primary Timeline" is the whole-frame GPU total when present.
KEY_STAGES = ["Primary Timeline", "Raytracing", "DLSS", "Rasterization", "Tonemap", "Post process"]


# --------------------------------------------------------------- scene setup
def check_assets(scene: Scene) -> List[str]:
    """Return asset paths that do not exist on disk (so we can warn early)."""
    missing: List[str] = []
    for coll in (scene.splat_sets, scene.mesh_assets):
        for asset in coll:
            if not os.path.isfile(asset.path):
                missing.append(asset.path)
    return missing


def apply_base_defaults(scene: Scene) -> None:
    """Force the default base described in the module header onto scene.renderer."""
    r = scene.renderer
    r.pipeline = int(Pipeline.RTX)
    r.rtx_trace_strategy = ALL_PASS
    r.gs_shadow_mask = True
    r.force_surfel = True
    r.max_passes = 200
    r.min_transmittance = 0.01
    r.temporal_sampling_mode = int(TemporalSamplingMode.DISABLED)
    r.temporal_sampling = False
    r.dlss_enabled = True
    r.dlss_size_mode = DLSS_OPTIMAL


# --------------------------------------------------------------- sweep config
class Config:
    __slots__ = ("sweep", "label", "renderer", "spp")

    def __init__(self, sweep: str, label: str, renderer: Dict[str, object], spp: int):
        self.sweep = sweep
        self.label = label
        self.renderer = renderer  # renderer-attr -> value overrides
        self.spp = spp

    @property
    def key(self) -> str:
        return f"{self.sweep}__{self.label}"


def build_configs(warmup_spp: int, temporal_steps: List[int], dlss_spp_max: int) -> List[Config]:
    configs: List[Config] = []

    # A) DLSS tier sweep (temporal disabled, All pass, maxpass 200).
    configs.append(Config("A_dlss_tier", "off", {"dlss_enabled": False}, warmup_spp))
    configs.append(Config("A_dlss_tier", "min", {"dlss_enabled": True, "dlss_size_mode": DLSS_MIN}, warmup_spp))
    configs.append(Config("A_dlss_tier", "optimal", {"dlss_enabled": True, "dlss_size_mode": DLSS_OPTIMAL}, warmup_spp))
    configs.append(Config("A_dlss_tier", "max", {"dlss_enabled": True, "dlss_size_mode": DLSS_MAX}, warmup_spp))

    # B) max_passes sweep (DLSS Optimal on, temporal disabled). Four points.
    for mp in (50, 100, 200, 400):
        configs.append(Config("B_maxpass", f"mp{mp}", {"max_passes": mp}, warmup_spp))

    # C) DLSS off + temporal accumulation (count == frames rendered so it
    #    converges exactly at that sample count).
    for n in temporal_steps:
        configs.append(
            Config(
                "C_temporal",
                f"t{n}",
                {
                    "dlss_enabled": False,
                    "temporal_sampling_mode": int(TemporalSamplingMode.ENABLED),
                    "temporal_sampling": True,
                    "temporal_samples_count": n,
                },
                n,
            )
        )

    # D) DLSS accumulation speed: DLSS Optimal on, temporal off, spp 1..N.
    for spp in range(1, dlss_spp_max + 1):
        configs.append(
            Config("D_dlss_spp", f"spp{spp:02d}", {"dlss_enabled": True, "dlss_size_mode": DLSS_OPTIMAL}, spp)
        )

    return configs


# --------------------------------------------------------------- metrics
def extract_metrics(result) -> Tuple[Dict[str, Dict[str, float]], Optional[float]]:
    """Return (per-stage timers, whole-frame GPU ms) for the render sequence
    (named 'cam0'; the 'cam0 save' 1-frame sequence is ignored)."""
    seq = next((s for s in result.sequences if s.name == f"cam{CAMERA}"), None)
    if seq is None:  # fall back to the richest timer set
        seq = max(result.sequences, key=lambda s: len(s.timers), default=None)
    timers = dict(seq.timers) if seq else {}
    frame = timers.get("Primary Timeline", {}).get("gpu_ms")
    if frame is None and timers:
        frame = sum(t["gpu_ms"] for t in timers.values())
    return timers, frame


def final_image_path(result) -> Optional[str]:
    """Prefer the tonemapped LDR image, else the main color buffer."""
    for buf in ("ldr", "main"):
        try:
            return result.path(CAMERA, buf)
        except KeyError:
            continue
    return None


# --------------------------------------------------------------- run one
def run_one(base: Scene, cfg: Config, out_root: str, size, gpu, exe, timeout, buffers) -> Dict[str, object]:
    scene = copy.deepcopy(base)
    for attr, value in cfg.renderer.items():
        setattr(scene.renderer, attr, value)

    out_dir = os.path.join(out_root, cfg.sweep, cfg.label)
    t0 = time.monotonic()
    with warnings.catch_warnings():  # silence the low-spp underconverged warning
        warnings.simplefilter("ignore")
        result = render_scene(
            scene,
            cameras=[CAMERA],
            size=size,
            spp=cfg.spp,
            buffers=buffers,
            out_dir=out_dir,
            executable=exe,
            gpu=gpu,
            timeout=timeout,
        )
    wall = time.monotonic() - t0

    timers, frame_gpu = extract_metrics(result)
    fps = 1000.0 / frame_gpu if frame_gpu else None

    # Copy the final image into a flat, browsable set.
    img_src = final_image_path(result)
    img_dst = None
    if img_src and os.path.isfile(img_src):
        images_dir = os.path.join(out_root, "images")
        os.makedirs(images_dir, exist_ok=True)
        img_dst = os.path.join(images_dir, cfg.key + os.path.splitext(img_src)[1])
        shutil.copyfile(img_src, img_dst)

    return {
        "sweep": cfg.sweep,
        "label": cfg.label,
        "spp": cfg.spp,
        "overrides": ";".join(f"{k}={v}" for k, v in cfg.renderer.items()),
        "frame_gpu_ms": round(frame_gpu, 4) if frame_gpu else "",
        "fps": round(fps, 2) if fps else "",
        "wall_s": round(wall, 2),
        "image": img_dst or (img_src or ""),
        "log": result.log_path,
        "_timers": timers,
    }


# --------------------------------------------------------------- reporting
def write_csv(rows: List[Dict[str, object]], stages: List[str], path: str) -> None:
    fixed = ["sweep", "label", "spp", "overrides", "frame_gpu_ms", "fps", "wall_s"]
    stage_cols = [f"{s} gpu_ms" for s in stages] + [f"{s} cpu_ms" for s in stages]
    header = fixed + stage_cols + ["image", "log"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            timers = r["_timers"]
            row = [r[k] for k in fixed]
            row += [round(timers.get(s, {}).get("gpu_ms", float("nan")), 4) if s in timers else "" for s in stages]
            row += [round(timers.get(s, {}).get("cpu_ms", float("nan")), 4) if s in timers else "" for s in stages]
            row += [r["image"], r["log"]]
            w.writerow(row)


def print_table(rows: List[Dict[str, object]]) -> None:
    hdr = f"{'sweep':<12} {'label':<10} {'spp':>4} {'frame_ms':>9} {'fps':>7} {'wall_s':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['sweep']:<12} {r['label']:<10} {r['spp']:>4} "
              f"{str(r['frame_gpu_ms']):>9} {str(r['fps']):>7} {str(r['wall_s']):>7}")


def write_summary_md(rows: List[Dict[str, object]], path: str) -> None:
    lines = ["# teapot+scene 性能测试结果", "",
             "| sweep | label | spp | frame GPU ms | FPS | wall s | image |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        img = os.path.relpath(r["image"], os.path.dirname(path)) if r["image"] else ""
        lines.append(f"| {r['sweep']} | {r['label']} | {r['spp']} | "
                     f"{r['frame_gpu_ms']} | {r['fps']} | {r['wall_s']} | {img} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Headless perf benchmark for teapot+scene.vkgs")
    ap.add_argument("--scene", default=os.path.expanduser("~/teapot+scene.vkgs"))
    ap.add_argument("--out", default=os.path.expanduser("~/vkgs_perf_out"))
    ap.add_argument("--exe", default=None, help="path to vk_gaussian_splatting (else $VKGS_BIN / auto)")
    ap.add_argument("--gpu", type=int, default=None, help="--forcegpu index")
    ap.add_argument("--size", default="1920x1080", help="output resolution WxH")
    ap.add_argument("--warmup-spp", type=int, default=16, help="frames for DLSS-on / temporal-off sweeps (A, B)")
    ap.add_argument("--temporal-steps", default="100,200,300,400,500", help="sweep C accumulation frame counts")
    ap.add_argument("--dlss-spp-max", type=int, default=15, help="sweep D max spp (1..N)")
    ap.add_argument("--sweeps", default="A,B,C,D", help="comma list of sweep letters to run")
    ap.add_argument("--timeout", type=float, default=3600, help="per-config subprocess timeout (s)")
    ap.add_argument("--list", action="store_true", help="print the config plan and exit (no rendering)")
    args = ap.parse_args(argv)

    w, h = (int(x) for x in args.size.lower().split("x"))
    temporal_steps = [int(x) for x in args.temporal_steps.split(",") if x.strip()]
    want = {s.strip().upper() for s in args.sweeps.split(",") if s.strip()}
    letter_to_sweep = {"A": "A_dlss_tier", "B": "B_maxpass", "C": "C_temporal", "D": "D_dlss_spp"}
    wanted_sweeps = {letter_to_sweep[l] for l in want if l in letter_to_sweep}

    configs = [c for c in build_configs(args.warmup_spp, temporal_steps, args.dlss_spp_max)
               if c.sweep in wanted_sweeps]
    if not configs:
        print(f"no configs selected (sweeps={args.sweeps})", file=sys.stderr)
        return 2

    if args.list:
        print(f"{len(configs)} configs; scene={args.scene}; size={w}x{h}; out={args.out}")
        for c in configs:
            print(f"  {c.key:<24} spp={c.spp:<4} {c.renderer}")
        return 0

    if not os.path.isfile(args.scene):
        print(f"scene not found: {args.scene}", file=sys.stderr)
        return 2

    base = Scene.load(args.scene)
    apply_base_defaults(base)
    missing = check_assets(base)
    if missing:
        print("WARNING: asset(s) not found on disk (rendering will fail unless present):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    buffers = ("main", "ldr") if getattr(base.tonemapping, "is_active", False) else ("main",)
    os.makedirs(args.out, exist_ok=True)

    rows: List[Dict[str, object]] = []
    stage_names: List[str] = []
    seen_stages = set()
    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{total}] {cfg.key} (spp={cfg.spp}) ...", flush=True)
        try:
            row = run_one(base, cfg, args.out, (w, h), args.gpu, args.exe, args.timeout, buffers)
        except Exception as exc:  # keep going; record the failure
            print(f"    FAILED: {exc}", file=sys.stderr)
            rows.append({"sweep": cfg.sweep, "label": cfg.label, "spp": cfg.spp,
                         "overrides": ";".join(f"{k}={v}" for k, v in cfg.renderer.items()),
                         "frame_gpu_ms": "", "fps": "", "wall_s": "", "image": "",
                         "log": str(exc), "_timers": {}})
            continue
        for s in row["_timers"]:
            if s not in seen_stages:
                seen_stages.add(s)
                stage_names.append(s)
        rows.append(row)
        print(f"    frame={row['frame_gpu_ms']}ms fps={row['fps']} wall={row['wall_s']}s -> {row['image']}")

    # Order stage columns: key stages first (if present), then the rest.
    ordered_stages = [s for s in KEY_STAGES if s in seen_stages] + \
                     [s for s in stage_names if s not in KEY_STAGES]

    csv_path = os.path.join(args.out, "metrics.csv")
    md_path = os.path.join(args.out, "summary.md")
    write_csv(rows, ordered_stages, csv_path)
    write_summary_md(rows, md_path)
    print_table(rows)
    print(f"\nmetrics : {csv_path}\nsummary : {md_path}\nimages  : {os.path.join(args.out, 'images')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
