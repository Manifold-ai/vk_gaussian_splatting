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
#                   history warm-up / accumulation speed). Runs in ONE
#                   process (single cold start) since only spp varies.
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
from typing import Dict, List, Optional

from vkgs import (
    HeadlessRunner,
    Pipeline,
    RenderScript,
    Scene,
    TemporalSamplingMode,
    find_outputs,
    render_scene,
)

# DLSS size modes (src/gaussian_splatting_ui.cpp: "Min"/"Optimal"/"Max"; -1 =
# disabled, expressed here via dlss_enabled=False).
DLSS_MIN, DLSS_OPTIMAL, DLSS_MAX = 0, 1, 2
ALL_PASS = 0  # RtxTraceStrategy.FULL_ANYHIT

# The camera position rendered from the scene's preset list (teapot+scene.vkgs
# ships exactly one preset -> index 0).
CAMERA = 0

# Load/settle frames before the first capture (matches facade._MIN_SETTLE_FRAMES).
SETTLE_FRAMES = 5

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
def frame_gpu_from_timers(timers: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Whole-frame GPU ms: 'Primary Timeline' when present, else the sum of
    stage timers (only reached if that top-level timer is ever renamed)."""
    frame = timers.get("Primary Timeline", {}).get("gpu_ms")
    if frame is None and timers:
        frame = sum(t["gpu_ms"] for t in timers.values())
    return frame


def timers_for_sequence(sequences, name: str, fallback_richest: bool = True) -> Dict[str, Dict[str, float]]:
    """Timers of the sequence named `name`. When absent and `fallback_richest`,
    use the richest timer set (safe only when one capture sequence exists; a
    multi-capture batch must key exactly, so it passes fallback_richest=False)."""
    seq = next((s for s in sequences if s.name == name), None)
    if seq is None and fallback_richest:
        seq = max(sequences, key=lambda s: len(s.timers), default=None)
    return dict(seq.timers) if seq else {}


def copy_final_image(buffer_paths: Dict[str, str], out_root: str, key: str) -> Optional[str]:
    """Pick the tonemapped LDR (else main) image from a buffer->path map and copy
    it into out_root/images/<key><ext>. Returns the copied path (or the source if
    it cannot be copied, or None)."""
    img_src = (buffer_paths or {}).get("ldr") or (buffer_paths or {}).get("main")
    if not (img_src and os.path.isfile(img_src)):
        return img_src or None
    images_dir = os.path.join(out_root, "images")
    os.makedirs(images_dir, exist_ok=True)
    dst = os.path.join(images_dir, key + os.path.splitext(img_src)[1])
    shutil.copyfile(img_src, dst)
    return dst


def result_buffer_paths(result, camera: int) -> Dict[str, str]:
    """buffer->path map for one camera of a RenderResult (missing buffers skipped)."""
    paths: Dict[str, str] = {}
    for buf in ("ldr", "main"):
        try:
            paths[buf] = result.path(camera, buf)
        except KeyError:
            pass
    return paths


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

    timers = timers_for_sequence(result.sequences, f"cam{CAMERA}")
    frame_gpu = frame_gpu_from_timers(timers)
    fps = 1000.0 / frame_gpu if frame_gpu else None
    image = copy_final_image(result_buffer_paths(result, CAMERA), out_root, cfg.key)

    return {
        "sweep": cfg.sweep,
        "label": cfg.label,
        "spp": cfg.spp,
        "overrides": ";".join(f"{k}={v}" for k, v in cfg.renderer.items()),
        "frame_gpu_ms": round(frame_gpu, 4) if frame_gpu else "",
        "fps": round(fps, 2) if fps else "",
        "wall_s": round(wall, 2),
        "image": image or "",
        "log": result.log_path,
        "_timers": timers,
    }


# ------------------------------------------------ D sweep in a single process
def run_dlss_spp_batch(base: Scene, spp_list, out_root: str, size, gpu, exe, timeout, buffers):
    """Render the DLSS spp sweep (1..N) in ONE process -> a single cold start
    for all spp instead of one per spp.

    Each spp still measures a *fresh* k-frame DLSS accumulation, identical to
    the per-process version. The trick: DLSS history only resets when the frame
    counter resets, and the frame counter resets only when the view matrix/fov
    changes (updateFrameCounter, gaussian_splatting.cpp:3795; DLSS reset =
    frameSampleId==0, :623/:1081). Re-activating the *same* preset 0 does NOT
    change the view -> no reset -> spp would accumulate cumulatively. So before
    each real capture we activate a throwaway 'park' preset (a different pose);
    the following activateCameraPreset(0) is then a real view change -> reset ->
    frameSampleId=0 -> DLSS history cleared -> exactly k fresh frames.

    Returns (rows, batch_wall_s). Outputs are validated per-spp (not batch-wide),
    so a single missing capture only blanks that one row.
    """
    scene = copy.deepcopy(base)
    scene.renderer.dlss_enabled = True
    scene.renderer.dlss_size_mode = DLSS_OPTIMAL
    scene.renderer.temporal_sampling_mode = int(TemporalSamplingMode.DISABLED)
    scene.renderer.temporal_sampling = False

    if not scene.camera_presets:
        raise ValueError("scene has no camera presets; add preset 0 before the D sweep")

    # Park preset: preset 0 offset far away, so it differs from preset 0 for any
    # scene and re-activating preset 0 always counts as a view change (reset).
    p0 = scene.camera_presets[0]
    park = copy.deepcopy(p0)
    park.eye = (p0.eye[0] + 1000.0, p0.eye[1] + 1000.0, p0.eye[2] + 1000.0)
    park_idx = scene.add_camera_preset(park)

    out_dir = os.path.join(out_root, "D_dlss_spp")
    os.makedirs(out_dir, exist_ok=True)
    vkgs_path = scene.save(os.path.join(out_dir, "scene.vkgs"))

    script = RenderScript()
    script.load_block(frames=SETTLE_FRAMES)  # load + settle (home camera != preset 0)

    stems: Dict[int, str] = {}
    for spp in spp_list:
        # throwaway frame at the park view -> forces the next preset-0 activation
        # to reset the accumulation.
        script.sequence(f"park{spp:02d}", frames=1, averages=1, reset_frames=0,
                        activateCameraPreset=park_idx)
        stem = os.path.join(out_dir, f"D_spp{spp:02d}")
        script.capture(f"D_spp{spp:02d}", camera_preset=0, out_stem=stem,
                       frames=spp, averages=min(spp, 32), buffers=buffers)
        stems[spp] = stem

    cfg_path = script.write(os.path.join(out_dir, "render.cfg"))
    # expected_outputs=() so one missing capture does not fail the whole batch;
    # each spp's output is checked individually below (find_outputs).
    run = HeadlessRunner(exe).run(
        cfg_path, project=vkgs_path, size=size, gpu=gpu, timeout=timeout,
        log_path=os.path.join(out_dir, "render.log"), expected_outputs=(),
    )

    seq_by_name = {s.name: s for s in run.sequences}
    rows: List[Dict[str, object]] = []
    for spp in spp_list:
        seq = seq_by_name.get(f"D_spp{spp:02d}")
        # Exact key only: a global richest-set fallback would grab another spp's
        # timers in this multi-capture run, so leave blank if this seq is absent.
        timers = dict(seq.timers) if seq else {}
        frame_gpu = frame_gpu_from_timers(timers)
        fps = 1000.0 / frame_gpu if frame_gpu else None
        image = copy_final_image(find_outputs(stems[spp]), out_root, f"D_dlss_spp__spp{spp:02d}")

        rows.append({
            "sweep": "D_dlss_spp", "label": f"spp{spp:02d}", "spp": spp,
            "overrides": "dlss_size_mode=1;single-process batch",
            "frame_gpu_ms": round(frame_gpu, 4) if frame_gpu else "",
            "fps": round(fps, 2) if fps else "",
            "wall_s": "",  # per-spp wall is not meaningful in a batch; see the summary note
            "image": image or "",
            "log": run.log_path,
            "_timers": timers,
        })
    return rows, round(run.duration_s, 2)


def _collect_stages(row, seen_stages, stage_names) -> None:
    for s in row["_timers"]:
        if s not in seen_stages:
            seen_stages.add(s)
            stage_names.append(s)


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


def write_summary_md(rows: List[Dict[str, object]], path: str, notes: Optional[List[str]] = None) -> None:
    lines = ["# teapot+scene 性能测试结果", "",
             "| sweep | label | spp | frame GPU ms | FPS | wall s | image |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        img = os.path.relpath(r["image"], os.path.dirname(path)) if r["image"] else ""
        lines.append(f"| {r['sweep']} | {r['label']} | {r['spp']} | "
                     f"{r['frame_gpu_ms']} | {r['fps']} | {r['wall_s']} | {img} |")
    for note in notes or []:
        lines += ["", note]
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
        n_d = sum(1 for c in configs if c.sweep == "D_dlss_spp")
        n_proc = (len(configs) - n_d) + (1 if n_d else 0)
        print(f"{len(configs)} configs -> {n_proc} process(es); scene={args.scene}; "
              f"size={w}x{h}; out={args.out}")
        for c in configs:
            tag = " [batched: 1 process]" if c.sweep == "D_dlss_spp" else ""
            print(f"  {c.key:<24} spp={c.spp:<4} {c.renderer}{tag}")
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
    notes: List[str] = []

    # D (DLSS spp sweep) shares one .vkgs / camera and only varies sequenceframes,
    # so it runs in ONE process (single cold start). A/B/C change renderer
    # settings baked into the .vkgs -> one process each (cold start unavoidable).
    d_configs = [c for c in configs if c.sweep == "D_dlss_spp"]
    other_configs = [c for c in configs if c.sweep != "D_dlss_spp"]
    total = len(other_configs) + (1 if d_configs else 0)
    step = 0

    for cfg in other_configs:
        step += 1
        print(f"[{step}/{total}] {cfg.key} (spp={cfg.spp}) ...", flush=True)
        try:
            row = run_one(base, cfg, args.out, (w, h), args.gpu, args.exe, args.timeout, buffers)
        except Exception as exc:  # keep going; record the failure
            print(f"    FAILED: {exc}", file=sys.stderr)
            rows.append({"sweep": cfg.sweep, "label": cfg.label, "spp": cfg.spp,
                         "overrides": ";".join(f"{k}={v}" for k, v in cfg.renderer.items()),
                         "frame_gpu_ms": "", "fps": "", "wall_s": "", "image": "",
                         "log": str(exc), "_timers": {}})
            continue
        _collect_stages(row, seen_stages, stage_names)
        rows.append(row)
        print(f"    frame={row['frame_gpu_ms']}ms fps={row['fps']} wall={row['wall_s']}s -> {row['image']}")

    if d_configs:
        step += 1
        spp_list = [c.spp for c in d_configs]
        print(f"[{step}/{total}] D_dlss_spp batch: {len(spp_list)} spp in ONE process "
              f"(single cold start) ...", flush=True)
        try:
            d_rows, batch_wall = run_dlss_spp_batch(
                base, spp_list, args.out, (w, h), args.gpu, args.exe, args.timeout, buffers)
            for r in d_rows:
                _collect_stages(r, seen_stages, stage_names)
            rows.extend(d_rows)
            notes.append(f"> D_dlss_spp ran in ONE process (single cold start): "
                         f"{len(spp_list)} spp in {batch_wall}s wall total "
                         f"(per-spp wall_s is blank by design; compare frame GPU ms).")
            print(f"    {len(spp_list)} spp checkpoints, batch wall={batch_wall}s "
                  f"(was ~{len(spp_list)} cold starts)")
        except Exception as exc:
            print(f"    D BATCH FAILED: {exc}", file=sys.stderr)
            for c in d_configs:
                rows.append({"sweep": c.sweep, "label": c.label, "spp": c.spp,
                             "overrides": "single-process batch", "frame_gpu_ms": "", "fps": "",
                             "wall_s": "", "image": "", "log": str(exc), "_timers": {}})

    # Order stage columns: key stages first (if present), then the rest.
    ordered_stages = [s for s in KEY_STAGES if s in seen_stages] + \
                     [s for s in stage_names if s not in KEY_STAGES]

    csv_path = os.path.join(args.out, "metrics.csv")
    md_path = os.path.join(args.out, "summary.md")
    write_csv(rows, ordered_stages, csv_path)
    write_summary_md(rows, md_path, notes)
    print_table(rows)
    print(f"\nmetrics : {csv_path}\nsummary : {md_path}\nimages  : {os.path.join(args.out, 'images')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
