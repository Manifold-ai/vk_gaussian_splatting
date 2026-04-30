# Image Comparison

The **Image Comparison** mode provides a split-view overlay for comparing rendering results side by side within the viewport. It is activated from the toolbar or by pressing the comparison button.

The comparison works by capturing a reference frame and comparing it against the live render. Each side of the split view can independently display one of the following modes:

*   **Frame Capture** – The captured reference image.
*   **Current Render** – The live rendering output.
*   **Difference (Raw)** – Absolute per-pixel difference between capture and current render.
*   **Difference (Red on Gray)** – Differences highlighted in red over a grayscale background.
*   **Difference (Red Only)** – Differences shown as red intensity on black.
*   **FLIP Error Map** – Perceptual error visualization using the FLIP metric.

An **Amplify** slider is available in difference modes to magnify subtle differences. The panel also reports quantitative metrics: **MSE**, **PSNR**, and **FLIP**. When temporal sampling is active, a **Capture vs Current** chart tracks these metrics over the accumulation frames.
