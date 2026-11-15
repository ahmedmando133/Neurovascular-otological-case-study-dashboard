#!/usr/bin/env python3
"""
Skill 2: High-Density Neuroimaging & PDF Page Extractor
1. Extracts and deduplicates all unique 1.0-mm axial source slices from the MRA 3D TOF
   DICOM viewer cine video (Video01), naming each frame semantically as lossless PNG.
2. Builds contact sheets across the Z-stack for rapid radiological inspection.
3. Auto-orients (correcting 90/180 deg capture rotations) and converts all 10 MRI/MRA/MRV
   film sheets into upright lossless PNGs organized by sequence subfolder.
4. Renders all pages of the 3-page Radiology PDF and 29-page Holter ECG PDF at 200 DPI.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

WORKSPACE_ROOT = Path(r"d:\mum")
INPUTS_DIR = WORKSPACE_ROOT / "01_Structured_and_Renamed_Inputs"
VIDEOS_DIR = INPUTS_DIR / "01_Neuroimaging_MRI_MRA_MRV_Videos"
FRAMES_ROOT = INPUTS_DIR / "02_Extracted_and_Enhanced_MRI_Frames"
HOLTER_DIR = INPUTS_DIR / "06_Cardiology_24h_Holter_ECG"

OUT_AXIAL = FRAMES_ROOT / "01_Axial_T1_T2_FLAIR_Brain_and_BasalGanglia"
OUT_MRA = FRAMES_ROOT / "02_MRA_Intracranial_Arteries"
OUT_MRV = FRAMES_ROOT / "03_MRV_Venous_Sinuses_and_Left_IJV"


def extract_unique_video_slices(video_path: Path, out_dir: Path, diff_thresh: float = 2.8) -> list[dict]:
    """Extract unique axial slices from cine screen recording by comparing the brain ROI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_roi = None
    saved_frames = []
    raw_idx = 0
    unique_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_idx += 1
        h, w = frame.shape[:2]
        # Central brain image region (excludes bottom slider & top OS bar, keeps slice text)
        roi = cv2.cvtColor(frame[int(h * 0.16) : int(h * 0.80), int(w * 0.05) : int(w * 0.95)], cv2.COLOR_BGR2GRAY)
        roi_small = cv2.resize(roi, (160, 200), interpolation=cv2.INTER_AREA).astype(np.float32)

        is_new = False
        mad = 999.0
        if prev_roi is None:
            is_new = True
        else:
            mad = float(np.mean(np.abs(roi_small - prev_roi)))
            if mad >= diff_thresh:
                is_new = True

        if is_new:
            unique_idx += 1
            fname = f"MRA_2026-11-05_Video01_3D_TOF_Frame_{unique_idx:04d}_Axial_SourceSlice.png"
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), frame)
            saved_frames.append(
                {
                    "unique_index": unique_idx,
                    "raw_frame_index": raw_idx,
                    "timestamp_sec": round((raw_idx - 1) / fps, 3),
                    "mad_from_prev": round(mad, 2),
                    "filename": fname,
                }
            )
            prev_roi = roi_small

    cap.release()

    # Build a curated contact sheet (up to 48 evenly spaced unique slices across the volume)
    if saved_frames:
        sample_count = min(48, len(saved_frames))
        indices = np.linspace(0, len(saved_frames) - 1, sample_count, dtype=int)
        cols = 8
        rows = math.ceil(sample_count / cols)
        thumb_w, thumb_h = 270, 480
        sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (12, 12, 12))
        for grid_i, s_idx in enumerate(indices):
            fpath = out_dir / saved_frames[s_idx]["filename"]
            im = Image.open(fpath).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            r = grid_i // cols
            c = grid_i % cols
            sheet.paste(im, (c * thumb_w, r * thumb_h))
        sheet.save(out_dir.parent / "MRA_2026-11-05_Video01_48Slice_ZStack_ContactSheet.png")

    return {
        "total_raw_frames": total_raw,
        "fps": round(fps, 2),
        "unique_slices_extracted": len(saved_frames),
        "slices": saved_frames,
    }


def orient_and_export_films() -> list[dict]:
    """Convert raw neuroimaging film photos to upright lossless PNGs in sequence folders."""
    film_specs = [
        ("MRI_2026-11-05_Film01_Sagittal_T1W_SE_Brain_Midline_NormalBrainstem.jpg", OUT_AXIAL, 0),
        ("MRI_2026-11-05_Film05_Axial_DWI_ADC_NoAcuteRestrictedDiffusion.jpg", OUT_AXIAL, 0),
        ("MRI_2026-11-05_Film07_Axial_eFLAIR_CSVD_LacunarInfarcts_EthmoidSinusitis.jpg", OUT_AXIAL, 0),
        ("MRI_2026-11-05_Film08_Axial_T1W_SE_SulcalProminence_LtEyeSiliconeOil.jpg", OUT_AXIAL, 0),
        ("MRI_2026-11-05_Film09_Axial_eT2W_TSE_Lacunes_IAC_EthmoidSinusitis.jpg", OUT_AXIAL, 180),
        ("MRI_2026-11-05_Film10_Coronal_T2W_and_T1W_TSE_BasalGanglia_IAC_EthmoidSinusitis.jpg", OUT_AXIAL, 0),
        ("MRA_2026-11-05_Film02_3D_TOF_Coronal_MIP_Intracranial_Arteries_Unremarkable.jpg", OUT_MRA, 0),
        ("MRA_2026-11-05_Film03_3D_TOF_Axial_MIP_CircleOfWillis_Unremarkable.jpg", OUT_MRA, -90),  # 90 deg CW
        ("MRA_2026-11-05_Film04a_3D_TOF_SagittalOblique_MIP_Intracranial_Arteries_Take1.jpg", OUT_MRA, 0),
        ("MRA_2026-11-05_Film04b_3D_TOF_SagittalOblique_MIP_Intracranial_Arteries_Take2.jpg", OUT_MRA, 0),
        ("MRV_2026-11-05_Film06_3D_PCA_Venography_Attenuated_Lt_Transverse_Sigmoid_IJV.jpg", OUT_MRV, 90),  # 90 deg CCW
    ]

    exported = []
    for fname, dest_dir, ccw_deg in film_specs:
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = VIDEOS_DIR / fname
        im = Image.open(src)
        if ccw_deg != 0:
            im = im.rotate(ccw_deg, expand=True)
        out_name = Path(fname).stem + "_Upright.png"
        out_path = dest_dir / out_name
        im.save(out_path, format="PNG")
        exported.append(
            {
                "source": fname,
                "output_png": str(out_path.relative_to(WORKSPACE_ROOT)),
                "rotation_ccw_applied_deg": ccw_deg,
                "size": list(im.size),
            }
        )
    return exported


def render_pdf_pages(pdf_path: Path, out_dir: Path, prefix_names: list[str], dpi: int = 200) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    outputs = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        name = prefix_names[i] if i < len(prefix_names) else f"Page_{i+1:02d}.png"
        out_path = out_dir / name
        pix.save(str(out_path))
        outputs.append(str(out_path.relative_to(WORKSPACE_ROOT)))
    doc.close()
    return outputs


def main() -> None:
    video_path = VIDEOS_DIR / "MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil.mov"
    video_meta = extract_unique_video_slices(video_path, OUT_MRA / "Video01_Source_Slices", diff_thresh=2.8)
    films_meta = orient_and_export_films()

    rad_pdf = VIDEOS_DIR / "RAD_2026-11-05_MRI_MRA_MRV_Brain_Report_PortSaeedMilitary_DrGhadaAwad.pdf"
    rad_pages = render_pdf_pages(
        rad_pdf,
        VIDEOS_DIR / "Report_Pages",
        [
            "RAD_2026-11-05_Report_Page01_MRI_Brain.png",
            "RAD_2026-11-05_Report_Page02_MRA_Cerebral.png",
            "RAD_2026-11-05_Report_Page03_MRV_Cerebral.png",
        ],
        dpi=200,
    )

    holter_pdf = HOLTER_DIR / "CARD_2026-11-05_Holter24h_Complete_29Pages_Summary_HRV_and_FullDisclosure.pdf"
    holter_names = [
        "CARD_2026-11-05_Holter24h_Page01_ReportSummary.png",
        "CARD_2026-11-05_Holter24h_Page02_24HourTrends.png",
        "CARD_2026-11-05_Holter24h_Page03_AbsoluteSTAnalysis.png",
        "CARD_2026-11-05_Holter24h_Page04_QTc_QT_3CH_Analysis.png",
        "CARD_2026-11-05_Holter24h_Page05_HeartRateVariability_HRV.png",
    ] + [
        f"CARD_2026-11-05_Holter24h_Page{p:02d}_FullDisclosure_Hour{(9 + p - 6) % 24:02d}00.png"
        for p in range(6, 30)
    ]
    holter_pages = render_pdf_pages(holter_pdf, HOLTER_DIR / "Extracted_Pages", holter_names, dpi=200)

    summary = {
        "video_extraction": {
            "total_raw_frames": video_meta["total_raw_frames"],
            "fps": video_meta["fps"],
            "unique_slices_extracted": video_meta["unique_slices_extracted"],
            "sample_slices": video_meta["slices"][:10],
        },
        "upright_films_exported": films_meta,
        "radiology_report_pages": rad_pages,
        "holter_pages_rendered": len(holter_pages),
    }

    meta_out = FRAMES_ROOT / "extraction_metadata.json"
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video_extraction": video_meta,
                "upright_films_exported": films_meta,
                "radiology_report_pages": rad_pages,
                "holter_pages": holter_pages,
            },
            f,
            indent=2,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
