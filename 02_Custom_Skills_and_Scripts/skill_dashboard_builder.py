"""Autonomous MDT Clinical Dashboard & Neuroimaging Video Suite Builder.

1. Compiles browser-ready MP4 (H.264/yuv420p +faststart) neuroimaging videos:
   - Video 01 (WebH264): Lossless faststart MP4 remux of raw 3D-TOF MRA IMG_8048.MOV
   - Video 02 (Derived Cine): Synchronized 20-level Axial eT2W + eFLAIR sweep
     compiled from upright primary sheets (Film09 & Film07) highlighting
     Basal Ganglia lacunar infarcts, CSVD, silicone oil, IACs & ethmoiditis.
   - Video 03 (Derived Cine): 18-projection rotational 3D-PCA MRV venography
     sweep (Dark + Inverted Light MIP) compiled from primary Film06 highlighting
     Right-dominant venous drainage & attenuated Lt Transverse/Sigmoid/IJV (0.48).
2. Reads case_manifest.json (37 primary files) and clinical_quantification_results.json
   to build 06_Clinical_Dashboard/index.html with:
   - Prominent "Neuroimaging Video Suite (MRI, MRA, MRV)" right at top of main viewer
   - Relative paths (../01_Structured_and_Renamed_Inputs/...) compatible with local FS,
     GitHub Pages, and Netlify
   - Updated 2026-11-03 Definitive Masked PTA + GSI-39 Tympanometry charts
   - All 6 Consultant audits, 4 resolved dilemmas, 11 ROI zoom plates (incl. ROI5a/5b),
     37-file searchable table, drug hazard audit, checklists, EN/AR & Dark/Light toggles.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(r"d:\mum")
INPUTS_DIR = ROOT / "01_Structured_and_Renamed_Inputs"
VIDEOS_DIR = INPUTS_DIR / "01_Neuroimaging_MRI_MRA_MRV_Videos"
FRAMES_DIR = INPUTS_DIR / "02_Extracted_and_Enhanced_MRI_Frames"
AXIAL_DIR = FRAMES_DIR / "01_Axial_T1_T2_FLAIR_Brain_and_BasalGanglia"
MRV_DIR = FRAMES_DIR / "03_MRV_Venous_Sinuses_and_Left_IJV"
ROI_DIR = FRAMES_DIR / "04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA"
MANIFEST_PATH = INPUTS_DIR / "case_manifest.json"
QUANT_PATH = ROOT / "02_Custom_Skills_and_Scripts" / "clinical_quantification_results.json"
DASH_DIR = ROOT / "06_Clinical_Dashboard"
DASH_HTML = DASH_DIR / "index.html"
ROOT_REDIRECT_HTML = ROOT / "index.html"


def find_ffmpeg() -> Path:
    """Locate ffmpeg executable on PATH or standard WinGet install location."""
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    winget_glob = list(
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet")).rglob("ffmpeg.exe")
    )
    if winget_glob:
        return winget_glob[0]
    raise FileNotFoundError("ffmpeg executable not found on PATH or WinGet directory.")


def warp_grid_tiles(
    img: np.ndarray,
    src_pts: list[tuple[int, int]],
    cols: int,
    rows: int,
    cell_w: int,
    cell_h: int,
) -> list[np.ndarray]:
    """Perspective-rectify a photographed film sheet and slice into (rows*cols) panels."""
    width, height = cols * cell_w, rows * cell_h
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array(src_pts, dtype=np.float32), dst)
    rect = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_CUBIC)
    tiles: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            tiles.append(rect[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w].copy())
    return tiles


def fit_tile_on_canvas(tile: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize tile preserving aspect ratio centered inside (target_h, target_w, 3)."""
    th, tw = tile.shape[:2]
    scale = min(target_w / tw, target_h / th)
    nw, nh = max(1, int(round(tw * scale))), max(1, int(round(th * scale)))
    resized = cv2.resize(tile, (nw, nh), interpolation=cv2.INTER_CUBIC)
    out = np.full((target_h, target_w, 3), (28, 15, 10), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def encode_frames_to_mp4(
    frames_with_durations: Iterable[tuple[np.ndarray, float]],
    dest_mp4: Path,
    ffmpeg_bin: Path,
    fps: int = 30,
) -> None:
    """Encode BGR frame sequence to browser-safe H.264/yuv420p +faststart MP4."""
    dest_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp = Path(tmpdir)
        concat_lines: list[str] = []
        for idx, (frame, dur) in enumerate(frames_with_durations):
            img_path = tmp / f"f_{idx:03d}.png"
            cv2.imwrite(str(img_path), frame)
            concat_lines.append(f"file '{img_path.as_posix()}'")
            concat_lines.append(f"duration {dur:.3f}")
        # Repeat last frame per ffmpeg concat demuxer spec so final duration holds properly
        concat_lines.append(f"file '{(tmp / f'f_{idx:03d}.png').as_posix()}'")
        list_file = tmp / "frames.txt"
        list_file.write_text("\n".join(concat_lines), encoding="utf-8")

        cmd = [
            str(ffmpeg_bin),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-vf",
            f"fps={fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "21",
            "-movflags",
            "+faststart",
            str(dest_mp4),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode != 0 or not dest_mp4.exists():
            raise RuntimeError(f"ffmpeg failed encoding {dest_mp4.name}: {res.stderr}")


def build_neuroimaging_videos(ffmpeg_bin: Path) -> dict[str, dict[str, object]]:
    """Create/refresh the 3 streaming MP4 videos inside 01_Neuroimaging_MRI_MRA_MRV_Videos/."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. MRA 3D-TOF raw .mov -> lossless faststart WebH264 MP4 container
    src_mov = VIDEOS_DIR / "MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil.mov"
    mra_mp4 = VIDEOS_DIR / "MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil_WebH264.mp4"
    res = subprocess.run(
        [
            str(ffmpeg_bin),
            "-y",
            "-i",
            str(src_mov),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(mra_mp4),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0 or not mra_mp4.exists():
        raise RuntimeError(f"MRA MP4 remux failed: {res.stderr}")

    # 2. Axial T2/FLAIR Synchronized Derived Cine (Film09 + Film07, 20 slice levels)
    t2_img = cv2.imread(
        str(AXIAL_DIR / "MRI_2026-11-05_Film09_Axial_eT2W_TSE_Lacunes_IAC_EthmoidSinusitis_Upright.png")
    )
    fl_img = cv2.imread(
        str(AXIAL_DIR / "MRI_2026-11-05_Film07_Axial_eFLAIR_CSVD_LacunarInfarcts_EthmoidSinusitis_Upright.png")
    )
    if t2_img is None or fl_img is None:
        raise FileNotFoundError("Upright Film09 or Film07 PNG missing for Axial T2/FLAIR cine build.")

    t2_tiles = warp_grid_tiles(t2_img, [(45, 319), (668, 311), (690, 892), (31, 902)], 5, 4, 240, 280)
    fl_tiles = warp_grid_tiles(fl_img, [(44, 290), (652, 285), (681, 882), (27, 892)], 5, 4, 240, 280)

    slice_notes = {
        1: ("Skull Base / Maxillary & Ethmoid Sinuses", 1.0),
        2: ("Bilateral Ethmoidal Mucosal Thickening", 1.0),
        3: ("Inferior Cerebellum & Mastoid Air Cells (Clear)", 1.0),
        4: ("Ethmoid Sinusitis + Inferior Globes / CPA Level", 1.2),
        5: ("Lt Intraocular Silicone Oil Hyperintensity (T1FFE/FLAIR) + IACs", 1.4),
        6: ("Pons, IACs & Mid-Orbits (Symmetric Cranial Nerves)", 1.2),
        7: ("Midbrain, Temporal Lobes & Optic Pathways", 1.0),
        8: ("Suprasellar Cistern & Basal Cisterns", 1.0),
        9: ("Basal Ganglia — Right Lentiform Chronic Lacunar Infarct", 1.5),
        10: ("Basal Ganglia / Internal Capsule — Chronic Lacune & Periventricular CSVD", 1.5),
        11: ("Corona Radiata / Body of Lateral Ventricles (Fazekas PV 1-2)", 1.3),
        12: ("Centrum Semi-Ovale — Punctate DWM Foci (NO Dawson Fingers)", 1.2),
        13: ("High Centrum Semi-Ovale — Scattered Microangiopathic Foci", 1.0),
        14: ("High Fronto-Parietal White Matter (DWM Fazekas 1-2)", 1.0),
        15: ("Upper Convexity & Cortical Sulcal Prominence", 1.0),
        16: ("Superior Frontal/Parietal Cortex (Involutional Sulci)", 0.9),
        17: ("High Convexity Gyri — Normal Grey-White Differentiation", 0.9),
        18: ("Sub-Vertex Level — No Cortical Mass or Herniation", 0.9),
        19: ("High Vertex Slice 19", 0.9),
        20: ("Superior Vertex Slice 20", 1.0),
    }
    flair_frames: list[tuple[np.ndarray, float]] = []
    for idx in range(20):
        sl_num = idx + 1
        note, dur = slice_notes[sl_num]
        canvas = np.full((720, 1280, 3), (32, 18, 12), dtype=np.uint8)
        left_p = fit_tile_on_canvas(t2_tiles[idx], 612, 576)
        right_p = fit_tile_on_canvas(fl_tiles[idx], 612, 576)
        canvas[88:664, 20:632] = left_p
        canvas[88:664, 648:1260] = right_p
        cv2.line(canvas, (640, 88), (640, 664), (248, 189, 56), 2)
        # Header & Footer Annotations (BGR palette)
        cv2.rectangle(canvas, (0, 0), (1280, 80), (42, 23, 15), -1)
        cv2.putText(
            canvas,
            f"AXIAL eT2W (LEFT) vs eFLAIR (RIGHT)  |  SLICE {sl_num:02d} / 20",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (249, 245, 241),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Level Focus: {note}",
            (24, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (248, 189, 56),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "DERIVED CINE (Film09 + Film07)",
            (945, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (36, 191, 251),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (0, 672), (1280, 720), (42, 23, 15), -1)
        prog_w = int(1240 * (sl_num / 20.0))
        cv2.rectangle(canvas, (20, 678), (20 + prog_w, 684), (248, 189, 56), -1)
        cv2.putText(
            canvas,
            "1.5T MRI (2026-11-05) | CSVD Fazekas 1-2 + Chronic Rt Basal Ganglia Lacune | MS & Acute Stroke EXCLUDED",
            (24, 707),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (225, 213, 203),
            1,
            cv2.LINE_AA,
        )
        flair_frames.append((canvas, dur))

    t2_flair_mp4 = VIDEOS_DIR / "MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4"
    encode_frames_to_mp4(flair_frames, t2_flair_mp4, ffmpeg_bin)

    # 3. 3D-PCA MRV Venography Rotational Sweep (Film06, 18 Dark + 18 Light MIPs)
    mrv_img = cv2.imread(
        str(MRV_DIR / "MRV_2026-11-05_Film06_3D_PCA_Venography_Attenuated_Lt_Transverse_Sigmoid_IJV_Upright.png")
    )
    if mrv_img is None:
        raise FileNotFoundError("Upright Film06 PNG missing for 3D-MRV cine build.")
    mrv_tiles = warp_grid_tiles(mrv_img, [(66, 330), (675, 319), (692, 900), (55, 918)], 6, 6, 200, 200)

    mrv_frames: list[tuple[np.ndarray, float]] = []
    view_idx = 0
    for r in range(6):
        for c in range(3):
            view_idx += 1
            dark_tile = mrv_tiles[r * 6 + c]
            light_tile = mrv_tiles[r * 6 + c + 3]
            dur = 1.4 if view_idx in (1, 2, 16, 17, 18) else 1.0
            canvas = np.full((720, 1280, 3), (32, 18, 12), dtype=np.uint8)
            left_p = fit_tile_on_canvas(dark_tile, 580, 576)
            right_p = fit_tile_on_canvas(light_tile, 580, 576)
            canvas[88:664, 34:614] = left_p
            canvas[88:664, 666:1246] = right_p
            cv2.line(canvas, (640, 88), (640, 664), (248, 189, 56), 2)
            cv2.rectangle(canvas, (0, 0), (1280, 80), (42, 23, 15), -1)
            cv2.putText(
                canvas,
                f"3D-PCA MRV VENOGRAPHY (DARK & INVERTED MIP)  |  ROTATION {view_idx:02d} / 18",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.76,
                (249, 245, 241),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Right-Dominant Venous Drainage | Attenuated Lt Transverse, Sigmoid & IJV (L/R Calibre = 0.48) | ZERO Thrombosis",
                (24, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (248, 189, 56),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "DERIVED CINE (Film06)",
                (1035, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (36, 191, 251),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(canvas, (0, 672), (1280, 720), (42, 23, 15), -1)
            prog_w = int(1240 * (view_idx / 18.0))
            cv2.rectangle(canvas, (20, 678), (20 + prog_w, 684), (248, 189, 56), -1)
            cv2.putText(
                canvas,
                "Verified Orientation: PA Projection (Right=Right) | Amplified by 3.2 cm3 Flaccid Lt Eardrum (2026-11-03 GSI-39)",
                (24, 707),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (225, 213, 203),
                1,
                cv2.LINE_AA,
            )
            mrv_frames.append((canvas, dur))

    mrv_mp4 = VIDEOS_DIR / "MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4"
    encode_frames_to_mp4(mrv_frames, mrv_mp4, ffmpeg_bin)

    return {
        "mra": {"path": mra_mp4, "raw_mov": src_mov, "size": mra_mp4.stat().st_size},
        "t2_flair": {"path": t2_flair_mp4, "size": t2_flair_mp4.stat().st_size},
        "mrv": {"path": mrv_mp4, "size": mrv_mp4.stat().st_size},
    }


def build_js_data_payload(manifest: dict, quant: dict) -> str:
    """Assemble clean JSON data literals for injection into index.html."""
    holter_hours = [
        {"h": "09:00", "beats": 4653, "hr": 79, "pvc": 9, "vp": 0, "vr": 0, "st": "0.0"},
        {"h": "10:00", "beats": 5060, "hr": 90, "pvc": 13, "vp": 0, "vr": 0, "st": "-0.3"},
        {"h": "11:00", "beats": 5277, "hr": 100, "pvc": 69, "vp": 2, "vr": 0, "st": "+0.2"},
        {"h": "12:00", "beats": 5563, "hr": 96, "pvc": 6, "vp": 0, "vr": 0, "st": "-0.5"},
        {"h": "13:00", "beats": 5617, "hr": 100, "pvc": 60, "vp": 0, "vr": 0, "st": "-2.1 (13:11 CH3)"},
        {"h": "14:00", "beats": 5810, "hr": 108, "pvc": 49, "vp": 0, "vr": 0, "st": "+0.1"},
        {"h": "15:00", "beats": 4449, "hr": 91, "pvc": 7, "vp": 0, "vr": 0, "st": "0.0"},
        {"h": "16:00", "beats": 4145, "hr": 77, "pvc": 21, "vp": 0, "vr": 0, "st": "0.0"},
        {"h": "17:00", "beats": 4607, "hr": 84, "pvc": 34, "vp": 0, "vr": 0, "st": "-0.2"},
        {"h": "18:00", "beats": 4166, "hr": 75, "pvc": 18, "vp": 1, "vr": 0, "st": "0.0"},
        {"h": "19:00", "beats": 3842, "hr": 69, "pvc": 15, "vp": 0, "vr": 0, "st": "0.0"},
        {"h": "20:00", "beats": 3848, "hr": 67, "pvc": 9, "vp": 0, "vr": 0, "st": "0.0"},
        {"h": "21:00", "beats": 4195, "hr": 85, "pvc": 18, "vp": 1, "vr": 0, "st": "+6.7 artifact"},
        {"h": "22:00", "beats": 3926, "hr": 72, "pvc": 10, "vp": 0, "vr": 0, "st": "0.0"},
        {"h": "23:00", "beats": 3882, "hr": 71, "pvc": 33, "vp": 0, "vr": 1, "st": "0.0 (4b @73)"},
        {"h": "00:00", "beats": 4208, "hr": 81, "pvc": 24, "vp": 0, "vr": 0, "st": "0.0"},
        {"h": "01:00", "beats": 3501, "hr": 61, "pvc": 7, "vp": 0, "vr": 0, "st": "0.0"},
        {"h": "02:00", "beats": 3440, "hr": 63, "pvc": 9, "vp": 0, "vr": 0, "st": "0.0 (Min 51)"},
        {"h": "03:00", "beats": 3853, "hr": 72, "pvc": 24, "vp": 1, "vr": 0, "st": "0.0"},
        {"h": "04:00", "beats": 3944, "hr": 78, "pvc": 54, "vp": 3, "vr": 0, "st": "-0.1"},
        {"h": "05:00", "beats": 4050, "hr": 78, "pvc": 28, "vp": 1, "vr": 0, "st": "0.0"},
        {"h": "06:00", "beats": 4439, "hr": 92, "pvc": 71, "vp": 4, "vr": 1, "st": "-0.3 (SURGE)"},
        {"h": "07:00", "beats": 4565, "hr": 93, "pvc": 76, "vp": 3, "vr": 1, "st": "-0.4 (SURGE)"},
        {"h": "08:00", "beats": 4339, "hr": 98, "pvc": 164, "vp": 6, "vr": 1, "st": "-0.2 (104bpm run)"},
    ]

    consultants = [
        {
            "id": 1,
            "role": "Consultant 1 — Diagnostic Head, Neck & Neuroradiology",
            "badge": "b-blue",
            "verdict": "Right-Dominant MRV (L/R 0.48), CSVD Fazekas 1-2, Persistent Lt Silicone Oil (2.59x since 2024 CT), No High-Flow dAVF",
            "findings": [
                "<strong>MRV Asymmetry (Film06, Verified PA R-on-Right):</strong> Attenuated/hypoplastic Left Transverse, Sigmoid Sinus & Left IJV (<strong>0.48</strong> calibre ratio vs dominant Right); <strong>zero dural thrombosis</strong> and no giant sigmoid diverticulum.",
                "<strong>513-Slice 3D-TOF MRA Source Audit (Video01 & ROI1b/c):</strong> Aerated mastoids; normal Circle of Willis; no enlarged occipital feeder or cortical venous flow voids (note: 1.5T TOF cannot completely exclude low-flow Cognard Type I dAVF).",
                "<strong>Critical Radiology Report Omission & 2024 CT Continuity:</strong> Official 2026 Rology PDF completely omitted prominent <strong>Left Intraocular Silicone Oil</strong> (mean intensity <strong>130.4 vs 50.3 = 2.59x</strong> on T1FFE/FLAIR), proven pre-existing on <strong>2024-07-27 El-Zohor Non-Contrast CT</strong>; optic nerves show severe bilateral atrophy (<1/60 legal blindness).",
                "<strong>IAC/CPA & Sinuses (ROI2):</strong> Intact, symmetric IACs/VIII complexes (excluding schwannoma); bilateral ethmoidal sinus mucosal thickening.",
            ],
            "ar": "تقييم الأشعة التخصصي: الجيب الوريدي الشمال ضيق خلقياً (نسبة 0.48) بدون جلطة، ومقاطع الـ MRA (513 شريحة) لا تظهر ناسوراً عالي التدفق. التقرير الرسمي أغفل وجود زيت السيليكون بالعين الشمال (مؤكد منذ أشعة المقطعية CT في يوليو 2024) وضمور العصب البصري، بالإضافة لالتهاب الجيوب الغربالية.",
        },
        {
            "id": 2,
            "role": "Consultant 2 — Multiple Sclerosis (MS) & White Matter Disorders",
            "badge": "b-green",
            "verdict": "MS & CNS Vasculitis EXCLUDED — Hypertensive-Diabetic CSVD (Fazekas PV 1-2 / DWM 1-2) Accelerated by Holter Ischemia",
            "findings": [
                "<strong>2017 McDonald DIS / DIT Negative:</strong> Punctate deep WM and smooth thin periventricular FLAIR hyperintensities lack perpendicular Dawson fingers, corpus callosum callosal-septal involvement, brainstem/cerebellar plaques, or cortical lesions.",
                "<strong>DWI/ADC & Chronic Lacunes (ROI3):</strong> IsoDWI/ADC is completely negative (no acute stroke); chronic cavitary/FLAIR lacunes present in Right Basal Ganglia, Left Frontal & Right Parietal subcortex.",
                "<strong>Shared Holter–CSVD Microangiopathy Axis:</strong> The 24h Holter finding of <code>-2.1 mm ST depression</code> (13:11) + morning <code>3.78%</code> PVC burden + carotid <code>0.13 cm</code> IMT compounds episodic cerebral autoregulatory hypoperfusion.",
                "<strong>Blindness Explained Non-Demyelinating:</strong> Legal blindness (<1/60) & silicone oil stem from long-standing retinal/vascular retinopathy, NOT MS optic neuritis. <strong>Do NOT perform LP / MS workup.</strong>",
            ],
            "ar": "استبعاد التصلب المتعدد (MS) تماماً: البؤر البيضاء ناتجة عن تصلب الشرايين الدقيقة المرتبط بالضغط والسكر (CSVD) وجلطة قديمة بالعقد القاعدية، وليست التهاباً مناعياً، ومجال الـ DWI سليم بدون جلطات حادة. اضطراب ضربات القلب وهبوط التروية التاجي بالهولتر يضاعفان إجهاد هذه الشرايين الدقيقة.",
        },
        {
            "id": 3,
            "role": "Consultant 3 — Medical Neurology, Vascular & Movement Disorders",
            "badge": "b-amber",
            "verdict": "Mixed Tremor (Essential Tremor + Head Titubation + BG Lacunar Vulnerability + Morning Insulin Adrenergic Dips)",
            "findings": [
                "<strong>Movement Reconciliation (2026-11-03 Note + 2024 2h EEG):</strong> Prior PD/Sinemet label conflicts with Prof. Dr. Galhom's exam (<code>Essential tremor, Titubation, لا lower</code>). Moreover, <strong>2024-07-17 IMC 2h EEG (9 timestamped epochs, 11:42–13:28)</strong> shows benign 9 Hz alpha with bilateral fronto-temporal EMG tremor bursts and <strong>zero epileptiform spikes</strong>.",
                "<strong>Circadian 06:00-08:59 AM Holter Surge:</strong> <code>36.29%</code> of all 24h PVCs (311/857; peaking at 164 PVCs/hr & 104 bpm V-run @ 08:00 AM) cluster alongside <code>80 IU/day</code> insulin (Toujeo 30U + Apidra 50U), strongly pointing to occult early-morning neuroglycopenia/catecholamine spikes!",
                "<strong>Urgent Beta-Blocker & Drug Safety Conflict:</strong> Non-selective <code>Inderal 10 mg BID</code> masks insulin hypoglycemia warnings; <code>Angiopress Comp (HCTZ) + Cidamex</code> risks dangerous hypokalemia/PVCs; triple platelet inhibition (<code>Aggrex + Ginkgo + Escitalopram</code>) raises bleed hazard.",
                "<strong>Action Plan:</strong> Run supervised off/on <code>Sinemet MDS-UPDRS</code> test + <strong>14-day CGM</strong> + K+/Mg2+/eGFR before finalizing bisoprolol/propranolol & high-dose statin for <code>5 mm</code> bulb plaque.",
            ],
            "ar": "تحليل الأعصاب والرعشة: تخطيط المخ المطول (9 مقاطع عام 2024) استبعد البؤر الصرعية وأثبت وجود نشاط ألفا سليم 9 هيرتز مع ارتعاش عضلي (EMG tremor) متوافق مع تشخيص د. جلهوم (Essential Tremor + Titubation)، تتداخل معه الجلطة القديمة والطفرة الصباحية لضربات القلب (36.29% بين 6–9 ص) وهبوط السكر المحتمل مع 80 وحدة أنسولين.",
        },
        {
            "id": 4,
            "role": "Consultant 4 — Senior Consultant in Neurosurgery",
            "badge": "b-green",
            "verdict": "No Space-Occupying Lesion, Aneurysm, Hydrocephalus, or High-Grade Cognard IIb–IV dAVF — Surgery Not Indicated",
            "findings": [
                "<strong>2024 CT & 2026 MRI Cross-Check (ROI5b + ROI3):</strong> Zero extra-axial tumor (CPA meningioma/schwannoma excluded), aneurysm, Chiari malformation, or obstructive hydrocephalus.",
                "<strong>Venous & Cortical Reflux Clearance:</strong> No dilated cortical veins or cerebral venous hypertension edema; dominant Right Transverse-Sigmoid-IJV system compensates safely and MUST never be compromised.",
                "<strong>Endorsement of Deferred Invasive Path:</strong> Validates colleague's dAVF diagnostic vigilance while proving surgical/hemorrhagic emergency is absent; invasive DSA should wait behind plaque/renal/cardiac clearance.",
            ],
            "ar": "رأي جراحة المخ: لا يوجد ورم أو تمدد شرياني أو استسقاء أو ناسور خطر يضغط على القشرة المخية في كل من أشعة 2024 و2026، ولا حاجة لأي تدخل جراحي حالي أو قسطرة متعجلة.",
        },
        {
            "id": 5,
            "role": "Consultant 5 — Neuro-Interventional Radiology (DSA Controversy Resolution)",
            "badge": "b-red",
            "verdict": "DEFER IMMEDIATE CATHETER DSA — Enforce Mandatory 3-Step Non-Invasive Gatekeeper Protocol",
            "findings": [
                "<strong>Procedural Catheter Hazard Across Left Carotid Bulb:</strong> Carotid Duplex reveals <code>0.13 cm</code> IMT and a <code>5 mm</code> atheromatous plaque at the Left Carotid Bulb — navigating a diagnostic catheter into Left CCA/ECA risks plaque dislodgement/stroke.",
                "<strong>Pending Renal/Cardiology Gate:</strong> Unknown serum Cr/eGFR while taking Metformin (<code>Gliptus Plus</code>) risks iodinated contrast nephropathy/lactic acidosis, while active <code>-2.1 mm</code> Holter ST ischemia + <code>3.78%</code> AM PVC burden elevates periprocedural cardiac risk.",
                "<strong>3-Step Gatekeeper:</strong> (1) Bedside light anterior IJV vs pinpoint occipital compression + mastoid auscultation -> (2) Venous/sinus medical therapy first if venous; OR <strong>4D-MRA / Temporal Bone HRCT+CTA/CTV</strong> (after eGFR & 48h Metformin hold) -> (3) Conditional DSA only if positive.",
            ],
            "ar": "حسم جدل القسطرة المخية (DSA): يُمنع إجراؤها فوراً بسبب وجود بلاك 5 مم عند عنق الشريان السباتي الشمال، وعدم فحص وظائف الكلى مع الميتفورمين، وهبوط ST -2.1 مم بالهولتر. نلتزم بالفرز الثلاثي الآمن (اختبار الضغط السريري -> أشعة 4D-MRA أو مقطعية بالصبغة عند اللزوم -> قسطرة مشروطة فقط عند وجود اشتباه قاطع).",
        },
        {
            "id": 6,
            "role": "Consultant 6 — ENT, Neurotology & Clinical Audiology",
            "badge": "b-blue",
            "verdict": "2026-11-03 Definitive Masked SNHL (0 dB ABG, Rt WRS 84%, Lt WRS 76%) + Extreme Type Ad (3.2 cm³) + 4-Hit Vertigo",
            "findings": [
                "<strong>2026-11-03 Definitive Baseline Promoted (ROI5a):</strong> Masked AC/BC proves <strong>0 dB ABG</strong> bilaterally (Right <code>20–70 dB</code>, 3f PTA <code>40.0 dB</code>, SRT 25 dB, WRS <strong>84%</strong>; Left <code>50–80 dB</code>, 3f PTA <code>54.2 dB</code>, SRT 50 dB, WRS <strong>76%</strong>).",
                "<strong>Printout Disproves Handwritten 'Rt As':</strong> GSI-39 thermal trace measures Right Peak <strong>1.5 cm³</strong> @ -5 daPa (ECV 1.3, GR 25 — normal Type A) and Left Peak <strong>3.2 cm³</strong> exceeding 3.0 graph ceiling (ECV 1.2, GR 10 — <strong>Extreme Type Ad, 2.13x Rt</strong>, I 1000 Hz NT).",
                "<strong>Mechanical Vascular Tinnitus Amplifier + Holter Modulation:</strong> Flaccid monomeric Left TM (<code>3.2 cm³</code>) acts as an acoustic drumhead amplifying internal venous hum across the <code>0.48</code> hypoplastic sinus + <code>36.29%</code> AM PVC beat irregularity.",
                "<strong>4-Hit Vertigo & Mandatory ENT Interventions:</strong> Blindness (<1/60) strips ~95% visual reserve and <strong>INVALIDATES light-tracked VNG subtests</strong>. <strong>STOP Meclizine & Ginkgo</strong>; treat ethmoiditis (Mometasone); adapted dark VNG + non-visual VRT; Bilateral <strong>Open-Fit WDRC RIC (`OSPL90 <= 95-100 dB`)</strong>.",
            ],
            "ar": "تقييم السمعيات والاتزان النهائي (2026-11-03): فحص النغمات الموثق بقناع عكسي أثبت ضعف سمع حسي عصبي مستقر بدون أي فجوة توصيلية (0 dB ABG)، وضغط أذن يمين طبيعي (1.5 سم³ يصحح ملاحظة 'Rt As' اليدوية) وطبلية شمال مرنة جداً (3.2 سم³ Type Ad) تضخم سماع نبض الدم وتقلبات ضربات القلب (PVCs). يجب وقف الميكليزين فوراً وإلغاء تتبع الضوء في VNG مع تركيب سماعتين مفتوحتين بحد أقصى <= 100 dB.",
        },
    ]

    dilemmas = [
        {
            "title": "1. Left Pulsatile Tinnitus & Catheter DSA Controversy",
            "left_lbl": "Clinical Conflict: Pro-DSA vs Anti-DSA",
            "left": "One specialist urged immediate angiography (DSA) to rule out Cognard dAVF; another noted MRV asymmetry + negative 1.5T TOF MRA. Furthermore, prior posterior neck pressure stopped the sound — a pitfall that compresses BOTH IJV and Occipital Artery!",
            "right_lbl": "MDT Board Consensus & Pathophysiology",
            "right": "<strong>DEFER DSA.</strong> Likely venous hum from attenuated Lt TS/Sigmoid/IJV (<code>L/R 0.48</code>) amplified by the <strong>3.2 cm³ Extreme Type Ad Lt TM</strong> + nocturnal supine CVP surge & Holter <code>3.78%</code> AM PVC pauses. Follow <strong>3-Step Gatekeeper</strong> (light anterior IJV vs pinpoint occipital -> CTA/CTV or 4D-MRA after eGFR & 48h Metformin hold -> DSA only if positive) to avoid crossing the <code>5 mm</code> Lt carotid bulb plaque.",
        },
        {
            "title": "2. White Matter Foci, 2024 EEG & Mixed Tremor Conflict",
            "left_lbl": "Diagnostic Conflict: MS? PD? Epilepsy?",
            "left": "MRI report queried white matter foci; prior notes treated PD (Sinemet); IMC 2024 CT/EEG were ordered for 'tremors'; yet Prof. Dr. Galhom (2026-11-03) logged <code>Essential tremor + Titubation, لا lower</code>.",
            "right_lbl": "MDT Board Consensus & Pathophysiology",
            "right": "<strong>MS & Epilepsy EXCLUDED.</strong> 2026 MRI = <strong>CSVD Fazekas 1-2</strong> + Rt BG chronic lacune; 2024-07-17 2h EEG (<strong>9 epochs</strong>) = 9 Hz alpha + benign EMG tremor bursts only (0 spikes). Tremor is <strong>mixed</strong> (ET + head titubation + vascular BG + <code>80 IU/d</code> morning insulin surges hitting <code>36.29%</code> PVC peak at 06–09 AM). Run <strong>MDS-UPDRS off/on Sinemet</strong> + <strong>14-day CGM</strong>.",
        },
        {
            "title": "3. Disabling \"4-Hit\" Multi-Sensory Vertigo & VNG Safety Trap",
            "left_lbl": "Trap: Visual VNG Ordering & Meclizine Suppression",
            "left": "Standard VNG was ordered alongside daily Meclizine x1 month despite bilateral legal blindness (<1/60, MOH 2021) + persistent Lt silicone oil (<code>2.59x</code>).",
            "right_lbl": "MDT Board Consensus & Pathophysiology",
            "right": "<strong>4-Hit Vestibular Failure:</strong> Hit 1 (Lt inner ear vascular insult) + Hit 2 (<strong><1/60 legal blindness</strong> eliminating ~95% visual stabilization) + Hit 3 (CSVD + diabetic neuropathy + chronic <strong>Meclizine</strong> suppression) + Hit 4 (PVCs / <code>-2.1 mm ST</code> / glycemic dips). <strong>STOP Meclizine immediately</strong>; run <strong>dark positional + calorics VNG ONLY</strong>; start non-visual somatosensory VRT.",
        },
        {
            "title": "4. Definitive 2026-11-03 Auditory & 'Rt As' Tympanometry Paradox",
            "left_lbl": "Paradox: Handwritten 'Rt As' Note vs GSI-39 & Extreme Ad",
            "left": "2026-11-03 audiogram bears handwritten note <code>'Rt As'</code> while GSI-39 prints <code>Rt 1.5 cm³</code> & <code>Lt 3.2 cm³</code> (>3.0 ceiling), despite prior sudden SNHL pain sensitivity.",
            "right_lbl": "MDT Board Consensus & Pathophysiology",
            "right": "<strong>Primary GSI-39 Thermal Printout Governs:</strong> Rt peak is <strong>1.5 cm³ normal Type A</strong> (refuting 'Rt As'; ECV 1.3), and Lt is <strong>3.2 cm³ Extreme Type Ad</strong> with <strong>0 dB ABG</strong> (monomeric healed drum, NOT ossicular disruption!). Pure cochlear SNHL (Rt WRS <code>84%</code>, Lt WRS <code>76%</code>) + historical Metz recruitment demands <strong>Bilateral Open-Fit WDRC RIC (`OSPL90 <= 95-100 dB`)</strong>.",
        },
    ]

    roi_cards = [
        {"title": "ROI5a — 2026-11-03 Definitive Masked PTA & GSI-39", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI5a_Audiology_2026-11-03_MaskedPTA_and_GSI39_Tymp_Zoom.png", "desc": "Definitive primary audiology baseline: 0 dB ABG (Rt 20-70 dB WRS 84%; Lt 50-80 dB WRS 76%), GSI-39 Rt 1.5 cm3 Type A (refutes handwritten 'Rt As') and Lt 3.2 cm3 Type Ad (>3.0 ceiling, I 1000 Hz NT)."},
        {"title": "ROI5b — 2024 Non-Contrast CT & 2h EEG (9 Epochs)", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI5b_2024_CT_Brain_and_ProlongedEEG_TremorArtifacts.png", "desc": "2024-07-17 IMC CT req (#474510 tremors), 2024-07-27 El-Zohor CT confirming pre-existing Lt intraocular silicone oil, and 2024-07-17 2h EEG report/epochs (9 Hz alpha + somatic tremor EMG bursts, zero spikes)."},
        {"title": "ROI1a — MRV Venous Sinuses & Marker Verification", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1a_MRV_Left_Transverse_Sigmoid_IJV_CLAHE_Zoom.png", "desc": "3D PCA MRV CLAHE panel proving PA R-on-right orientation, dominant Right sinus drainage, and smooth attenuated Left Transverse/Sigmoid/IJV (L/R 0.48) with ZERO thrombosis."},
        {"title": "ROI1b — MRA 1.0-mm Source Slices (Skull Base & Mastoid)", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1b_MRA_3D_TOF_SourceSlices_SkullBase_Mastoid_Occipital_CLAHE.png", "desc": "Extractions from 513 deduplicated 3D-TOF source slices auditing skull base, ICA canals, and sigmoid grooves — zero high-flow AVF shunt or transosseous occipital feeder."},
        {"title": "ROI1c — Left Mastoid / Sigmoid Plate Micro-Zoom", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1c_Left_Mastoid_Sigmoid_Occipital_MicroZoom_dAVF_Audit.png", "desc": "High-magnification CLAHE audit of Left mastoid air cells (clear) and sigmoid plate contour."},
        {"title": "ROI2 — Neuro-Otological Axis, Orbits (Silicone 2.59x) & Ethmoids", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI2_NeuroOtological_IAC_CPA_Orbits_Ethmoid_CLAHE.png", "desc": "Normal IACs/CPAs, bilateral ethmoidal mucosal inflammation, and bright Left Intraocular Silicone Oil (130.4 vs 50.3 = 2.59x) overlooked on official MRI PDF."},
        {"title": "ROI3 — Basal Ganglia Lacune & White Matter CSVD (Fazekas 1-2)", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI3_BasalGanglia_WhiteMatter_LacunarInfarcts_CLAHE.png", "desc": "Side-by-side eFLAIR, eT2W, Coronal, and DWI/ADC proving chronic right lentiform lacune + CSVD Fazekas 1-2 with zero acute diffusion restriction and zero MS plaques."},
        {"title": "ROI4 — Dr. Galhom Handwriting & Holter -2.1 mm ST / Morning Surge", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI4_Clinical_Handwriting_and_Holter_KeyStrips_Zoom.png", "desc": "Handwritten Essential Tremor + Titubation exam alongside Holter Page 2/3/5 proving 36.29% AM PVC cluster (08:00 = 164 PVCs/V-run @ 104 bpm) & -2.1 mm CH3 ST depression @ 13:11."},
        {"title": "MRA Z-Stack — 48-Slice Reconstruction (Video01 Source)", "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/02_MRA_Intracranial_Arteries/MRA_2026-11-05_Video01_48Slice_ZStack_ContactSheet.png", "desc": "48 uniformly sampled levels across the 513 unique 1.0-mm 3D-TOF axial slices from skull base to vertex."},
        {"title": "Carotid & Vertebral Duplex (IMT 0.13 cm & 5 mm Bulb Plaque)", "img": "../01_Structured_and_Renamed_Inputs/05_Vascular_Carotid_Duplex/VASC_2026-11-07_Carotid_Vertebral_ColorDuplex_IMT0.13cm_LtBulbPlaque5mm.jpg", "desc": "Dr. Ibrahim Hashim (2026-11-07): IMT 0.13 cm (+44%), non-stenotic bilateral plaques with 5 mm plaque in Left Carotid Bulb directly on DSA catheter trajectory."},
        {"title": "MOH Medical Council Certificate (< 1/60 Legal Blindness)", "img": "../01_Structured_and_Renamed_Inputs/08_Clinical_Notes_and_Prescriptions/CLIN_2021-09-01_MOH_MedicalCouncil_LegalBlindness_LessThan1of60_RetinalOpticAtrophy.jpg", "desc": "Primary legal & ID anchor (ID 26304090300188, DOB 1963-04-09): legal blindness (<1/60 better eye) due to retinal and optic nerve atrophy."},
    ]

    # Map all 37 raw manifest files cleanly for table & filters
    files_index = []
    for f in manifest.get("files", []):
        cat = f["category_code"]
        date = f["date_of_test"]
        if cat in ("MRI", "MRA", "MRV") or (cat == "RAD" and date.startswith("2026")):
            grp = "RAD_MRI_2026"
        elif cat == "RAD":
            grp = "RAD_CT_2024"
        elif cat == "NEURO":
            grp = "NEURO_EEG_2024"
        elif cat == "AUD":
            grp = "AUD_2026"
        elif cat == "CARD":
            grp = "CARD"
        elif cat == "VASC":
            grp = "VASC"
        else:
            grp = "CLIN"
        sz_kb = f["size_bytes"] / 1024.0
        sz_str = f"{sz_kb/1024:.2f} MB" if sz_kb >= 1024 else f"{sz_kb:.0f} KB"
        files_index.append(
            {
                "cat": cat,
                "grp": grp,
                "date": date,
                "raw": f["original_raw_filename"],
                "newName": f["new_semantic_filename"],
                "href": "../" + f["folder_path"].replace("\\", "/"),
                "size": sz_str,
                "summary": f["one_line_clinical_summary"],
            }
        )

    med_audit = [
        {"med": "Angiopress Comp (Captopril 50 / HCTZ 25 OD) + Proposed Cidamex", "tag": "STOP", "badge": "b-red", "mech": "Dual renal potassium wasting + short 2h captopril half-life", "hazard": "Severe hypokalemia fueling 857 PVCs / 4 V-runs + nocturnal BP rebound surging tinnitus", "action": "STOP HCTZ & AVOID Cidamex; after K+/Mg2+/eGFR + Echo/Stress, transition to Perindopril 5 mg or Telmisartan 40-80 mg + Bisoprolol."},
        {"med": "Cetrovit (Ginkgo Biloba) + Aggrex 75 + Ciprapra 10", "tag": "STOP", "badge": "b-red", "mech": "Triple platelet aggregation blockade (COX-1 + PAF + platelet 5-HT)", "hazard": "Significant GI bleed & cerebral microbleed hazard on CSVD background without proven SNHL gain", "action": "STOP Cetrovit Ginkgo immediately; keep Aggrex 75-81 mg OD; review necessity of Escitalopram 10 mg."},
        {"med": "Meclizine 25 mg qPM x1 mo + Brain-Ox (Vincamine 30 mg)", "tag": "STOP", "badge": "b-red", "mech": "Central antihistamine vestibulosuppression + anticholinergic sedation + cerebral vasodilator", "hazard": "Blocks CNS vestibular compensation & elevates fall hazard in <1/60 blindness; Vincamine risks BP dips", "action": "STOP daily Meclizine (prn <=48h acute crisis only) & STOP Brain-Ox; initiate Non-Visual Somatosensory VRT."},
        {"med": "Inderal (Propranolol 10 mg BID) + Toujeo 30U + Apidra 50U", "tag": "CAUTION", "badge": "b-amber", "mech": "Non-selective beta1/beta2 blockade during high-dose insulin (80 IU/day)", "hazard": "Blunts tachycardia/tremor hypoglycemia alarms & delays hepatic glycogen recovery during 06-09 AM nadir", "action": "Start 14-day CGM immediately; ask Cardiology/Endocrinology about switching to cardioselective Bisoprolol (or monitored Propranolol LA)."},
        {"med": "Zatorva 20 mg (Atorvastatin)", "tag": "UPGRADE", "badge": "b-blue", "mech": "Moderate-intensity HMG-CoA reductase inhibitor", "hazard": "Insufficient plaque stabilization for 5 mm Lt carotid bulb plaque + -2.1 mm silent coronary ischemia", "action": "Upgrade to Atorvastatin 40-80 mg nocte targeting LDL < 55 mg/dL (monitor baseline ALT/CK)."},
        {"med": "Gliptus Plus (Vildagliptin/Metformin 50/1000 BID)", "tag": "CAUTION", "badge": "b-amber", "mech": "Renally cleared biguanide in pending eGFR patient", "hazard": "Metformin-associated lactic acidosis if iodinated CTA/DSA contrast given without eGFR & 48h hold", "action": "Check Cr/eGFR immediately (Metformin safe only if eGFR >= 45; HOLD 48h before/after any CTA/DSA); discuss Empagliflozin 10 mg."},
        {"med": "Bilateral Ethmoidal Sinusitis + Cobal F (Vit B12)", "tag": "UPGRADE", "badge": "b-green", "mech": "Intranasal steroid mucosal decongestion + Eustachian tube relief", "hazard": "Untreated sinus/mucosal congestion worsening ear fullness; empiric B12 before serum level", "action": "Add Mometasone 50 mcg 2 sprays/nostril daily + isotonic saline x6-8 wks + 30 deg HOB elevation; verify serum B12."},
    ]

    rx_table = [
        {"dom": "Cardiology & Vascular Risk", "reg": "Perindopril 5 mg OD (or Telmisartan 40-80) + Bisoprolol 2.5-5 mg + Atorvastatin 40-80 mg + Aggrex 75 mg", "dose": "After K+/Mg2+/eGFR + TTE & Dobutamine/SPECT Stress", "note": "Replaces Angiopress Comp (stops HCTZ K+ loss) & upgrades plaque/coronary protection; no treadmill test due to <1/60 blindness."},
        {"dom": "Endocrinology & Hypoglycemia Audit", "reg": "14-Day CGM (FreeStyle Libre) auditing Toujeo 30U + Apidra 50U + Gliptus Plus", "dose": "Check HbA1c, eGFR, Urine ACR; discuss Empagliflozin 10 mg", "note": "Targets 03:00-09:00 AM window driving 36.29% morning PVC surge & tremors; hold Metformin 48h for any contrast CTA/DSA."},
        {"dom": "Neurology & Tremor Clarification", "reg": "Formal MDS-UPDRS off/on Sinemet Challenge -> Taper if non-responsive", "dose": "Controlled video/clinical scoring in Neuro clinic", "note": "Reconciles prior PD label with Dr. Galhom's Essential Tremor + Titubation (2024 EEG = non-epileptic EMG tremor only)."},
        {"dom": "Neuro-Intervention & ENT Sinus Care", "reg": "Mometasone 50 mcg + Saline x6-8 wks + HOB 30 deg + Bedside Compression Gatekeeper", "dose": "Before 4D-MRA or Temporal HRCT+CTA/CTV (DSA only if positive)", "note": "Relieves venous tinnitus pitch & ethmoiditis; avoids crossing 5 mm Lt carotid bulb plaque blindly."},
        {"dom": "Neurotology, VNG & Hearing Aids", "reg": "Non-Visual VRT + Dark-Only VNG + Bilateral Open-Fit WDRC RIC Aids", "dose": "Left OSPL90 <= 95-100 dB SPL; soft vented domes", "note": "STOP Meclizine immediately; skip light tracking in VNG (<1/60 blindness); protects OHC loudness pain (Metz+) & avoids venous occlusion."},
    ]

    checklist = [
        {"id": "lab_elec", "txt": "STAT Serum Electrolytes & Renal Panel: K+, Mg2+, Na+, Ca2+, Creatinine/eGFR, Urine ACR (Stop HCTZ / Avoid Cidamex until K+ verified)"},
        {"id": "lab_met", "txt": "Start 14-Day CGM (03:00-09:00 AM hypoglycemia audit on 80 IU/d insulin) + HbA1c, Fasting Lipids (LDL target <55), TSH, CBC, Vit B12, ALT/AST"},
        {"id": "card_img", "txt": "Transthoracic Echocardiogram (TTE) + Pharmacological Stress Echo/SPECT for CH3 -2.1 mm silent ischemia (Treadmill INVALID due to <1/60 blindness)"},
        {"id": "bed_tin", "txt": "Quiet Mastoid Auscultation + Selective Light Anterior IJV vs Pinpoint Occipital Artery Compression to separate venous hum from occipital dAVF"},
        {"id": "ent_vng", "txt": "STOP Meclizine, Cetrovit Ginkgo & Brain-Ox; notify VNG Lab in writing: <1/60 legal blindness -> Dark Positional + Bithermal Calorics ONLY (Skip Light Tracking!)"},
        {"id": "sinus_ha", "txt": "Mometasone nasal spray + saline x6-8 wks + 30 deg head elevation; Bilateral Open-Fit WDRC RIC Hearing Aids (Left OSPL90 <= 95-100 dB SPL)"},
        {"id": "mov_updrs", "txt": "Perform formal off/on Sinemet MDS-UPDRS test before tapering Levodopa; coordinate Beta-Blocker selection with active CGM"},
        {"id": "gate_dsa", "txt": "Gatekeeper Rule: Only if bruit/red flags persist or medical/sinus trial fails -> 4D-MRA or Temporal HRCT+CTA/CTV (hold Metformin 48h!) BEFORE conditional DSA"},
    ]

    return json.dumps(
        {
            "holter_hours": holter_hours,
            "consultants": consultants,
            "dilemmas": dilemmas,
            "roi_cards": roi_cards,
            "files_index": files_index,
            "med_audit": med_audit,
            "rx_table": rx_table,
            "checklist": checklist,
        },
        ensure_ascii=False,
    )


def render_dashboard_html(data_json: str, file_count: int) -> str:
    """Return full standalone interactive HTML clinical dashboard with Neuroimaging Video Suite on top."""
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light" data-lang="both">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MDT Clinical Dashboard — Amal Mokhtar Mohamed Ibrahim (Age 63.5)</title>
<style>
:root {{
  --bg: #f1f5f9;
  --surface: #ffffff;
  --surface-alt: #f8fafc;
  --border: #cbd5e1;
  --text: #0f172a;
  --text-muted: #475569;
  --accent: #0284c7;
  --accent-dark: #0369a1;
  --accent-soft: #e0f2fe;
  --navy: #0f172a;
  --navy-grad: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
  --red: #dc2626;
  --red-bg: #fef2f2;
  --red-border: #fca5a5;
  --amber: #d97706;
  --amber-bg: #fffbeb;
  --amber-border: #fcd34d;
  --green: #16a34a;
  --green-bg: #f0fdf4;
  --green-border: #86efac;
  --ar-bg: #f0f9ff;
  --ar-border: #0284c7;
  --shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
  --code-bg: #f1f5f9;
}}
[data-theme="dark"] {{
  --bg: #0b1120;
  --surface: #111827;
  --surface-alt: #1e293b;
  --border: #334155;
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --accent-dark: #7dd3fc;
  --accent-soft: rgba(56, 189, 248, 0.14);
  --navy: #020617;
  --navy-grad: linear-gradient(135deg, #020617 0%, #0f172a 100%);
  --red: #f87171;
  --red-bg: rgba(220, 38, 38, 0.14);
  --red-border: rgba(248, 113, 113, 0.45);
  --amber: #fbbf24;
  --amber-bg: rgba(217, 119, 6, 0.14);
  --amber-border: rgba(251, 191, 36, 0.45);
  --green: #4ade80;
  --green-bg: rgba(22, 163, 74, 0.14);
  --green-border: rgba(74, 222, 128, 0.45);
  --ar-bg: rgba(2, 132, 199, 0.12);
  --ar-border: #38bdf8;
  --shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  --code-bg: #1e293b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Tahoma', Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  font-size: 14px;
}}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 12px;
  color: var(--text);
  background: var(--code-bg);
  border: 1px solid var(--border);
  padding: 1px 5px;
  border-radius: 4px;
  direction: ltr;
  unicode-bidi: isolate;
  display: inline-block;
}}
.topbar code {{ background: rgba(255,255,255,0.14); border-color: rgba(255,255,255,0.3); color: #f0f9ff; }}
[data-lang="en"] .lang-ar {{ display: none !important; }}
[data-lang="ar"] .lang-en {{ display: none !important; }}

.topbar {{
  background: var(--navy-grad);
  color: #fff;
  padding: 14px 20px;
  position: sticky;
  top: 0;
  z-index: 40;
  border-bottom: 2px solid rgba(56, 189, 248, 0.35);
  box-shadow: 0 4px 18px rgba(0,0,0,0.2);
}}
.topbar-inner {{
  max-width: 1480px;
  margin: 0 auto;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}}
.patient-title h1 {{ margin: 0; font-size: 17.5px; letter-spacing: 0.2px; color: #fff; }}
.patient-sub {{ font-size: 12px; color: #bae6fd; margin-top: 3px; display: flex; flex-wrap: wrap; gap: 10px; }}
.topbar-actions {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
.btn {{
  border: 1px solid rgba(255,255,255,0.25);
  background: rgba(255,255,255,0.1);
  color: #fff;
  padding: 6px 11px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  transition: all 0.15s ease;
}}
.btn:hover {{ background: rgba(255,255,255,0.22); text-decoration: none; }}
.btn-primary {{ background: #0284c7; border-color: #38bdf8; }}
.btn-primary:hover {{ background: #0369a1; }}
.btn-surface {{ background: var(--surface); color: var(--text); border: 1px solid var(--border); }}
.btn-surface:hover {{ background: var(--accent-soft); color: var(--accent-dark); }}
.btn-active {{ background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important; }}

.shell {{
  max-width: 1480px;
  margin: 18px auto;
  padding: 0 16px 40px 16px;
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: 18px;
}}
@media (max-width: 980px) {{ .shell {{ grid-template-columns: 1fr; }} }}

.sidebar {{
  position: sticky;
  top: 86px;
  align-self: start;
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.nav-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px;
  box-shadow: var(--shadow);
}}
.nav-label {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  color: var(--text-muted);
  padding: 4px 8px 6px 8px;
}}
.nav-btn {{
  width: 100%;
  text-align: left;
  background: transparent;
  border: none;
  color: var(--text);
  padding: 10px 11px;
  border-radius: 7px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 3px;
  transition: all 0.15s ease;
}}
.nav-btn:hover {{ background: var(--accent-soft); color: var(--accent-dark); }}
.nav-btn.active {{ background: var(--accent); color: #fff; }}
.nav-count {{ font-size: 11px; padding: 1px 6px; border-radius: 999px; background: rgba(148,163,184,0.22); }}
.nav-btn.active .nav-count {{ background: rgba(255,255,255,0.25); color: #fff; }}
.pdf-links a {{
  display: block;
  padding: 7px 9px;
  font-size: 12px;
  font-weight: 600;
  border-radius: 6px;
  color: var(--text);
  border: 1px solid transparent;
}}
.pdf-links a:hover {{ background: var(--accent-soft); border-color: var(--border); text-decoration: none; }}

.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}
.section-head {{
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}}
.section-head h2 {{ margin: 0; font-size: 20px; }}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}}
.card h3 {{ margin: 0 0 10px 0; font-size: 15.5px; color: var(--accent-dark); }}

/* Prominent Neuroimaging Video Suite Hero */
.video-suite-hero {{
  background: linear-gradient(180deg, rgba(14,165,233,0.08) 0%, var(--surface) 42%);
  border: 2px solid var(--accent);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 18px;
  box-shadow: var(--shadow);
}}
.video-suite-head {{
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}}
.video-suite-head h2 {{
  margin: 0;
  font-size: 18.5px;
  color: var(--accent-dark);
  display: flex;
  align-items: center;
  gap: 8px;
}}
.grid-cols-1 {{ display: grid; grid-template-columns: 1fr; }}
@media (min-width: 768px) {{ .md\:grid-cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
.gap-6 {{ gap: 1.5rem; }}
.w-full {{ width: 100%; }}
.h-64 {{ height: 16rem; }}
.object-contain {{ object-fit: contain; }}
.rounded-lg {{ border-radius: 0.5rem; }}
.shadow-md {{ box-shadow: 0 4px 10px rgba(0,0,0,0.22); }}
.bg-black {{ background: #000000; }}
.video-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.5rem;
}}
@media (min-width: 768px) {{ .video-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
.video-card {{
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.video-frame-wrap {{
  background: #000000;
  width: 100%;
  height: 16rem;
  border-radius: 0.5rem;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.video-frame-wrap video {{
  width: 100%;
  height: 16rem;
  object-fit: contain;
  display: block;
  background: #000000;
}}
.video-card-body {{ display: flex; flex-direction: column; gap: 8px; font-size: 12.6px; flex: 1; }}
.video-card-body h4 {{ margin: 0; font-size: 14.5px; color: var(--text); }}
.video-actions {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; margin-top: auto; padding-top: 8px; border-top: 1px solid var(--border); }}
.btn-download {{
  background: var(--accent);
  color: #ffffff !important;
  border: 1px solid var(--accent-dark);
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 700;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}}
.btn-download:hover {{ background: var(--accent-dark); text-decoration: none; }}

.grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
.grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
.grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
@media (max-width: 1100px) {{ .grid-4 {{ grid-template-columns: repeat(2, 1fr); }} .grid-3 {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 720px) {{ .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }} }}

.kpi {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 9px;
  padding: 12px 14px;
  box-shadow: var(--shadow);
}}
.kpi.danger {{ border-left-color: var(--red); }}
.kpi.warn {{ border-left-color: var(--amber); }}
.kpi.ok {{ border-left-color: var(--green); }}
.kpi .lbl {{ font-size: 11.5px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; }}
.kpi .val {{ font-size: 20px; font-weight: 800; margin: 3px 0; }}
.kpi .sub {{ font-size: 12px; color: var(--text-muted); }}

.alert {{ padding: 12px 15px; border-radius: 8px; margin-bottom: 14px; border: 1px solid; font-size: 13.5px; }}
.alert-red {{ background: var(--red-bg); border-color: var(--red-border); }}
.alert-blue {{ background: var(--accent-soft); border-color: var(--accent); }}
.badge {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  border: 1px solid transparent;
}}
.b-red {{ background: var(--red-bg); color: var(--red); border-color: var(--red-border); }}
.b-amber {{ background: var(--amber-bg); color: var(--amber); border-color: var(--amber-border); }}
.b-green {{ background: var(--green-bg); color: var(--green); border-color: var(--green-border); }}
.b-blue {{ background: var(--accent-soft); color: var(--accent-dark); border-color: var(--accent); }}

.arabic-box {{
  background: var(--ar-bg);
  border-right: 4px solid var(--ar-border);
  border-radius: 7px;
  padding: 12px 15px;
  margin-top: 10px;
  direction: rtl;
  text-align: right;
  font-family: 'Segoe UI', 'Tahoma', Arial, sans-serif;
  line-height: 1.7;
  font-size: 13.8px;
}}
.arabic-box .ar-head {{ font-weight: 800; color: var(--accent-dark); margin-bottom: 4px; display: block; }}

.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.8px; }}
th, td {{ border-bottom: 1px solid var(--border); padding: 8px 9px; text-align: left; vertical-align: top; }}
th {{ background: var(--surface-alt); font-weight: 700; color: var(--text-muted); font-size: 11.5px; text-transform: uppercase; }}
tr:hover td {{ background: var(--surface-alt); }}

.accordion {{ border: 1px solid var(--border); border-radius: 9px; margin-bottom: 11px; background: var(--surface); overflow: hidden; box-shadow: var(--shadow); }}
.acc-head {{
  width: 100%;
  background: var(--surface);
  color: var(--text);
  border: none;
  padding: 14px 16px;
  text-align: left;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  font-size: 14.5px;
  font-weight: 700;
}}
.acc-head:hover {{ background: var(--surface-alt); }}
.acc-body {{ display: none; padding: 0 16px 16px 16px; border-top: 1px solid var(--border); }}
.accordion.open .acc-body {{ display: block; }}
.accordion.open .acc-chevron {{ transform: rotate(180deg); }}
.acc-chevron {{ transition: transform 0.2s ease; font-size: 14px; color: var(--text-muted); }}

.timeline {{ border-left: 3px solid var(--accent); padding-left: 16px; margin-left: 6px; }}
.tl-item {{ position: relative; margin-bottom: 14px; }}
.tl-item::before {{
  content: '';
  width: 11px; height: 11px;
  border-radius: 50%;
  background: var(--accent);
  border: 2px solid var(--surface);
  position: absolute;
  left: -23px; top: 4px;
}}
.tl-date {{ font-size: 11.5px; font-weight: 800; color: var(--accent-dark); }}

.gallery-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 9px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow);
}}
.gallery-card img {{
  width: 100%;
  height: 185px;
  object-fit: cover;
  background: #020617;
  cursor: zoom-in;
  border-bottom: 1px solid var(--border);
}}
.gallery-meta {{ padding: 10px 12px; font-size: 12.2px; flex: 1; display: flex; flex-direction: column; justify-content: space-between; gap: 8px; }}
.filter-bar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }}
.search-input {{
  padding: 7px 11px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  min-width: 240px;
  font-size: 13px;
}}
.modal {{
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(2, 6, 23, 0.88);
  z-index: 100;
  padding: 20px;
  overflow: auto;
  align-items: center;
  justify-content: center;
}}
.modal.open {{ display: flex; }}
.modal-box {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  max-width: 1180px;
  width: 100%;
  padding: 16px;
}}
.modal-img-wrap {{ max-height: 74vh; overflow: auto; background: #020617; border-radius: 6px; text-align: center; margin: 10px 0; }}
.modal-img-wrap img {{ max-width: 100%; transition: transform 0.2s ease; transform-origin: top center; }}

@media print {{
  @page {{ size: A4; margin: 12mm; }}
  body {{ background: #fff !important; color: #000 !important; font-size: 10.5pt; }}
  .sidebar, .topbar-actions, .filter-bar, .no-print, .modal {{ display: none !important; }}
  .topbar {{ position: static; background: #0f172a !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; margin-bottom: 12px; border-radius: 6px; }}
  .shell {{ display: block; margin: 0; padding: 0; max-width: 100%; }}
  body.print-all .tab-panel {{ display: block !important; page-break-after: always; }}
  body.print-all .accordion .acc-body {{ display: block !important; }}
  .card, .accordion, .gallery-card, .kpi, .video-suite-hero {{ box-shadow: none !important; break-inside: avoid; }}
  a {{ color: #0369a1 !important; }}
}}
</style>
</head>
<body>

<header class="topbar">
  <div class="topbar-inner">
    <div class="patient-title">
      <h1>MDT Clinical Command Dashboard — Amal Mokhtar Mohamed Ibrahim <span style="font-weight:400;opacity:0.9;">(آمال مختار محمد إبراهيم)</span></h1>
      <div class="patient-sub">
        <span><strong>Age/Sex:</strong> 63.5yo F</span>
        <span><strong>Verified DOB:</strong> 1963-04-09 (ID: <code>26304090300188</code>)</span>
        <span><strong>Audited Primary Files:</strong> {file_count}</span>
        <span><strong>Definitive Audiology:</strong> 2026-11-03</span>
        <span><strong>Holter / MRI:</strong> 2026-11-05</span>
      </div>
    </div>
    <div class="topbar-actions no-print">
      <button class="btn" id="btnTheme" onclick="toggleTheme()">🌙 Dark Mode</button>
      <button class="btn" id="btnLang" onclick="cycleLang()">🌐 EN + عربي</button>
      <button class="btn" onclick="printSection()">🖨️ Print Current Tab</button>
      <button class="btn btn-primary" onclick="printFullDossier()">📄 Print Full Dossier (PDF)</button>
    </div>
  </div>
</header>

<div class="shell">
  <aside class="sidebar no-print">
    <nav class="nav-card" aria-label="Dashboard sections">
      <div class="nav-label">Clinical Modules</div>
      <button class="nav-btn active" data-tab="tab-overview" onclick="switchTab('tab-overview', this)">
        <span>1. Video Suite &amp; Overview</span><span class="nav-count">3 Vid</span>
      </button>
      <button class="nav-btn" data-tab="tab-board" onclick="switchTab('tab-board', this)">
        <span>2. 6-Consultant Board</span><span class="nav-count">6</span>
      </button>
      <button class="nav-btn" data-tab="tab-dilemmas" onclick="switchTab('tab-dilemmas', this)">
        <span>3. 4 Dilemmas Resolved</span><span class="nav-count">4</span>
      </button>
      <button class="nav-btn" data-tab="tab-imaging" onclick="switchTab('tab-imaging', this)">
        <span>4. Files &amp; ROI Viewer</span><span class="nav-count">{file_count}</span>
      </button>
      <button class="nav-btn" data-tab="tab-treatment" onclick="switchTab('tab-treatment', this)">
        <span>5. Meds &amp; Action Plan</span><span class="nav-count">Rx</span>
      </button>
    </nav>

    <div class="nav-card pdf-links">
      <div class="nav-label">Shareable PDFs &amp; Dossiers</div>
      <a href="../05_Shareable_PDF_Reports/01_MDT_Executive_Summary_and_Action_Plan.pdf" target="_blank">📑 01. 2-Page Executive Brief (PDF)</a>
      <a href="../05_Shareable_PDF_Reports/02_Consultant_6_Board_Audit_Reports.pdf" target="_blank">🩺 02. 6-Consultant Board Audits (PDF)</a>
      <a href="../05_Shareable_PDF_Reports/03_Patient_Case_Study_Full_Report.pdf" target="_blank">📘 03. Full Case Study Monograph (PDF)</a>
      <a href="../04_Final_MDT_Deliverables/Master_MDT_Case_Study_Report.md" target="_blank">📝 Master Markdown Report</a>
      <a href="../01_Structured_and_Renamed_Inputs/FILE_RENAMING_INDEX.md" target="_blank">🗂️ Complete 37-File Renaming Index</a>
    </div>
  </aside>

  <main>

    <!-- ==================== TAB 1: VIDEO SUITE & OVERVIEW ==================== -->
    <section id="tab-overview" class="tab-panel active">

      <!-- TOP PATIENT SUMMARY CARDS (IMMEDIATELY ABOVE VIDEO SHOWCASE) -->
      <div class="grid-4" style="margin-bottom:16px;">
        <div class="kpi warn">
          <div class="lbl">MRV Sinus Asymmetry (L/R)</div>
          <div class="val">0.48 Ratio</div>
          <div class="sub">Attenuated Lt Transverse/Sigmoid/IJV; 0 clot</div>
        </div>
        <div class="kpi danger">
          <div class="lbl">2026-11-03 GSI-39 &amp; PTA</div>
          <div class="val">Lt 3.2 | Rt 1.5 cm³</div>
          <div class="sub"><strong>0 dB ABG</strong>; WRS 84%/76%; refutes 'Rt As'</div>
        </div>
        <div class="kpi danger">
          <div class="lbl">24h Holter (PVCs &amp; ST)</div>
          <div class="val">857 PVCs | -2.1 mm</div>
          <div class="sub"><strong>36.29% AM surge</strong>; CH3 ST @ 13:11; 5 mm bulb</div>
        </div>
        <div class="kpi warn">
          <div class="lbl">Vision &amp; Brain Parenchyma</div>
          <div class="val">&lt;1/60 | Fazekas 1–2</div>
          <div class="sub">CSVD + BG lacune; Lt silicone oil (2024 CT)</div>
        </div>
      </div>

      <!-- PROMINENT HERO NEUROIMAGING VIDEO SUITE IMMEDIATELY BELOW HEADER & PATIENT SUMMARY CARDS -->
      <section id="neuroimaging-video-suite" class="video-suite-hero" aria-label="Neuroimaging Video Suite (Interactive DICOM/MRI/MRV Viewer)">
        <div class="video-suite-head">
          <h2>🎥 Neuroimaging Video Suite (Interactive DICOM/MRI/MRV Viewer)</h2>
          <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
            <span class="badge b-blue">1.5T MRI / MRA / MRV (2026-11-05)</span>
            <span class="badge b-green">Relative Web Paths Ready</span>
          </div>
        </div>
        <div class="video-grid grid-cols-1 md:grid-cols-3 gap-6">

          <article class="video-card">
            <div class="video-frame-wrap">
              <video controls preload="metadata" playsinline
                class="w-full h-64 object-contain rounded-lg shadow-md bg-black"
                src="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4">
              </video>
            </div>
            <div class="video-card-body">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
                <h4>Card 1: 3D-MRV Venography Video</h4>
                <span class="badge b-amber">Derived Cine</span>
              </div>
              <p style="margin:0;color:var(--text-muted);">
                Highlighting the <strong>attenuated Left Transverse, Sigmoid Sinus &amp; IJV</strong> (L/R calibre ratio <code>0.48</code> vs dominant Right) across 18 Dark &amp; Inverted 3D-PCA MIP projections with <strong>zero dural thrombosis</strong> (compiled from primary <code>Film06</code>).
              </p>
              <div class="arabic-box lang-ar" style="margin-top:4px;padding:7px 9px;font-size:12.3px;" dir="rtl" lang="ar">
                فيديو الرنين الوريدي MRV بـ 18 زاوية: يبرز ضيق الجيب الوريدي المستعرض والسيني والوداجي الشمال (نسبة 0.48) مع خلوّه من أي جلطات وريدية.
              </div>
              <div class="video-actions">
                <a href="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4" download class="btn-download">⬇ Download / Open Video</a>
                <a href="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4" target="_blank" style="font-size:11.5px;font-weight:600;">Open Tab ↗</a>
              </div>
            </div>
          </article>

          <article class="video-card">
            <div class="video-frame-wrap">
              <video controls preload="metadata" playsinline
                class="w-full h-64 object-contain rounded-lg shadow-md bg-black"
                src="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil_WebH264.mp4">
              </video>
            </div>
            <div class="video-card-body">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
                <h4>Card 2: 3D-TOF MRA Video</h4>
                <span class="badge b-green">Primary 513-Slice Stack</span>
              </div>
              <p style="margin:0;color:var(--text-muted);">
                Showing <strong>unremarkable intracranial arteries</strong> &amp; patent Circle of Willis across 1.0-mm Sc 11 axial source slices (skull base to vertex), zero high-flow dAVF, and bright <strong>2.59x Lt intraocular silicone oil</strong>.
              </p>
              <div class="arabic-box lang-ar" style="margin-top:4px;padding:7px 9px;font-size:12.3px;" dir="rtl" lang="ar">
                فيديو الرنين الشرياني MRA (513 شريحة 1 مم): شرايين المخ سليمة تماماً بدون ناسور عالي التدفق مع ظهور زيت السيليكون الساطع بالعين الشمال.
              </div>
              <div class="video-actions">
                <a href="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil_WebH264.mp4" download class="btn-download">⬇ Download / Open Video</a>
                <a href="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil.mov" download style="font-size:11.5px;font-weight:600;">Raw .MOV ↗</a>
              </div>
            </div>
          </article>

          <article class="video-card">
            <div class="video-frame-wrap">
              <video controls preload="metadata" playsinline
                class="w-full h-64 object-contain rounded-lg shadow-md bg-black"
                src="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4">
              </video>
            </div>
            <div class="video-card-body">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
                <h4>Card 3: Axial T2/FLAIR MRI Video</h4>
                <span class="badge b-amber">Derived Cine</span>
              </div>
              <p style="margin:0;color:var(--text-muted);">
                Displaying <strong>Basal Ganglia lacunar infarcts</strong> (right lentiform) and <strong>CSVD small vessel disease (Fazekas 1–2)</strong> alongside ethmoiditis &amp; intact IACs across 20 synchronized slices (compiled from primary <code>Film09 + Film07</code>).
              </p>
              <div class="arabic-box lang-ar" style="margin-top:4px;padding:7px 9px;font-size:12.3px;" dir="rtl" lang="ar">
                فيديو الرنين T2/FLAIR عبر 20 مستوى: يوضح جلطة الـ Basal Ganglia اليمين وتصلب الشرايين الدقيقة CSVD والتهاب الجيوب الغربالية وزيت السيليكون.
              </div>
              <div class="video-actions">
                <a href="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4" download class="btn-download">⬇ Download / Open Video</a>
                <a href="../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4" target="_blank" style="font-size:11.5px;font-weight:600;">Open Tab ↗</a>
              </div>
            </div>
          </article>

        </div>
      </section>

      <div class="section-head">
        <h2>Critical Safety Alerts, Telemetry &amp; Clinical Overview</h2>
        <span class="badge b-blue">Primary-Source Verified Audit ({file_count} Files)</span>
      </div>

      <div class="alert alert-red">
        <strong>CRITICAL DISCOVERIES FROM PRIMARY DOCUMENTS (READ BEFORE TESTS OR DRUGS):</strong>
        <ul style="margin:6px 0 0 18px; padding:0;">
          <li><strong>Bilateral Legal Blindness (&lt; 1/60 Better Eye — MOH 2021) + Left Intraocular Silicone Oil (<code>2.59x</code> on MRI, Persistent Since <code>2024-07-27 CT</code>):</strong> Strips ~95% visual balance compensation ("4-Hit" Vertigo) and <strong>INVALIDATES light-tracked VNG subtests</strong>!</li>
          <li><strong>Catheter DSA Deferral Mandate:</strong> <code>5 mm</code> Left Carotid Bulb Plaque (IMT <code>0.13 cm</code>) + pending eGFR on Metformin + <code>-2.1 mm</code> silent myocardial ischemia. Complete bedside maneuvers &amp; non-invasive 4D-MRA/CTA gatekeeper first.</li>
          <li><strong>Holter–CSVD–Audiology Coupling &amp; Movement Reconciliation:</strong> <code>36.29%</code> of 857 PVCs (164/hr = <code>3.78%</code> &amp; <code>104 bpm</code> V-run @ <code>08:00 AM</code>) cluster on <code>80 IU/d</code> insulin; <code>2024-07-17 2h EEG (9 epochs)</code> excludes epilepsy while Dr. Galhom logged <code>Essential tremor + Titubation</code> + prescribed <code>Inderal 10 mg BID</code> (masks hypoglycemia!).</li>
        </ul>
      </div>

      <div class="grid-4" style="margin-bottom:14px;">
        <div class="kpi">
          <div class="lbl">2026-11-03 Masked PTA &amp; WRS</div>
          <div class="val">0 dB ABG | 84%/76%</div>
          <div class="sub">Rt 20–70 dB (3f 40.0), Lt 50–80 dB (3f 54.2); I 1kHz NT; Hx Metz 45 SL</div>
        </div>
        <div class="kpi warn">
          <div class="lbl">Carotid IMT &amp; Left Plaque</div>
          <div class="val">0.13 cm | 5 mm</div>
          <div class="sub">+44.4% IMT above 0.9 mm ULN; non-stenotic Lt bulb plaque</div>
        </div>
        <div class="kpi">
          <div class="lbl">Insulin &amp; Prolonged 2h EEG</div>
          <div class="val">80 IU/d | 9 Hz Alpha</div>
          <div class="sub">Toujeo 30U + Apidra 50U; 9 EEG epochs EMG tremor only (0 spikes)</div>
        </div>
        <div class="kpi ok">
          <div class="lbl">MS / Acute Stroke Exclusion</div>
          <div class="val">0 Dawson | DWI Neg</div>
          <div class="sub">Spared corpus callosum &amp; brainstem; ethmoid sinusitis active</div>
        </div>
      </div>

      <div class="card">
        <h3>Interactive 24-Hour Holter Circadian PVC Surge, HR &amp; ST Analyzer</h3>
        <p style="font-size:12.5px;color:var(--text-muted);margin:0 0 10px 0;">Click or hover any hour bar (<code>09:00</code> to <code>08:00</code>) to inspect beat counts, HR, ventricular pairs/runs, and ST deviations from Holter Page 2.</p>
        <div id="holterChartHost"></div>
        <div id="holterInspector" class="alert alert-blue" style="margin-top:10px;margin-bottom:0;font-size:12.5px;">
          <strong>Peak Hour Pre-Selected (08:00 AM):</strong> 4,339 beats | Avg HR 72 bpm (58–98) | <strong>164 PVCs (3.78% hourly burden)</strong> | 6 V-pairs | <strong>1 V-run (3 beats @ 104 bpm, 08:00:07)</strong> | SDNN 81 ms.
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <h3>Definitive Audiology Baseline (2026-11-03): GSI-39 Tymp &amp; Masked PTA</h3>
          <div id="audioChartHost"></div>
        </div>
        <div class="card">
          <h3>Ranked Multidisciplinary Diagnosis (Top 8)</h3>
          <ol style="margin:4px 0 0 18px;padding:0;font-size:12.7px;">
            <li><strong>Left Pulsatile Tinnitus &amp; Retroauricular Headache:</strong> Venous hum across attenuated Lt TS/Sigmoid/IJV (<code>0.48</code>) amplified by <strong>3.2 cm³ Extreme Type Ad drum</strong> + <code>36.29%</code> AM PVC beat irregularity.</li>
            <li><strong>Hypertensive-Diabetic CSVD (Fazekas 1–2) &amp; Chronic Lacunes</strong> (Rt BG, Lt frontal, Rt parietal); <strong>DWI negative, MS excluded</strong>.</li>
            <li><strong>Mixed Movement Disorder:</strong> Essential Tremor + Head Titubation (<code>2026-11-03</code>) + <strong>2024 2h EEG EMG tremor bursts (0 spikes)</strong> + BG lacune + Insulin Adrenergic Dips.</li>
            <li><strong>"4-Hit" Multi-Sensory Vertigo:</strong> Lt vestibulopathy + <strong>&lt;1/60 Legal Blindness</strong> + CSVD/Meclizine + Holter/glycemic dips.</li>
            <li><strong>Silent Myocardial Ischemia (<code>-2.1 mm</code> @ 13:11) &amp; Morning PVC Surge (857 PVCs / 4 V-runs)</strong>.</li>
            <li><strong>Carotid Atherosclerosis:</strong> <code>0.13 cm</code> IMT + <code>5 mm</code> Lt carotid bulb plaque.</li>
            <li><strong>Definitive Bilateral Cochlear SNHL (2026-11-03):</strong> <strong>0 dB ABG</strong>, Rt 20–70 dB (WRS 84%, Tymp 1.5 cm³ refutes 'Rt As'), Lt 50–80 dB (WRS 76%, Tymp 3.2 cm³ Type Ad).</li>
            <li><strong>Bilateral Ethmoidal Sinusitis + T2DM (80 IU/d) + HTN + Dyslipidemia</strong>.</li>
          </ol>
        </div>
      </div>

      <div class="card">
        <h3>Chronological Investigation &amp; Clinical Event Timeline</h3>
        <div class="timeline">
          <div class="tl-item"><span class="tl-date">2021-09-01 / 2021-09-19</span> — <strong>MOH Specialized Medical Councils Certificate:</strong> Legal blindness (<strong>&lt; 1/60</strong> better eye), retinal and optic nerve atrophy, ID <code>26304090300188</code> verifying DOB <code>1963-04-09</code>.</div>
          <div class="tl-item"><span class="tl-date">2024-07-17 &amp; 2024-07-27</span> — <strong>IMC 2h Prolonged EEG (Report + 9 Epochs) &amp; El-Zohor Non-Contrast CT:</strong> Awake 9 Hz alpha with frontal EMG tremor bursts (zero epileptiform waves); CT ordered for "tremors" confirmed age-related involutional changes and <strong>pre-existing left intraocular silicone oil</strong>.</div>
          <div class="tl-item"><span class="tl-date">2026-11-02 &amp; 2026-11-03</span> — <strong>Definitive Masked Audiology, ENT &amp; Neurosurgery:</strong> Dr. Shahenda's <strong>2026-11-03 Masked PTA &amp; GSI-39</strong> proved 0 dB ABG (Rt 20–70 dB, WRS 84%, Tymp <code>1.5 cm³</code> refuting handwritten <code>'Rt As'</code>; Lt 50–80 dB, WRS 76%, Tymp <code>3.2 cm³</code> Type Ad). Prof. Dr. Galhom logged <code>Essential tremor + Titubation</code> (<code>Inderal 10 BID + Ciprapra 10 + Cobal F + Brain-Ox</code>).</div>
          <div class="tl-item"><span class="tl-date">2026-11-05 &amp; 2026-11-07</span> — <strong>1.5T MRI/MRA/MRV (513 Source Slices), 24h Holter (29 pp) &amp; Carotid Duplex:</strong> CSVD Fazekas 1–2 + BG lacune; attenuated Lt TS/Sigmoid/IJV (0.48); Lt silicone oil (2.59x); ethmoiditis; 857 PVCs (36.29% AM surge); -2.1 mm ST @ 13:11; IMT 0.13 cm &amp; 5 mm Lt carotid bulb plaque.</div>
        </div>
      </div>

      <div class="arabic-box lang-ar" dir="rtl" lang="ar">
        <strong class="ar-head">الملخص التأسيسي السريع للحالة (37 ملفاً موثقاً — بالمصري العلمي)</strong>
        الحالة لمريضة عمرها <span dir="ltr">63.5</span> سنة تعاني من تداخل أربعة أجهزة: (1) <strong>فقدان بصر قانوني ثنائي (&lt; 1/60) وزيت سيليكون بالعين الشمال مؤكد منذ مقطعية 2024-07-27</strong>؛ (2) <strong>تصلب شرايين مخية دقيقة (<span dir="ltr">CSVD Fazekas 1–2</span>) وليس <span dir="ltr">MS</span></strong>، مع ضيق وريدي شمال (<span dir="ltr">L/R 0.48</span>) تضخمه <strong>طبلية 2026-11-03 المرنة (<span dir="ltr">3.2 cm³ Type Ad</span> مقابل <span dir="ltr">1.5 cm³ Type A</span> باليمين؛ <span dir="ltr">0 dB ABG</span>)</strong>؛ (3) <strong>رعشة مختلطة</strong> (<span dir="ltr">Essential Tremor + Titubation</span> + تخطيط مخ 2024 سليم بدون صرع + هبوط سكر محتمل)؛ (4) <strong>قصور تاجي صامت (<span dir="ltr">-2.1 mm ST</span>) وطفرة صباحية للـ <span dir="ltr">PVCs</span> (<span dir="ltr">36.29%</span> بين 6–9 ص)</strong> وبلاك <span dir="ltr">5 mm</span> بالسباتي الشمال يوجب تأجيل القسطرة <span dir="ltr">DSA</span>.
      </div>
    </section>

    <!-- ==================== TAB 2: 6-CONSULTANT BOARD PANEL ==================== -->
    <section id="tab-board" class="tab-panel">
      <div class="section-head">
        <h2>The 6-Consultant Multidisciplinary Medical Board Panel</h2>
        <div style="display:flex;gap:8px;">
          <button class="btn btn-surface" onclick="setAllAccordions(true)">Expand All 6 Audits</button>
          <button class="btn btn-surface" onclick="setAllAccordions(false)">Collapse All</button>
        </div>
      </div>
      <div id="consultantsContainer"></div>
    </section>

    <!-- ==================== TAB 3: CORE CLINICAL DILEMMAS ==================== -->
    <section id="tab-dilemmas" class="tab-panel">
      <div class="section-head">
        <h2>The 4 Core Clinical Dilemmas — Resolved Side-by-Side</h2>
        <span class="badge b-green">Consensus Evidence Matrix</span>
      </div>
      <div id="dilemmasContainer"></div>
    </section>

    <!-- ==================== TAB 4: DIAGNOSTIC & IMAGING VIEWER INDEX ==================== -->
    <section id="tab-imaging" class="tab-panel">
      <div class="section-head">
        <h2>CLAHE / ROI Micro-Zoom Gallery &amp; Complete {file_count}-File Renamed Index</h2>
        <span class="badge b-blue">513 Deduplicated 3D TOF Slices + All {file_count} Primary Files</span>
      </div>

      <div class="card">
        <h3>Extracted &amp; CLAHE-Enhanced Clinical / Neuroimaging Gallery (Click Image to Zoom)</h3>
        <div class="grid-3" id="roiGallery"></div>
      </div>

      <div class="card">
        <h3>Structured File Renaming Lookup Table (All {file_count} Workspace Raw Files)</h3>
        <div class="filter-bar">
          <input type="text" class="search-input" id="fileSearch" placeholder="Search raw filename, new semantic name, modality, or finding..." oninput="renderFileTable()">
          <button class="btn btn-surface btn-active" onclick="filterFiles('ALL', this)">All ({file_count})</button>
          <button class="btn btn-surface" onclick="filterFiles('RAD_MRI_2026', this)">2026 MRI/MRA/MRV (13)</button>
          <button class="btn btn-surface" onclick="filterFiles('AUD_2026', this)">2026 Audiology (2)</button>
          <button class="btn btn-surface" onclick="filterFiles('CARD', this)">Holter ECG (5)</button>
          <button class="btn btn-surface" onclick="filterFiles('RAD_CT_2024', this)">2024 CT Brain (2)</button>
          <button class="btn btn-surface" onclick="filterFiles('NEURO_EEG_2024', this)">2024 2h EEG (10)</button>
          <button class="btn btn-surface" onclick="filterFiles('VASC', this)">Duplex (1)</button>
          <button class="btn btn-surface" onclick="filterFiles('CLIN', this)">Clinical Notes (4)</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>#</th><th>Cat</th><th>Date</th><th>Original Raw Name</th><th>New Semantic File (Click to Open)</th><th>Size</th><th>One-Line Clinical Summary</th></tr>
            </thead>
            <tbody id="fileTableBody"></tbody>
          </table>
        </div>
        <p style="font-size:12px;color:var(--text-muted);margin-top:8px;">
          <em>Inventory Transparency Note:</em> Historical documented records without raw image scans on disk (<code>2024-11-02 Al-Hayah Unmasked Audiology</code> superseded by 2026-11-03 primary scans, <code>2026-11-02 EHA Note</code>, <code>2026-11-03 ENT Rx</code>) remain logged separately in <a href="../01_Structured_and_Renamed_Inputs/FILE_RENAMING_INDEX.md" target="_blank">FILE_RENAMING_INDEX.md</a>.
        </p>
      </div>
    </section>

    <!-- ==================== TAB 5: MASTER TREATMENT & PRESCRIPTION PLAN ==================== -->
    <section id="tab-treatment" class="tab-panel">
      <div class="section-head">
        <h2>Master Pharmacotherapy Audit, Hazards, Modified Regimen &amp; Action Checklist</h2>
        <button class="btn btn-surface" onclick="resetChecklist()">Reset Checkboxes</button>
      </div>

      <div class="card">
        <h3>1. Drug-Interaction &amp; Risk Audit (Current + Newly Prescribed Medications)</h3>
        <div class="filter-bar">
          <button class="btn btn-surface btn-active" onclick="filterMeds('ALL', this)">All (7)</button>
          <button class="btn btn-surface" onclick="filterMeds('STOP', this)">Stop / Contraindicated</button>
          <button class="btn btn-surface" onclick="filterMeds('CAUTION', this)">High Caution / Switch</button>
          <button class="btn btn-surface" onclick="filterMeds('UPGRADE', this)">Upgrade / Optimize</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Medication / Combo</th><th>Action Tag</th><th>Pharmacological Mechanism</th><th>Specific Patient Hazard</th><th>MDT Resolution</th></tr></thead>
            <tbody id="medAuditBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h3>2. Complete Modified Prescription &amp; Rehabilitation Table</h3>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Clinical Domain</th><th>Modified Regimen / Intervention</th><th>Dose &amp; Timing</th><th>Clinical Rationale &amp; Safety Guardrails</th></tr></thead>
            <tbody id="rxBody"></tbody>
          </table>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <h3>3. Interactive Phase 1–2 Diagnostic &amp; Clinical Action Checklist</h3>
          <div id="checklistHost" style="font-size:13px;"></div>
        </div>
        <div class="card">
          <h3>4. Precision Bilateral Hearing Aid Spec (For &lt; 1/60 Blindness)</h3>
          <ul>
            <li><strong>Bilateral Mandatory:</strong> Binaural spatial hearing is her primary environmental orientation sense (&lt; 1/60 vision).</li>
            <li><strong>Hardware:</strong> Bilateral <strong>Receiver-In-Canal (RIC) Digital</strong> with <strong>soft vented/open domes</strong> (avoid occluded molds so internal vascular hum isn't trapped!).</li>
            <li><strong>Compression &amp; Output Limit:</strong> <strong>WDRC</strong> with strict <strong>OSPL90 &lt;= 95–100 dB SPL</strong> on Left Ear protecting against cochlear loudness sensitivity.</li>
            <li><strong>VNG Technical Warning:</strong> Instruct VNG lab to skip visual tracking/OPK and perform <strong>dark positional + bithermal calorics only</strong>.</li>
          </ul>
          <div class="arabic-box lang-ar" dir="rtl" lang="ar">
            <strong class="ar-head">خلاصة تعديل العلاج والتوصيات العملية</strong>
            1. <strong>أوقف فوراً:</strong> <bdi dir="ltr">Meclizine</bdi> و <bdi dir="ltr">Cetrovit Ginkgo</bdi> و <bdi dir="ltr">Brain-Ox</bdi> وامنع <bdi dir="ltr">Cidamex</bdi>.<br>
            2. <strong>تحاليل ومراقبة:</strong> <bdi dir="ltr">K+, Mg2+, Cr/eGFR, HbA1c, B12, Lipids</bdi> + سنسور سكر (<bdi dir="ltr">CGM</bdi>) لكشف هبوط السكر بين 6–9 صباحاً.<br>
            3. <strong>تعديل القلب:</strong> إيكو + إيكو دوائي، تبديل <bdi dir="ltr">Angiopress Comp</bdi> بـ <bdi dir="ltr">Perindopril/Telmisartan + Bisoprolol</bdi> ورفع <bdi dir="ltr">Atorvastatin</bdi> إلى 40–80 مجم.<br>
            4. <strong>الأذن والجيوب:</strong> بخاخ <bdi dir="ltr">Mometasone</bdi>، رفع الرأس 30°، <bdi dir="ltr">VNG</bdi> بالضلمة فقط، تأهيل اتزان لا بصري، وسماعتين <bdi dir="ltr">Open-Fit WDRC RIC</bdi> بحد أقصى <bdi dir="ltr">&lt;= 100 dB</bdi>.
          </div>
        </div>
      </div>
    </section>

  </main>
</div>

<div class="modal" id="imgModal" onclick="if(event.target===this) closeModal()">
  <div class="modal-box">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">
      <strong id="modalTitle" style="font-size:15px;">Image Inspector</strong>
      <div style="display:flex;gap:6px;">
        <button class="btn btn-surface" onclick="setZoom(1)">100%</button>
        <button class="btn btn-surface" onclick="setZoom(1.6)">160%</button>
        <button class="btn btn-surface" onclick="setZoom(2.4)">240%</button>
        <a class="btn btn-primary" id="modalRawLink" target="_blank">Open Original ↗</a>
        <button class="btn btn-surface" onclick="closeModal()">✕ Close</button>
      </div>
    </div>
    <div class="modal-img-wrap"><img id="modalImg" src="" alt="ROI Preview"></div>
    <div id="modalCaption" style="font-size:12.5px;color:var(--text-muted);"></div>
  </div>
</div>

<script>
const MDT_DATA = {data_json};
const HOLTER_HOURS = MDT_DATA.holter_hours;
const CONSULTANTS = MDT_DATA.consultants;
const DILEMMAS = MDT_DATA.dilemmas;
const ROI_CARDS = MDT_DATA.roi_cards;
const FILES_INDEX = MDT_DATA.files_index;
const MED_AUDIT = MDT_DATA.med_audit;
const RX_TABLE = MDT_DATA.rx_table;
const CHECKLIST_ITEMS = MDT_DATA.checklist;

let activeFileFilter = "ALL";
let activeMedFilter = "ALL";

function renderHolterSVG() {{
  const W = 760, H = 225, padL = 44, padR = 44, padT = 26, padB = 34;
  const cw = W - padL - padR, ch = H - padT - padB;
  const bw = cw / HOLTER_HOURS.length;
  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:auto;display:block;background:var(--surface-alt);border:1px solid var(--border);border-radius:7px;">`;
  svg += `<rect x="${{padL + 21*bw}}" y="${{padT}}" width="${{3*bw}}" height="${{ch}}" fill="rgba(220,38,38,0.11)"/>`;
  svg += `<text x="${{padL + 22.5*bw}}" y="${{padT + 12}}" text-anchor="middle" font-size="9" font-weight="700" fill="#dc2626">06–09 AM Surge (36.29% PVCs)</text>`;
  [0, 50, 100, 150].forEach(v => {{
    const y = padT + ch - (v / 180) * ch;
    svg += `<line x1="${{padL}}" y1="${{y}}" x2="${{W-padR}}" y2="${{y}}" stroke="var(--border)" stroke-dasharray="3 3"/>`;
    svg += `<text x="${{padL-5}}" y="${{y+3}}" text-anchor="end" font-size="9" fill="var(--text-muted)">${{v}}</text>`;
  }});
  let hrPts = [];
  HOLTER_HOURS.forEach((d, i) => {{
    const x = padL + i * bw + 3;
    const w = bw - 6;
    const bh = (d.pvc / 180) * ch;
    const y = padT + ch - bh;
    const isPeak = (i >= 21);
    const isST = (d.h === "13:00");
    const col = isST ? "#d97706" : (isPeak ? "#dc2626" : "#2563eb");
    svg += `<rect x="${{x}}" y="${{y}}" width="${{w}}" height="${{bh}}" rx="2" fill="${{col}}" style="cursor:pointer" onmouseover="inspectHolter(${{i}})" onclick="inspectHolter(${{i}})" />`;
    if (d.pvc >= 45) svg += `<text x="${{x+w/2}}" y="${{y-3}}" text-anchor="middle" font-size="8.5" font-weight="700" fill="${{col}}">${{d.pvc}}</text>`;
    if (i % 2 === 0 || i === 23) svg += `<text x="${{x+w/2}}" y="${{H-16}}" text-anchor="middle" font-size="8.5" fill="var(--text-muted)">${{d.h.slice(0,2)}}h</text>`;
    const hy = padT + ch - ((d.hr - 50) / 70) * ch;
    hrPts.push(`${{(x+w/2).toFixed(1)}},${{hy.toFixed(1)}}`);
  }});
  svg += `<polyline fill="none" stroke="var(--text)" stroke-width="2" points="${{hrPts.join(" ")}}"/>`;
  hrPts.forEach(p => {{
    const [px, py] = p.split(",");
    svg += `<circle cx="${{px}}" cy="${{py}}" r="2.6" fill="var(--text)"/>`;
  }});
  svg += `<text x="${{W/2}}" y="${{H-3}}" text-anchor="middle" font-size="8.5" fill="var(--text-muted)">Bars: Hourly PVCs (Amber = 13:00 ST -2.1mm | Red = 06–08 AM Surge) | Black Line: Peak Hourly HR</text>`;
  svg += `</svg>`;
  document.getElementById("holterChartHost").innerHTML = svg;
}}

function inspectHolter(i) {{
  const d = HOLTER_HOURS[i];
  const burden = ((d.pvc / d.beats) * 100).toFixed(2);
  document.getElementById("holterInspector").innerHTML =
    `<strong>Hour ${{d.h}}:</strong> ${{d.beats.toLocaleString()}} beats | Peak HR ${{d.hr}} bpm | <strong>${{d.pvc}} PVCs (${{burden}}% hourly burden)</strong> | ${{d.vp}} V-pairs | ${{d.vr}} V-runs | Max ST Δ: <code>${{d.st}}</code>`;
}}

function renderAudioSVG() {{
  document.getElementById("audioChartHost").innerHTML = `
  <svg viewBox="0 0 460 225" style="width:100%;height:auto;display:block;background:var(--surface-alt);border:1px solid var(--border);border-radius:7px;">
    <text x="230" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="var(--text)">2026-11-03 GSI-39 Peak Compliance &amp; Masked PTA (0 dB ABG)</text>
    <line x1="55" y1="36" x2="55" y2="150" stroke="var(--border)"/>
    <line x1="55" y1="150" x2="255" y2="150" stroke="var(--border)"/>
    <line x1="55" y1="54" x2="255" y2="54" stroke="#dc2626" stroke-dasharray="4 3"/>
    <text x="50" y="57" text-anchor="end" font-size="8.5" fill="#dc2626">3.0 cm³</text>
    <text x="50" y="102" text-anchor="end" font-size="8.5" fill="var(--text-muted)">1.5 cm³</text>
    <text x="50" y="153" text-anchor="end" font-size="8.5" fill="var(--text-muted)">0.0</text>
    <rect x="75" y="102" width="52" height="48" fill="#2563eb" rx="3"/>
    <text x="101" y="96" text-anchor="middle" font-size="9" font-weight="700" fill="#2563eb">1.5 Type A</text>
    <text x="101" y="129" text-anchor="middle" font-size="7.5" font-weight="600" fill="#ffffff">Refutes 'Rt As'</text>
    <text x="101" y="166" text-anchor="middle" font-size="9" font-weight="700" fill="var(--text)">Right Peak</text>
    <rect x="170" y="48" width="52" height="102" fill="#dc2626" rx="3"/>
    <text x="196" y="42" text-anchor="middle" font-size="9" font-weight="700" fill="#dc2626">3.2 Type Ad!</text>
    <text x="196" y="98" text-anchor="middle" font-size="8.5" font-weight="700" fill="#ffffff">2.13x Ratio</text>
    <text x="196" y="166" text-anchor="middle" font-size="9" font-weight="700" fill="var(--text)">Left Peak</text>
    <rect x="272" y="34" width="176" height="122" rx="6" fill="var(--surface)" stroke="var(--border)"/>
    <text x="282" y="51" font-size="9" font-weight="700" fill="var(--accent-dark)">Definitive 2026-11-03 Summary:</text>
    <text x="282" y="68" font-size="8.2" fill="var(--text)">• ABG: 0 dB (Pure Cochlear SNHL)</text>
    <text x="282" y="83" font-size="8.2" fill="var(--text)">• Rt: 20–70 dB (3f 40.0) | WRS 84%</text>
    <text x="282" y="98" font-size="8.2" fill="var(--text)">• Lt: 50–80 dB (3f 54.2) | WRS 76%</text>
    <text x="282" y="114" font-size="8.2" font-weight="700" fill="#dc2626">• GSI-39: I 1000 Hz = NT bilaterally</text>
    <text x="282" y="129" font-size="8" fill="var(--text-muted)">• Lt 3.2 cm³ exceeds 3.0 graph ceiling</text>
    <text x="282" y="144" font-size="8" font-weight="600" fill="#16a34a">• RIC Open-Fit (OSPL90 &lt;= 100 dB)</text>
    <text x="230" y="196" text-anchor="middle" font-size="8.2" fill="var(--text-muted)">Replaces 2024 unverified notes: flaccid Lt TM amplifies 0.48 venous hum &amp; AM PVCs</text>
  </svg>`;
}}

function renderConsultants() {{
  document.getElementById("consultantsContainer").innerHTML = CONSULTANTS.map((c, idx) => `
    <div class="accordion ${{idx === 0 ? 'open' : ''}}">
      <button class="acc-head" onclick="this.parentElement.classList.toggle('open')">
        <span>${{c.role}}</span>
        <span style="display:flex;align-items:center;gap:8px;">
          <span class="badge ${{c.badge}}">${{c.verdict.slice(0, 58)}}${{c.verdict.length > 58 ? '…' : ''}}</span>
          <span class="acc-chevron">▼</span>
        </span>
      </button>
      <div class="acc-body">
        <p style="margin:10px 0 8px 0;"><strong>Board Verdict:</strong> <span class="badge ${{c.badge}}">${{c.verdict}}</span></p>
        <ul class="lang-en" style="margin:6px 0 10px 18px;padding:0;">
          ${{c.findings.map(f => `<li style="margin-bottom:5px;">${{f}}</li>`).join("")}}
        </ul>
        <div class="arabic-box lang-ar" dir="rtl" lang="ar">
          <strong class="ar-head">التحليل الطبي التخصصي (بالمصري العلمي):</strong>${{c.ar}}
        </div>
      </div>
    </div>
  `).join("");
}}

function renderDilemmas() {{
  document.getElementById("dilemmasContainer").innerHTML = DILEMMAS.map(d => `
    <div class="card">
      <h3>${{d.title}}</h3>
      <div class="grid-2">
        <div style="background:var(--red-bg);border:1px solid var(--red-border);border-radius:7px;padding:11px;">
          <div style="font-size:11.5px;font-weight:800;color:var(--red);text-transform:uppercase;margin-bottom:4px;">${{d.left_lbl}}</div>
          <div style="font-size:13px;">${{d.left}}</div>
        </div>
        <div style="background:var(--green-bg);border:1px solid var(--green-border);border-radius:7px;padding:11px;">
          <div style="font-size:11.5px;font-weight:800;color:var(--green);text-transform:uppercase;margin-bottom:4px;">${{d.right_lbl}}</div>
          <div style="font-size:13px;">${{d.right}}</div>
        </div>
      </div>
    </div>
  `).join("");
}}

function renderGallery() {{
  document.getElementById("roiGallery").innerHTML = ROI_CARDS.map((r, i) => `
    <div class="gallery-card">
      <img src="${{r.img}}" alt="${{r.title}}" onclick="openModal(${{i}})" loading="lazy">
      <div class="gallery-meta">
        <div>
          <strong style="font-size:13px;display:block;margin-bottom:3px;">${{r.title}}</strong>
          <span style="color:var(--text-muted);">${{r.desc}}</span>
        </div>
        <div style="display:flex;gap:6px;">
          <button class="btn btn-surface" onclick="openModal(${{i}})">🔍 Zoom Inspect</button>
          <a class="btn btn-surface" href="${{r.img}}" target="_blank">Open ↗</a>
        </div>
      </div>
    </div>
  `).join("");
}}

function renderFileTable() {{
  const q = (document.getElementById("fileSearch").value || "").toLowerCase();
  const rows = FILES_INDEX.filter(f => {{
    const catOk = (activeFileFilter === "ALL") || (f.grp === activeFileFilter);
    const txtOk = !q || [f.cat, f.date, f.raw, f.newName, f.summary].some(s => s.toLowerCase().includes(q));
    return catOk && txtOk;
  }});
  document.getElementById("fileTableBody").innerHTML = rows.map((f, idx) => `
    <tr>
      <td>${{idx+1}}</td>
      <td><span class="badge b-blue">${{f.cat}}</span></td>
      <td><code>${{f.date}}</code></td>
      <td><code style="word-break:break-all;">${{f.raw}}</code></td>
      <td><a href="${{f.href}}" target="_blank" style="font-weight:600;word-break:break-all;">${{f.newName}}</a></td>
      <td style="white-space:nowrap;">${{f.size}}</td>
      <td>${{f.summary}}</td>
    </tr>
  `).join("");
}}

function renderMeds() {{
  const auditRows = MED_AUDIT.filter(m => activeMedFilter === "ALL" || m.tag === activeMedFilter);
  document.getElementById("medAuditBody").innerHTML = auditRows.map(m => `
    <tr>
      <td><strong>${{m.med}}</strong></td>
      <td><span class="badge ${{m.badge}}">${{m.tag}}</span></td>
      <td>${{m.mech}}</td>
      <td>${{m.hazard}}</td>
      <td>${{m.action}}</td>
    </tr>
  `).join("");
  document.getElementById("rxBody").innerHTML = RX_TABLE.map(r => `
    <tr>
      <td><strong>${{r.dom}}</strong></td>
      <td><strong>${{r.reg}}</strong></td>
      <td><code>${{r.dose}}</code></td>
      <td>${{r.note}}</td>
    </tr>
  `).join("");
}}

function renderChecklist() {{
  const saved = JSON.parse(localStorage.getItem("mdt_checklist_v2") || "{{}}");
  document.getElementById("checklistHost").innerHTML = CHECKLIST_ITEMS.map(c => `
    <label style="display:flex;gap:9px;align-items:flex-start;padding:7px 8px;border-bottom:1px solid var(--border);cursor:pointer;">
      <input type="checkbox" ${{saved[c.id] ? "checked" : ""}} onchange="toggleCheck('${{c.id}}', this.checked)" style="margin-top:3px;transform:scale(1.15);">
      <span style="${{saved[c.id] ? 'text-decoration:line-through;color:var(--text-muted);' : ''}}">${{c.txt}}</span>
    </label>
  `).join("");
}}

function switchTab(id, btn) {{
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}}
function filterFiles(grp, btn) {{
  activeFileFilter = grp;
  btn.parentElement.querySelectorAll(".btn").forEach(b => b.classList.remove("btn-active"));
  btn.classList.add("btn-active");
  renderFileTable();
}}
function filterMeds(cat, btn) {{
  activeMedFilter = cat;
  btn.parentElement.querySelectorAll(".btn").forEach(b => b.classList.remove("btn-active"));
  btn.classList.add("btn-active");
  renderMeds();
}}
function setAllAccordions(openState) {{
  document.querySelectorAll("#consultantsContainer .accordion").forEach(a => a.classList.toggle("open", openState));
}}
function toggleTheme() {{
  const root = document.documentElement;
  const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  localStorage.setItem("mdt_theme", next);
  document.getElementById("btnTheme").textContent = next === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
}}
function cycleLang() {{
  const root = document.documentElement;
  const order = ["both", "en", "ar"];
  const labels = {{both:"🌐 EN + عربي", en:"🌐 English Only", ar:"🌐 عربي فقط"}};
  const cur = root.getAttribute("data-lang") || "both";
  const nxt = order[(order.indexOf(cur)+1) % order.length];
  root.setAttribute("data-lang", nxt);
  document.getElementById("btnLang").textContent = labels[nxt];
}}
function toggleCheck(id, val) {{
  const saved = JSON.parse(localStorage.getItem("mdt_checklist_v2") || "{{}}");
  saved[id] = val;
  localStorage.setItem("mdt_checklist_v2", JSON.stringify(saved));
  renderChecklist();
}}
function resetChecklist() {{
  localStorage.removeItem("mdt_checklist_v2");
  renderChecklist();
}}
function printSection() {{
  document.body.classList.remove("print-all");
  window.print();
}}
function printFullDossier() {{
  document.body.classList.add("print-all");
  window.print();
  setTimeout(() => document.body.classList.remove("print-all"), 500);
}}
function openModal(idx) {{
  const r = ROI_CARDS[idx];
  document.getElementById("modalTitle").textContent = r.title;
  document.getElementById("modalImg").src = r.img;
  document.getElementById("modalRawLink").href = r.img;
  document.getElementById("modalCaption").textContent = r.desc;
  setZoom(1);
  document.getElementById("imgModal").classList.add("open");
}}
function closeModal() {{ document.getElementById("imgModal").classList.remove("open"); }}
function setZoom(scale) {{ document.getElementById("modalImg").style.transform = `scale(${{scale}})`; }}

(function init() {{
  const t = localStorage.getItem("mdt_theme");
  if (t) {{
    document.documentElement.setAttribute("data-theme", t);
    document.getElementById("btnTheme").textContent = t === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
  }}
  renderHolterSVG();
  renderAudioSVG();
  renderConsultants();
  renderDilemmas();
  renderGallery();
  renderFileTable();
  renderMeds();
  renderChecklist();
}})();
</script>
</body>
</html>
"""


def write_root_redirect() -> None:
    """Create root index.html redirecting cleanly to 06_Clinical_Dashboard/index.html for web hosting."""
    html_text = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=./06_Clinical_Dashboard/index.html">
<title>Redirecting to MDT Clinical Dashboard...</title>
<script>window.location.replace("./06_Clinical_Dashboard/index.html");</script>
</head>
<body style="font-family:Segoe UI,Arial,sans-serif;background:#0f172a;color:#f1f5f9;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;">
  <div style="text-align:center;padding:28px;border:1px solid #334155;border-radius:12px;background:#111827;max-width:520px;">
    <h1 style="margin-top:0;font-size:20px;color:#38bdf8;">Neurovascular &amp; Neuro-Otological MDT Clinical Dashboard</h1>
    <p style="color:#cbd5e1;">Opening interactive clinical dashboard &amp; Neuroimaging Video Suite...</p>
    <p><a href="./06_Clinical_Dashboard/index.html" style="display:inline-block;padding:10px 18px;background:#0284c7;color:#fff;border-radius:8px;text-decoration:none;font-weight:700;">Open Clinical Dashboard →</a></p>
  </div>
</body>
</html>
"""
    ROOT_REDIRECT_HTML.write_text(html_text, encoding="utf-8")


def main() -> None:
    ffmpeg_bin = find_ffmpeg()
    print(f"Using ffmpeg: {ffmpeg_bin}")
    video_info = build_neuroimaging_videos(ffmpeg_bin)
    for k, info in video_info.items():
        print(f"[VIDEO OK] {k}: {info['path'].name} ({info['size']:,} bytes)")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    quant = json.loads(QUANT_PATH.read_text(encoding="utf-8"))
    data_json = build_js_data_payload(manifest, quant)
    file_count = len(manifest.get("files", []))

    DASH_DIR.mkdir(parents=True, exist_ok=True)
    html_out = render_dashboard_html(data_json, file_count)
    DASH_HTML.write_text(html_out, encoding="utf-8")
    write_root_redirect()
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"[DASHBOARD OK] {DASH_HTML} ({DASH_HTML.stat().st_size:,} bytes, {file_count} manifest files)")


if __name__ == "__main__":
    main()
