"""Autonomous MDT Clinical Dashboard & Neuroimaging Video Suite Builder.

1. Compiles/verifies browser-ready MP4 (H.264/yuv420p +faststart) neuroimaging videos:
   - Video 01 (WebH264): Lossless faststart MP4 remux of raw 3D-TOF MRA IMG_8048.MOV
   - Video 02 (Derived Cine): Synchronized 20-level Axial eT2W + eFLAIR sweep
     compiled from upright primary sheets (Film09 & Film07) highlighting
     Basal Ganglia lacunar infarcts, CSVD, silicone oil, IACs & ethmoiditis.
   - Video 03 (Derived Cine): 18-projection rotational 3D-PCA MRV venography
     sweep (Dark + Inverted Light MIP) compiled from primary Film06 highlighting
     Right-dominant venous drainage & attenuated Lt Transverse/Sigmoid/IJV (0.48).
2. Reads case_manifest.json (37 primary files) and clinical_quantification_results.json
   to build 06_Clinical_Dashboard/index.html featuring:
   - Strict Bilingual Architecture (100% Medical English default + one-click Egyptian Arabic)
   - Sequential 6-Tab Case-Study Workspace with Glassmorphism UI & Dark/Light Modes
   - Prominent Neuroimaging Video Suite (MRI, MRA, MRV) with relative streaming paths
   - Complete Demographics, Timeline, Medications, Prior Conflicts, 6-Consultant Audits,
     8 Ranked Diagnoses, 4 Resolved Dilemmas, Dynamic Telemetry SVGs, 11 ROI Plates,
     Searchable 37-File Semantic Directory, and Phase 1-2 Treatment & Action Checklist.
"""

from __future__ import annotations

import argparse
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


def find_ffmpeg() -> Path | None:
    """Locate ffmpeg executable on PATH or standard WinGet install location."""
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    winget_glob = list(
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet")).rglob("ffmpeg.exe")
    )
    if winget_glob:
        return winget_glob[0]
    return None


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
        last_idx = 0
        for idx, (frame, dur) in enumerate(frames_with_durations):
            last_idx = idx
            img_path = tmp / f"f_{idx:03d}.png"
            cv2.imwrite(str(img_path), frame)
            concat_lines.append(f"file '{img_path.as_posix()}'")
            concat_lines.append(f"duration {dur:.3f}")
        concat_lines.append(f"file '{(tmp / f'f_{last_idx:03d}.png').as_posix()}'")
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


def build_neuroimaging_videos(ffmpeg_bin: Path | None, force: bool = False) -> dict[str, dict[str, object]]:
    """Verify or compile the 3 streaming MP4 videos inside 01_Neuroimaging_MRI_MRA_MRV_Videos/."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    src_mov = VIDEOS_DIR / "MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil.mov"
    mra_mp4 = VIDEOS_DIR / "MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil_WebH264.mp4"
    t2_flair_mp4 = VIDEOS_DIR / "MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4"
    mrv_mp4 = VIDEOS_DIR / "MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4"

    all_ready = (
        mra_mp4.exists()
        and mra_mp4.stat().st_size > 100_000
        and t2_flair_mp4.exists()
        and t2_flair_mp4.stat().st_size > 100_000
        and mrv_mp4.exists()
        and mrv_mp4.stat().st_size > 100_000
    )
    if all_ready and not force:
        return {
            "mra": {"path": mra_mp4, "raw_mov": src_mov, "size": mra_mp4.stat().st_size, "rebuilt": False},
            "t2_flair": {"path": t2_flair_mp4, "size": t2_flair_mp4.stat().st_size, "rebuilt": False},
            "mrv": {"path": mrv_mp4, "size": mrv_mp4.stat().st_size, "rebuilt": False},
        }

    if ffmpeg_bin is None:
        raise FileNotFoundError("ffmpeg executable not found on PATH or WinGet directory.")

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

    encode_frames_to_mp4(flair_frames, t2_flair_mp4, ffmpeg_bin)

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

    encode_frames_to_mp4(mrv_frames, mrv_mp4, ffmpeg_bin)

    return {
        "mra": {"path": mra_mp4, "raw_mov": src_mov, "size": mra_mp4.stat().st_size, "rebuilt": True},
        "t2_flair": {"path": t2_flair_mp4, "size": t2_flair_mp4.stat().st_size, "rebuilt": True},
        "mrv": {"path": mrv_mp4, "size": mrv_mp4.stat().st_size, "rebuilt": True},
    }


FILE_SUMMARIES_AR = {
    "RAD_2026-11-05_MRI_MRA_MRV_Brain_Report_PortSaeedMilitary_DrGhadaAwad.pdf": "التقرير الرسمي للرنين الشامل (مستشفى بورسعيد العسكري / رولوجي - د. غادة عوض): تغيرات تراجعية، تصلب شرايين دقيقة وجلطات قديمة (بالعقد القاعدية اليمين والمادة البيضاء)، التهاب الجيوب الغربالية، شرايين سليمة بـ MRA، وضيق الجيب الوريدي الشمال في MRV.",
    "MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil.mov": "فيديو الأشعة الأصلي (28.1 ثانية) يعرض نحو 237 مقطعاً بسمك 1 مم لرنين الشرايين 3D-TOF من قاع الجمجمة لقمة الرأس، ويوضح سلامة الشرايين وساطع زيت السيليكون بالعين الشمال.",
    "MRI_2026-11-05_Film01_Sagittal_T1W_SE_Brain_Midline_NormalBrainstem.jpg": "فيلم الرنين السهمي T1W: سلامة الجسم الثفني والغدة النخامية وجذع المخ والمخيخ مع بروز طفيف بالتلافيف مناسب للعمر.",
    "MRA_2026-11-05_Film02_3D_TOF_Coronal_MIP_Intracranial_Arteries_Unremarkable.jpg": "فيلم MRA أمامي (24 صورة عادية ومعكوسة): سلامة الشرايين السباتية والوسطى والأمامية والخلفية والقاعدية بدون تمددات أو انسداد.",
    "MRA_2026-11-05_Film03_3D_TOF_Axial_MIP_CircleOfWillis_Unremarkable.jpg": "فيلم MRA محوري لدائرة ويليس (تظهر في الالتقاط الأصلي بزاوية 90 درجة): سلامة الوصلات الشريانية والمجرى القاعدي الفقري.",
    "MRA_2026-11-05_Film04a_3D_TOF_SagittalOblique_MIP_Intracranial_Arteries_Take1.jpg": "فيلم MRA مائل للشرايين الدماغية (لقطة أولى، 12 منظوراً).",
    "MRA_2026-11-05_Film04b_3D_TOF_SagittalOblique_MIP_Intracranial_Arteries_Take2.jpg": "فيلم MRA مائل للشرايين الدماغية (لقطة ثانية، 12 منظوراً).",
    "MRI_2026-11-05_Film05_Axial_DWI_ADC_NoAcuteRestrictedDiffusion.jpg": "فيلم الانتشار الخلوي DWI/ADC (18 مقطعاً): سليم تماماً وينفي وجود أي جلطة دماغية حادة.",
    "MRV_2026-11-05_Film06_3D_PCA_Venography_Attenuated_Lt_Transverse_Sigmoid_IJV.jpg": "فيلم الرنين الوريدي 3D-PCA MRV (40 منظوراً): سيادة وريدية يمينية مع ضيق متجانس بالجيب المستعرض والسيني والوريد الوداجي الشمال بدون أي تجلط.",
    "MRI_2026-11-05_Film07_Axial_eFLAIR_CSVD_LacunarInfarcts_EthmoidSinusitis.jpg": "فيلم eFLAIR المحوري: بؤر تصلب شرايين دقيقة بالمادة البيضاء وجلطات مزمنة قديمة مع التهاب الجيوب الغربالية وزيت السيليكون بالعين الشمال.",
    "MRI_2026-11-05_Film08_Axial_T1W_SE_SulcalProminence_LtEyeSiliconeOil.jpg": "فيلم T1W المحوري: بروز التلافيف وبؤر الجلطات المزمنة مع إشارة ساطعة مميزة لزيت السيليكون بالعين الشمال (2.59 ضعف).",
    "MRI_2026-11-05_Film09_Axial_eT2W_TSE_Lacunes_IAC_EthmoidSinusitis.jpg": "فيلم eT2W المحوري: بؤر لاكونار مزمنة وسلامة تامة لقنوات السمع الداخلية IAC والزاوية المخيخية الجسرية وتأكيد التهاب الجيوب الغربالية.",
    "MRI_2026-11-05_Film10_Coronal_T2W_and_T1W_TSE_BasalGanglia_IAC_EthmoidSinusitis.jpg": "فيلم الكورونال T2W و T1W: يؤكد التهاب الجيوب الغربالية وبؤر العقد القاعدية مع سلامة الحصين والتصالب البصري وأعصاب السمع الثامن.",
    "VASC_2026-11-07_Carotid_Vertebral_ColorDuplex_IMT0.13cm_LtBulbPlaque5mm.jpg": "دوبلكس شرايين الرقبة (د. إبراهيم هاشم 2026-11-07): زيادة سمك جدار الشريان IMT = 0.13 سم وتصلب مزمن مع بلاك بطول 5 مم في عنق السباتي الشمال بدون ضيق حرج، وتدفق فقري سليم.",
    "CARD_2026-11-05_Holter24h_Complete_29Pages_Summary_HRV_and_FullDisclosure.pdf": "ملف الهولتر الكامل (29 صفحة - مستشفى بورسعيد العسكري): متوسط النبض 65، 857 خفقة بطينية مبكرة (311 منها بين 6–9 صباحاً)، 4 نوبات تسارع بطيني قصيرة، 153 خفقة أذينية، وهبوط ST بمقدار -2.1 مم في القناة 3 الساعة 13:11.",
    "CARD_2026-11-05_Holter24h_Page01_Summary_857PVCs_153SVEs_STdepression.jpg": "صفحة 1 من الهولتر (الملخص): متوسط نبض 65 (51–99)، 857 خفقة بطينية، 22 ثنائياً، 4 نوبات V-runs، 153 خفقة أذينية، أقصى هبوط ST -2.1 مم، وتقلب ضربات القلب SDNN 101 مللي ثانية.",
    "CARD_2026-11-05_Holter24h_Page02_24HourTrends_HR51to99_MorningPVCSurge.jpg": "صفحة 2 من الهولتر (الاتجاهات على مدار 24 ساعة): تثبت طفرة الخفقات الصباحية بين 6–9 ص بذروة 164 خفقة في الساعة 08:00 ص.",
    "CARD_2026-11-05_Holter24h_Page03_Absolute_ST_Analysis_Minus2.1mm_CH3.jpg": "صفحة 3 من الهولتر (تحليل مقطع ST): 160 دقيقة إجمالي هبوط بالقناة الثالثة وأقصاه -2.1 مم الساعة 13:11 (بينما ارتفاع 21:27 خطوة تشويش تقنية باللصقة).",
    "CARD_2026-11-05_Holter24h_Page04_QTc_and_QT_3CH_Analysis_NormalQTc.jpg": "صفحة 4 من الهولتر (QTc/QT): فترة QTc طبيعية 100% (<= 450 مللي ثانية، والأقصى 326 مللي ثانية).",
    "CLIN_2026-11-03_Neurosurgery_Rx_DrAymanGalhom_Inderal10_Ciprapra10_CobalF_Brainox.jpg": "روشتة جراحة المخ والأعصاب (أ.د. أيمن جلهوم 2026-11-03، رقم 25334): تشير للرعشة الموضعية وسكر وضغط، وتصرف إندرال 10 مجم مرتين، سيبرابرا 10 مجم نصف قرص، كوبال إف، وبرين أوكس.",
    "CLIN_2026-11-03_Neurosurgery_ClinicalNote_DrAymanGalhom_EssentialTremor_Titubation.jpg": "ملاحظة الفحص السريري اليدوية (أ.د. أيمن جلهوم): توثق الرعشة الأساسية ورعشة الرأس (Essential tremor + Titubation) مع عدم تأثر الأطراف السفلية.",
    "CLIN_2026-11-02_Neurosurgery_Order_DrAymanGalhom_MRI_MRV_MRA_Duplex_Holter.jpg": "طلب الفحوصات الاستكشافية (أ.د. أيمن جلهوم 2026-11-02): طلب رنين المخ والشرايين والأوردة ودوبلكس الرقبة وهولتر 24 ساعة.",
    "CLIN_2021-09-01_MOH_MedicalCouncil_LegalBlindness_LessThan1of60_RetinalOpticAtrophy.jpg": "[تاريخي 2021] قرار المجالس الطبية المتخصصة بوزارة الصحة (2021-09-01 / 2021-09-19، رقم قومي 26304090300188): يوثق حدة إبصار أقل من 1/60 بالعين الأفضل بسبب ضمور الشبكية والعصب البصري.",
    "AUD_2026-11-03_PTA_Masked_AC_BC_FollowUp_RtWRS84_LtWRS76_PureSNHL.jpg": "تخطيط السمع الموثق بقناع عكسي (2026-11-03 مستشفى الحياة بورفؤاد - د. شاهيندا الخميسي): ضعف حسي عصبي نقي (فجوة هوائية عظمية 0 dB)، يمين WRS 84% وعتبة 40 dB، شمال WRS 76% وعتبة 54.2 dB؛ وملاحظة 'Rt As' اليدوية يبطلها سجل GSI-39 المطبوع.",
    "AUD_2026-11-03_Tympanometry_GSI39_FollowUp_Rt1.5cm3_Lt3.2cm3_ExtremeTypeAd.jpg": "سجل قياس ضغط الأذن الحراري GSI-39 (2026-11-03): اليمين 1.5 سم³ عند -5 daPa (طبيعي Type A يدحض ملاحظة Rt As)، والشمال 3.2 سم³ عند 0 daPa يتجاوز حد الرسم البياني 3.0 سم³ (مرونة مفرطة Type Ad)، وردود المنعكس لم تُختبر (NT).",
    "RAD_2024-07-17_CT_Brain_Requisition_IMC_Reason_Tremors.jpg": "[تاريخي 2024] تحويل المجمع الطبي العالمي رقم #474510 (أ.د. أحمد فهمي، 2024-07-17): طلب أشعة مقطعية على المخ بدون صبغة بسبب شكوى الرعشة.",
    "RAD_2024-07-27_CT_Brain_NonContrast_Report_Involutional_LtEyeSiliconeOil.jpg": "[تاريخي 2024] تقرير الأشعة المقطعية بمستشفى الزهور التخصصي (د. عماد محمد، 2024-07-27): تغيرات تراجعية متوافقة مع السن، حفر خلفية سليمة بدون نزيف، وتأكيد وجود كثافة عالية بزجاجي العين الشمال من زيت السيليكون.",
    "NEURO_2024-07-17_Prolonged_2h_EEG_Report_IMC_Normal9HzAlpha.jpg": "[تاريخي 2024] تقرير تخطيط المخ المطول ساعتين (المجمع الطبي العالمي 2024-07-17 - أ.د. أحمد فهمي): خلفية ألفا متماثلة 9 هيرتز بدون أي نشاط صرعي.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch01_11-42-44_AM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 01 (11:42:44 ص): خلفية ألفا 9 هيرتز طبيعية مع ذبذبة عضلية جبهية طفيفة وبدون بؤر صرعية.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch02_11-48-49_AM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 02 (11:48:49 ص): إيقاع مخي طبيعي متماثل مع ذبذبة عضلية متقطعة وبدون تفريغ صرعي.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch03_12-01-10_PM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 03 (12:01:10 م): موجات عضلية سريعة بالمنطقة الجبهية الصدغية مطابقة للرعشة بدون موجات صرعية.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch04_12-10-07_PM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 04 (12:10:07 م): خلفية مخية سليمة مع تداخل رعشة عضلية ثنائية وبدون بطء بؤري.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch05_12-35-17_PM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 05 (12:35:17 م): منتصف التسجيل يؤكد سلامة التخطيط وتماثل الجانبين.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch06_12-55-51_PM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 06 (12:55:51 م): نوبات كثيفة من النشاط العضلي الجسدي للرعشة بدون خمود لاحق أو شوكات صرعية.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch07_13-21-46_PM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 07 (13:21:46 م): تأكيد منشأ الرعشة الجسدي غير الصرعي عبر ذبذبات EMG الجبهية الصدغية.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch08_13-24-34_PM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 08 (13:24:34 م): نشاط مخي مستقر مع تراجع الذبذبة العضلية وغياب أي تطور نوبات صرعية.",
    "NEURO_2024-07-17_EEG_Tracing_Epoch09_13-28-39_PM_AlphaBackground_MuscleArtifact.jpg": "[تاريخي 2024] مقطع تخطيط المخ 09 (13:28:39 م): نهاية فحص الساعتين تؤكد براءة المخ من الصرع.",
    "VASC_2011-06-21_LowerLimb_ColorDoppler_Report_DrHelmyRashad_LtCalfPerforatorIncompetence.png": "[تاريخي 2011] تقرير دوبلكس شرايين وأوردة الطرفين السفليين (د. حلمي رشاد 2011-06-21): الشرايين والأوردة العميقة سليمة تماماً بدون جلطة أو ضيق، مع ارتجاع بثاقب الساق الأيسر ودوالي سطحية.",
    "VASC_2011-06-21_LowerLimb_ColorDoppler_Strip01_AltadamonHospital.png": "[تاريخي 2011] صورة دوبلكس ملون (مستشفى التضامن 2011-06-21): تدفق دموي طبيعي بالأوعية الرئيسية للطرف السفلي بدون انسداد شرياني أو جلطة عميقة.",
    "VASC_2011-06-21_LowerLimb_ColorDoppler_Strip02_NormalArteriesDeepVeins.png": "[تاريخي 2011] صورة دوبلكس ثنائي اللون: سلامة الشريان الظنبوبي الأمامي والخلفي والأوردة المرافقة وانعدام DVT.",
    "VASC_2011-06-21_LowerLimb_ColorDoppler_Strip03_Waveforms_Upright.png": "[تاريخي 2011] طيف الدوبلر النبضي (تم تعديل استقامته 90°): موجات شريانية ثلاثية/ثنائية الطور طبيعية وتدفق وريدي سليم.",
    "VASC_2011-06-21_LowerLimb_ColorDoppler_Strip04_DeepFemoralVenousFlow.png": "[تاريخي 2011] شريحة دوبلكس الوريد الفخذي العميق (مختوم 2011-06-21): مجرى وريدي عميق سالك وطبيعي تماماً.",
    "OPHTH_2014-04-23_Optovue_Macula_OD_CystoidEdema_RPEDisruption_HISTORICAL.png": "[تاريخي 2014] أشعة مقطعية لمركز إبصار العين اليمين (Optovue OCT): تكيسات وارتشاح وتكسر بالطبقة الصبغية RPE قبل عمليات الشبكية لاحقاً.",
    "OPHTH_2014-04-23_Optovue_MaculaMap_EMM5_OU_RPE_Disruption_OD280_OS100_HISTORICAL.png": "[تاريخي 2014] خريطة مقارنة مركز الإبصار للعينين (EMM5): خلل شبكي مزمن مبكر بالعينين (OD 280 / OS 100).",
    "OPHTH_2014-04-23_Optovue_OpticNerve_OS_RNFL78.36um_CDVert0.83_HISTORICAL.png": "[تاريخي 2014] أشعة العصب البصري الشمال (Optovue): سمك الألياف 78.36 ميكرون وتقعر رأسي C/D 0.83 يدل على علة عصب بصري منذ 2014.",
    "OPHTH_2016-02-16_Optovue_OpticNerve_OS_RNFL71.41um_CDVert0.74_HISTORICAL.png": "[تاريخي 2016] متابعة العصب البصري الشمال (د. سناء همام 2016-02-16): انخفاض إضافي بسمك الألياف إلى 71.41 ميكرون (تدهور تدريجي).",
    "OPHTH_2019-08-19_MOH_HealthInsurance_Referral_LtCataract_RD_PriorRtRD_HISTORICAL.png": "[تاريخي 2019] خطاب تحويل التأمين الصحي الرسمي (2019-08-19): يثبت سابقة عملية كتاراكت وانفصال شبكي بالعين اليمين وطلب عملية بالشمال.",
    "OPHTH_2019-08-20_Optopol_OCT_Report_OU_DrOmaimaMohamed_MarkedTSNIT_HISTORICAL.png": "[تاريخي 2019] تقرير OCT الطيفي للعينين #1107 (د. أميمة محمد 2019-08-20): يوثق ترققاً حاداً بعصب العين الشمال (متوسط TSNIT = 64 ميكرون باللون الأحمر).",
    "OPHTH_2019-08-20_Optopol_OCT_Report_OU_DrOmaimaMohamed_AltTake_HISTORICAL.png": "[تاريخي 2019] لقطة بديلة عالية الوضوح لنفس تقرير د. أميمة محمد #1107 بتاريخ 2019-08-20.",
    "OPHTH_2019-08-20_Optopol_OCT_OD_Spectral_MeanTSNIT260_CD0.24_HISTORICAL.png": "[تاريخي 2019] تفصيل Optopol للعين اليمين: متوسط TSNIT 260 (قراءة مبالغ بها بسبب تشويش وتليف واجهة الشبكية والزجاجي).",
    "OPHTH_2019-08-20_Optopol_OCT_OS_SevereRNFLThinning_MeanTSNIT64_HISTORICAL.png": "[تاريخي 2019] تفصيل Optopol للعين الشمال: ترقق شديد ومؤكد بعصب البصر الشمال بمتوسط 64 ميكرون قبل السيليكون.",
    "OPHTH_2019-09-11_BScan_OS_VitreousHemorrhage_NoRetinalDetachment_DrOmaimaSharara_HISTORICAL.png": "[تاريخي 2019] سونار العين الشمال (د. أميمة شرارة 2019-09-11): عتامات ونزيف بالسائل الزجاجي بدون انفصال شبكي كلي وقتها.",
    "OPHTH_2019-12-15_BScan_OS_SiliconizedEye_RetinaInPlace_DrHaithamKhalil_HISTORICAL.png": "[تاريخي 2019] سونار العين الشمال بعد الجراحة (د. هيثم خليل 2019-12-15): عين محقونة بزيت السيليكون والشبكية ثابتة في مكانها.",
    "OPHTH_2021-04-07_BScan_OD_EmulsifiedSiliconeOil_RetinaInPlace_DrYasminSinan_HISTORICAL.png": "[تاريخي 2021] سونار العين اليمين (د. ياسمين سنان 2021-04-07): يثبت حقن زيت السيليكون وتكسره (Emulsified) بالعين اليمين أيضاً مع بقاء الشبكية في مكانها.",
    "OPHTH_2021-04-07_BScan_OS_SiliconFilledEye_RetinaInPlace_DrYasminSinan_HISTORICAL.png": "[تاريخي 2021] سونار العين الشمال (د. ياسمين سنان 2021-04-07): عين مملوءة بزيت السيليكون والشبكية في مكانها.",
    "OPHTH_2021-05-29_BScan_OS_SiliconFilledEye_RetinaInPlace_DrManarElGayar_HISTORICAL.png": "[تاريخي 2021] أحدث سونار قديم متاح للعين الشمال (د. منار الجيار 2021-05-29): آخر توثيق قبل 5 سنوات لوجود الشبكية في مكانها خلف زيت السيليكون.",
    "OPHTH_2026-06-01_Topcon_OCT_Macula_Radial_OD_Avg221.9um_Qual10.png": "[حالي 2026] مقطعية مركز الإبصار للعين اليمين (توبكون Maestro2): جودة عالية 10/10 (الوسط رايق بلا سيليكون!)، الحفرة متصلة بسمك 259 ميكرون والمتوسط 221.9 ميكرون.",
    "OPHTH_2026-06-01_Topcon_OCT_Macula_Radial_OS_LowQual2_SiliconeMediaOpacity.png": "[حالي 2026] مقطعية مركز الإبصار للعين الشمال: جودة ضعيفة 2/10 بسبب زيت السيليكون والعتمة مما يجعل قياس 400.3 ميكرون مجرد تشويش جهاز.",
    "OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OD_RNFL40um_SevereAtrophy_Qual9.png": "[حالي 2026] فحص العصب البصري للعين اليمين (جودة عالية 9/10): يثبت ضموراً حقيقياً شديداً في ألياف العصب (إجمالي RNFL = 40 ميكرون، السفلي 34 ميكرون، تقعر 0.70).",
    "OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OS_Qual2_MediaArtifact.png": "[حالي 2026] فحص العصب البصري للعين الشمال: جودة متدنية 2/10 بسبب عتامة السيليكون؛ رقم الـ 90 ميكرون غير موثوق إطلاقاً ولا يمثل أليافاً حقيقية.",
    "OPHTH_2026-06-27_Zeiss_VisualField_HFA_VFI18pct_UnreliableTest_DrAlaaKhaled.png": "[حالي 2026] مجال الإبصار (زايس 24-2 - د. آلاء خالد 2026-06-27): مؤشر VFI = 18% مع فقدان واسع وبقعة سفلية أنفية، ومختوم رسمياً 'فحص غير موثوق' لفقد التثبيت (< 1/60).",
    "OPHTH_2026-07-08_CSO_Flash_VEP_OD_Present_N1P1_10.29uV_P2_125.39ms.png": "[حالي 2026] رسم العصب البصري بالوميض للعين اليمين (CSO 2026-07-08): إشارة كهربائية واضحة ومقاسة إلى قشرة المخ (N1-P1 = 10.29 µV، ذروة P2 عند 125.39 ms).",
    "OPHTH_2026-07-08_CSO_Flash_VEP_OS_Present_N1P1_7.99uV_P2_132.42ms.png": "[حالي 2026] رسم العصب البصري بالوميض للعين الشمال (CSO 2026-07-08): مفاجأة إكلينيكية — المسار البصري ليس ميتاً تماماً ولا تزال الإشارة تقاس (N1-P1 = 7.99 µV، P2 132.42 ms) رغم السيليكون.",
    "OPHTH_2026-07-08_CSO_ERG_DarkAdapted_0.01_OU_OD39.1uV_OS35.8uV.png": "[حالي 2026] رسم الشبكية بالظلام (النبضات العصوية 0.01): ضعف عام بالجهتين (يمين 39.1 µV وشمال 35.8 µV).",
    "OPHTH_2026-07-08_CSO_ERG_DarkAdapted_3.0_OU_OD94.4uV_OS76.8uV.png": "[حالي 2026] رسم الشبكية المختلط بالظلام (3.0): انخفاض استجابة المستقبلات الضوئية خصوصاً بالشمال (76.8 µV مقابل 94.4 µV يمين).",
    "OPHTH_2026-07-08_CSO_ERG_DarkAdapted_3.0_OU_AltTake.png": "[حالي 2026] لقطة ثانية موثقة من سجل ERG المختلط بالظلام 3.0 بتاريخ 2026-07-08.",
    "OPHTH_2026-07-08_CSO_ERG_OscillatoryPotentials_OD10.0uV_OS0.74uV_SevereOSIschemia.png": "[حالي 2026] الجهود التذبذبية للشبكية (OPs - الأهم تشخيصياً): انهيار شبه كامل لتروية الطبقات الداخلية للشبكية الشمال (0.74 µV فقط مقابل 10.00 µV باليمين!).",
    "OPHTH_2026-07-08_CSO_ERG_LightAdapted_3.0_OU_OD37.1uV_OS30.4uV.png": "[حالي 2026] رسم الشبكية بالإضاءة (المخاريط المفردة 3.0): تراجع استجابة المخاريط بالجهتين (يمين 37.1 µV وشمال 30.4 µV).",
    "OPHTH_2026-07-08_CSO_ERG_LightAdapted_3.0_Flicker_OD21.2uV_OS13.0uV.png": "[حالي 2026] رسم الشبكية بالوميض السريع (30/48 Hz Flicker): ضعف ملموس باستجابة مخاريط البقعة الصفراء بالشمال (13.0 µV) مقارنة باليمين (21.2 µV).",
}


def build_js_data_payload(manifest: dict, quant: dict, file_count: int) -> str:
    """Assemble strict bilingual (EN / Egyptian Arabic) data payload for the 6-tab dashboard."""
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

    files_index = []
    for f in manifest.get("files", []):
        cat = f["category_code"]
        date = f["date_of_test"]
        tstat = f.get("temporal_status", "CURRENT_2026")
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
        elif cat == "OPHTH":
            grp = "OPHTH_HIST" if tstat == "HISTORICAL_OLD" else "OPHTH_2026"
        elif cat == "VASC":
            grp = "VASC_LL" if date.startswith("2011") else "VASC"
        else:
            grp = "CLIN"
        sz_kb = f["size_bytes"] / 1024.0
        sz_str_en = f"{sz_kb/1024:.2f} MB" if sz_kb >= 1024 else f"{sz_kb:.0f} KB"
        sz_str_ar = f"{sz_kb/1024:.2f} م.ب" if sz_kb >= 1024 else f"{sz_kb:.0f} ك.ب"
        raw_name = f["original_raw_filename"]
        raw_en = "Amal_Mokhtar_Holter29p_Raw.pdf (Original Arabic filename on disk)" if raw_name == "أمال مختار.pdf" else raw_name
        files_index.append(
            {
                "cat": cat,
                "grp": grp,
                "date": date,
                "temporal_status": tstat,
                "is_historical": (tstat == "HISTORICAL_OLD"),
                "raw_en": raw_en,
                "raw_ar": raw_name,
                "newName": f["new_semantic_filename"],
                "href": "../" + f["folder_path"].replace("\\", "/"),
                "size_en": sz_str_en,
                "size_ar": sz_str_ar,
                "summary_en": f["one_line_clinical_summary"],
                "summary_ar": FILE_SUMMARIES_AR.get(f["new_semantic_filename"], f["one_line_clinical_summary"]),
            }
        )

    i18n = {
        "en": {
            "pageTitle": "MDT Clinical Case Study — Amal Mokhtar Mohamed Ibrahim (63.5y F)",
            "brandTitle": "7-Consultant MDT Case Study | Neuro-Vascular, Otological & Vitreoretinal Evaluation",
            "brandSub": f"Patient: Amal Mokhtar Mohamed Ibrahim • 63.5y Female • National ID: 26304090300188 • {file_count} Verified Records (2011–2026)",
            "btnThemeDark": "Dark Mode",
            "btnThemeLight": "Light Mode",
            "btnPrintTab": "Print Tab",
            "btnPrintAll": "Full Dossier (PDF)",
            "navLabel": "Case-Study Journey",
            "pdfLabel": "Shareable Reports & Index",
            "pdfLinks": [
                {"href": "../05_Shareable_PDF_Reports/01_MDT_Executive_Summary_and_Action_Plan.pdf", "txt": "01. 2-Page Executive Summary (PDF)"},
                {"href": "../05_Shareable_PDF_Reports/02_Consultant_7_Board_Audit_Reports.pdf", "txt": "02. 7-Consultant Board Audits (PDF)"},
                {"href": "../05_Shareable_PDF_Reports/03_Patient_Case_Study_Full_Report.pdf", "txt": "03. Complete Case Monograph (PDF)"},
                {"href": "../04_Final_MDT_Deliverables/Master_MDT_Case_Study_Report.md", "txt": "Master Markdown Dossier"},
                {"href": "../01_Structured_and_Renamed_Inputs/FILE_RENAMING_INDEX.md", "txt": f"Complete {file_count}-File Index"},
            ],
            "tabs": [
                {"id": "tab-profile", "num": "01", "label": "01. Patient Profile & Clinical History", "count": "Profile"},
                {"id": "tab-videos", "num": "02", "label": "02. Neuroimaging Video Suite (MRI / MRA / MRV)", "count": "3 Vid"},
                {"id": "tab-diagnostics", "num": "03", "label": "03. Diagnostics, Labs & Imaging Gallery", "count": str(file_count)},
                {"id": "tab-prior", "num": "04", "label": "04. Prior Consultants & Clinical Conflict", "count": "Conflict"},
                {"id": "tab-board", "num": "05", "label": "05. 7-Consultant MDT Board (Conclusions & Action Plans)", "count": "7 Board"},
                {"id": "tab-plan", "num": "06", "label": "06. Master Treatment & Medication Plan", "count": "Rx"},
            ],
            "histBadge": "HISTORICAL / OLD EXAM — DO NOT CONFUSE WITH CURRENT 2026 STATUS",
            "histShortBadge": "HISTORICAL EXAM",
            "currBadge": "CURRENT 2026 RECORD",
            # Tab 1: Patient Profile & Clinical History
            "t1": {
                "jumpTitle": "Interactive MRI / MRA / MRV Video Suite Ready in Module 02",
                "jumpSub": "Direct inline streaming players for 3D-MRV Venography (0.48 Lt attenuation), 513-slice 3D-TOF MRA, and Axial T2/FLAIR CSVD are mounted immediately in Module 02.",
                "jumpBtn": "Open Neuroimaging Video Suite",
                "secHead": "Patient Demographics, Risk Substrates & Longitudinal Clinical History (2011 – Nov 2026)",
                "kpis": [
                    {"cls": "accent", "lbl": "Identity & Verified Age", "val": "63.5y Female", "sub": f"DOB 1963-04-09 (ID ...00188); clerical 1963-03-27 on MRI film; {file_count} audited primary files"},
                    {"cls": "danger", "lbl": "Vision & 12-Yr Eye Axis", "val": "< 1/60 | OD 40um", "sub": "OD clear Qual 10 (atrophy RNFL 40um); OS silicone Qual 2 + OPs 0.74uV; Flash VEP measurable OU!"},
                    {"cls": "warn", "lbl": "Cardio-Metabolic Load", "val": "80 IU/d | 0.13 cm", "sub": "T2DM + HTN + Dyslipidemia; Carotid IMT +44.4% & 5 mm Lt bulb plaque (2011 leg Doppler PAD negative)"},
                    {"cls": "danger", "lbl": "Core Telemetry & Tymp", "val": "857 PVCs | 3.2 cm3", "sub": "36.29% AM surge, -2.1 mm ST (13:11); MRV L/R 0.48; 0 dB ABG & Rt 1.5 cm3 Type A"},
                ],
                "safetyTitle": "NON-NEGOTIABLE PATIENT SAFETY ALERTS BEFORE ANY DIAGNOSTIC ORDER OR PRESCRIPTION",
                "safetyBullets": [
                    "<strong>Light-Tracked VNG Is Technically Invalid:</strong> Bilateral legal blindness (&lt; 1/60 better eye, MOH 2021) + OD profound atrophy (<code>40 um</code>) + persistent OS silicone oil strips ~95% visual vestibular reserve and invalidates optical fixation, saccades, and OPK tracking.",
                    "<strong>Defer Catheter DSA Across Left Carotid Bulb:</strong> Unfiltered DSA traverses a documented <code>5 mm</code> atheromatous plaque at the Left Carotid Bulb while serum Cr/eGFR (on Metformin) and CH3 <code>-2.1 mm</code> silent myocardial ischemia remain unaddressed.",
                    "<strong>Do NOT Schedule OS Oil Removal on False Topcon Noise:</strong> OS Topcon <code>Qual 2</code> prints artifactual <code>90 um</code> RNFL noise (true RNFL was already <code>64 um</code> in 2019), while ~7-year silicone removal risks hypotony/re-detachment; get a <strong>fresh 2026 Ocular B-Scan + IOP</strong> first.",
                    "<strong>Morning Insulin + Electrolyte Arrhythmogenic Loop:</strong> Stop daily <code>Meclizine</code> &amp; <code>Ginkgo</code>, avoid systemic <code>Cidamex + HCTZ</code> potassium wasting, audit morning <code>80 IU/d</code> insulin hypoglycemia (driving <code>36.29%</code> of 857 PVCs), and protect optic disc BP (&gt;= 115/70 mmHg).",
                ],
                "compHead": "1. Five Integrated Clinical & Ophthalmological Presentation Domains",
                "complaints": [
                    {"title": "Left-Sided Pulsatile Tinnitus", "badge": "b-amber", "tag": "Primary Vascular/Acoustic", "desc": "Persistent rhythmic whooshing localized to the Left ear. Mechanical acoustic modeling demonstrates blood turbulence traversing a hypoplastic Left Transverse-Sigmoid-IJV conduit (<code>L/R 0.48</code>) amplified by an ultra-flaccid <code>3.2 cm3 Type Ad</code> monomeric tympanic membrane and post-PVC stroke-volume surges."},
                    {"title": "Disabling Balance Instability ('4-Hit' Vertigo)", "badge": "b-red", "tag": "Multi-Sensory Crisis", "desc": "Severe unsteadiness driven concurrently by (1) Left cochleovestibular microvascular insult, (2) bilateral legal blindness (&lt;1/60, OD RNFL 40um / OS OPs 0.74uV), (3) CSVD Fazekas 1-2 + daily <code>Meclizine</code> CNS suppression, and (4) cardiac/glycemic hemodynamic dips."},
                    {"title": "Progressive Asymmetric Sensorineural Hearing Loss", "badge": "b-blue", "tag": "Cochlear Localization", "desc": "Definitive Masked PTA (<code>2026-11-03</code>) documents <strong>0 dB Air-Bone Gap</strong> bilaterally (pure cochlear SNHL: Right 3f PTA <code>40.0 dB</code>, WRS <code>84%</code>; Left <code>54.2 dB</code>, WRS <code>76%</code>) with historical positive <strong>Metz recruitment (+45 dB SL)</strong> -> fit Bilateral Open-Fit WDRC RIC (<code>OSPL90 &lt;= 95-100 dB SPL</code>)."},
                    {"title": "Upper-Limb & Head Tremors (Titubation)", "badge": "b-purple", "tag": "Movement vs Metabolic", "desc": "Historical Parkinsonism label treated with <code>Sinemet</code> vs Prof. Galhom's <code>2026-11-03</code> exam logging <strong>Essential Tremor + Titubation</strong> (negative lower-limb involvement). Prolonged <code>2024 2h EEG (9 epochs)</code> proves non-epileptic EMG tremor bursts, worsened by morning insulin dips."},
                    {"title": "Bilateral Severe Vision Loss & Left Eye Hope Question", "badge": "b-red", "tag": "Neuro-Ophthalmology", "desc": "Pre-existing diabetic/glaucomatous retinopathy (<code>2014–2019</code>) progressed to bilateral RD vitrectomy + silicone oil (<code>2019–2021</code>). Current <code>2026</code> tests show OD clear media / <code>40 um</code> RNFL, OS siliconized / <code>OPs 0.74 uV</code> ischemic collapse, yet <strong>preserved Flash VEP bilaterally (OD 10.29 uV / OS 7.99 uV)</strong>."},
                ],
                "timelineHead": "2. Chronological Clinical, Ophthalmic & Vascular Timeline (2011 – Nov 2026)",
                "timeline": [
                    {"hist": True, "date": "2011-06-21", "src": "Altadamon Hospital (Dr. Helmy Rashad)", "detail": "HISTORICAL: Bilateral Lower Limb Arterial/Venous Color Doppler confirming patent triphasic/biphasic lower-limb arteries (NO PAD) and patent deep femoral/popliteal/tibial veins (NO DVT), with left calf perforator incompetence &amp; superficial varicosities."},
                    {"hist": True, "date": "2014-04-23 -> 2016-02-16", "src": "Historical Optovue OCT Baseline (Dr. Sanaa Hamam)", "detail": "HISTORICAL: Already showed early chronic glaucomatous/ischemic optic nerve cupping and thinning in OS (<code>RNFL 78.36 um, Vert C/D 0.83</code> in 2014 -> <code>71.41 um</code> in 2016) alongside OD cystoid macular edema &amp; RPE disruption (<code>OD 280 / OS 100</code>) BEFORE 2019 RD surgeries."},
                    {"hist": True, "date": "2019-08-19 -> 2019-12-15", "src": "MOH Referral, Optopol OCT & Early B-Scans (Drs. Omaima / Haitham)", "detail": "HISTORICAL: Health Insurance referral documented PRIOR Rt cataract+RD surgery and requested Lt surgery. 2019-08-20 Optopol OCT proved severe OS thinning (<code>Mean TSNIT 64 um</code>, red). B-Scans tracked OS vitreous hemorrhage (Sept 2019) -> siliconized OS with retina in place (Dec 2019)."},
                    {"hist": True, "date": "2021-04-07 -> 2021-09-19", "src": "Ramad Port Said B-Scans & MOH Disability Decree", "detail": "HISTORICAL: Apr 2021 B-Scans (Dr. Yasmin Sinan) proved BOTH OD (<code>emulsified silicon</code>) and OS (<code>silicon filled eye</code>) underwent silicone oil surgery; last OS B-Scan in May 2021 (Dr. Manar El-Gayar). Sept 2021 MOH Council certified <code>&lt; 1/60</code> legal blindness (ID 26304090300188)."},
                    {"hist": True, "date": "2024-07-17 -> 2024-11-02", "src": "IMC 2h EEG, El-Zohor Brain CT & Al-Hayah Unmasked Audio", "detail": "HISTORICAL: 2024-07-17 2h EEG (9 epochs) showed 9 Hz alpha + EMG tremor bursts, ZERO spikes. 2024-07-27 CT showed normal fossa &amp; left vitreous silicone oil. 2024-11-02 unmasked audiology (Rt As 0.3 / Lt Ad 1.7 / Metz +45 dB SL) superseded as primary baseline on 2026-11-03."},
                    {"hist": False, "date": "2026-06-01 -> 2026-07-08", "src": "Current Topcon OCT, Zeiss HFA 24-2 & CSO VEP/ERG", "detail": "CURRENT 2026: Topcon OD <code>Qual 10</code> clear media, attached fovea (259 um), <strong>profound Optic Atrophy (RNFL 40 um)</strong>; OS <code>Qual 2</code> scatter noise (90 um artifact!). Zeiss HFA <code>VFI 18% ('Not reliable Test', Dr. Alaa Khaled)</code>. CSO ERG proves OS inner-retinal ischemia (<code>OPs 0.74 vs 10.0 uV OD</code>), <strong>yet Flash VEP is measurable bilaterally (OD 10.29 uV / OS 7.99 uV)</strong>!"},
                    {"hist": False, "date": "2026-11-02 -> 2026-11-03", "src": "Definitive Follow-Up Audiology + Neurosurgery & ENT Consults", "detail": "CURRENT 2026: Masked PTA + GSI-39 (Dr. Shahenda): <code>0 dB ABG</code>, Rt WRS <code>84%</code> / Lt WRS <code>76%</code>, Rt Peak <code>1.5 cm3</code> Normal Type A (refutes handwritten 'Rt As'), Lt Peak <code>3.2 cm3</code> Extreme Type Ad. Prof. Galhom logged <code>Essential tremor + Titubation</code>; Dr. Salem ordered VNG + Meclizine/Ginkgo."},
                    {"hist": False, "date": "2026-11-05 -> 2026-11-07", "src": "1.5T MRI/MRA/MRV Brain, 24h Holter 29p & Carotid Duplex", "detail": "CURRENT 2026: CSVD Fazekas 1-2 + chronic Rt BG lacune + intact MRA + normal optic chiasm + attenuated Lt TS/Sigmoid/IJV (<code>L/R 0.48</code>) + <code>2.59x</code> Lt silicone oil. Holter: <code>857 PVCs</code> (311 between 06–09 AM), 4 V-runs, <code>153 SVEs</code>, CH3 <code>-2.1 mm ST</code> @ 13:11. Duplex: <code>IMT 0.13 cm</code> &amp; <strong>5 mm Lt Carotid Bulb plaque</strong>."},
                ],
                "medsHead": "3. Complete Current & Recently Prescribed Medications Inventory",
                "medCols": ["Medication & Formulation", "Dose / Timing", "Source / Prescriber", "Clinical Role & MDT Status Flag", "Flag"],
                "currentMeds": [
                    {"drug": "Toujeo SoloStar (Insulin Glargine U300)", "dose": "30 IU SC qHS", "src": "EHA Chronic DM", "role": "Basal glycemic control (part of 80 IU/d total insulin requiring CGM hypoglycemia audit).", "tag": "CAUTION", "badge": "b-amber"},
                    {"drug": "Apidra SoloStar (Insulin Glulisine)", "dose": "50 IU/day divided SC pre-meals", "src": "EHA Chronic DM", "role": "Heavy prandial bolus load (62.5% of daily insulin); prime suspect for morning adrenergic dips/PVCs.", "tag": "CAUTION", "badge": "b-amber"},
                    {"drug": "Gliptus Plus (Vildagliptin 50 / Metformin 1000 mg)", "dose": "1 tab PO BID", "src": "EHA Chronic DM", "role": "Contraindicated if eGFR &lt; 30; MUST hold 48h before/after any contrast CTA/DSA study.", "tag": "CAUTION", "badge": "b-amber"},
                    {"drug": "Angiopress Comp (Captopril 50 / HCTZ 25 mg)", "dose": "1 tab PO daily", "src": "EHA Chronic HTN", "role": "Short 2h captopril half-life fails nocturnal BP control while HCTZ drives K+/Mg2+ wasting &amp; PVCs.", "tag": "SWITCH", "badge": "b-red"},
                    {"drug": "Cidamex (Acetazolamide 250 mg — Systemic Glaucoma Alert)", "dose": "AVOID Systemic Use", "src": "MDT Safety Guardrail", "role": "DO NOT use alongside HCTZ: additive kaliuresis + metabolic acidosis worsens 857 PVCs/V-runs (use topical drops if IOP raised).", "tag": "AVOID", "badge": "b-red"},
                    {"drug": "Zatorva 20 mg (Atorvastatin)", "dose": "20 mg PO nocte", "src": "EHA Dyslipidemia", "role": "Under-dosed for 5 mm Lt bulb plaque, 0.13 cm IMT, CSVD &amp; -2.1 mm CH3 ST depression (upgrade to 40-80 mg).", "tag": "UPGRADE", "badge": "b-blue"},
                    {"drug": "Aggrex 75 mg (Aspirin)", "dose": "75 mg PO daily", "src": "EHA Vascular Prev", "role": "Core secondary antiplatelet; keep monotherapy (strip unnecessary Ginkgo co-bleeding risk).", "tag": "CONTINUE", "badge": "b-green"},
                    {"drug": "Sinemet (Carbidopa/Levodopa)", "dose": "Per chronic schedule", "src": "Prior Neurology", "role": "Keep strictly until formal off/on MDS-UPDRS exam confirms whether true levodopa response exists.", "tag": "VERIFY", "badge": "b-amber"},
                    {"drug": "Inderal 10 mg (Propranolol)", "dose": "10 mg PO BID", "src": "Neurosurgery (2026-11-03)", "role": "Non-selective beta-blocker masks hypoglycemia tachycardia/tremor on 80 IU/d insulin; ask Cardiology to switch to Bisoprolol.", "tag": "SWITCH", "badge": "b-amber"},
                    {"drug": "Ciprapra 10 mg (Escitalopram)", "dose": "1/2 tab (5 mg) PO post-dinner", "src": "Neurosurgery (2026-11-03)", "role": "Adds SSRI platelet 5-HT inhibition atop Aspirin; verify depressive/anxiety indication.", "tag": "REVIEW", "badge": "b-amber"},
                    {"drug": "Cobal F (Vitamin B12 / Methylcobalamin)", "dose": "1 tab PO qAM", "src": "Neurosurgery (2026-11-03)", "role": "Safe neuropathic support (Metformin depletion), though baseline serum B12 should be drawn.", "tag": "CONTINUE", "badge": "b-green"},
                    {"drug": "Brain-Ox 30 mg (Vincamine)", "dose": "1 cap PO post-lunch", "src": "Neurosurgery (2026-11-03)", "role": "Unproven cerebral vasodilator risking orthostatic/optic-disc hypoperfusion &amp; coronary steal in -2.1 mm ST ischemia.", "tag": "STOP", "badge": "b-red"},
                    {"drug": "Meclizine 25 mg", "dose": "1 tab qPM x 1 month", "src": "ENT (2026-11-03)", "role": "Daily antihistamine vestibulosuppression blocks central brain compensation &amp; worsens falls/Parkinsonism.", "tag": "STOP", "badge": "b-red"},
                    {"drug": "Cetrovit Ginkgo (Ginkgo Biloba)", "dose": "1 tab qAM x 2 months", "src": "ENT (2026-11-03)", "role": "No proven SNHL cure; creates dangerous triple antiplatelet bleed hazard alongside Aggrex + Ciprapra.", "tag": "STOP", "badge": "b-red"},
                ],
            },
            # Tab 2: Neuroimaging Video Suite (MRI / MRA / MRV)
            "t2": {
                "videoSuiteTitle": "Neuroimaging Video Suite (Interactive DICOM / MRI / MRV Viewer)",
                "videoBadges": ["1.5T MRI / MRA / MRV (2026-11-05)", "Relative Web Streaming Ready"],
                "jumpDiagTitle": "Continue to Complete Diagnostics, 69-File Manifest & 41-Plate Imaging Gallery",
                "jumpDiagSub": "Includes Quantitative Holter/Audiometry plots, 2026 & Historical (2011–2024) Ophthalmic/Vascular cropped images, and the Pending Workup Orders box.",
                "jumpDiagBtn": "Open Module 03: Diagnostics & Gallery",
                "videos": [
                    {
                        "title": "Card 1: 3D-MRV Venography Video",
                        "badge": "Derived Cine",
                        "badgeCls": "b-amber",
                        "src": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4",
                        "dlText": "Download / Open Video",
                        "secText": "Open Tab",
                        "secHref": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4",
                        "secDownload": False,
                        "desc": "Highlighting the <strong>attenuated Left Transverse, Sigmoid Sinus &amp; IJV</strong> (<code>L/R 0.48</code> calibre ratio vs dominant Right) across 18 Dark &amp; Inverted 3D-PCA MIP projections with <strong>zero dural thrombosis</strong> (compiled from primary <code>Film06</code>).",
                    },
                    {
                        "title": "Card 2: 3D-TOF MRA Video",
                        "badge": "Primary 513-Slice Stack",
                        "badgeCls": "b-green",
                        "src": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil_WebH264.mp4",
                        "dlText": "Download / Open Video",
                        "secText": "Raw .MOV",
                        "secHref": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil.mov",
                        "secDownload": True,
                        "desc": "Showing <strong>unremarkable intracranial arteries</strong> &amp; patent Circle of Willis across 1.0-mm Sc 11 axial source slices (skull base to vertex), zero high-flow dAVF, and bright <strong>2.59x Lt intraocular silicone oil</strong>.",
                    },
                    {
                        "title": "Card 3: Axial T2/FLAIR MRI Video",
                        "badge": "Derived Cine",
                        "badgeCls": "b-amber",
                        "src": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4",
                        "dlText": "Download / Open Video",
                        "secText": "Open Tab",
                        "secHref": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4",
                        "secDownload": False,
                        "desc": "Displaying <strong>Basal Ganglia lacunar infarcts</strong> (right lentiform) and <strong>CSVD small vessel disease (Fazekas 1–2)</strong> alongside ethmoiditis, intact IACs, and normal optic chiasm across 20 synchronized slices (<code>Film09 + Film07</code>).",
                    },
                ],
                "seqHead": "Quantitative Sequence Synthesis Across All 10 MRI Films & 513 MRA Source Slices",
                "seqCards": [
                    {"title": "Venous MRV Geometry (Film06, PA R-on-Right Verified)", "badge": "b-amber", "tag": "L/R 0.48 Ratio", "desc": "Right-dominant transverse/sigmoid sinus &amp; IJV drainage. Left transverse-sigmoid-IJV shows uniform congenital/chronic attenuation (0.48 ratio) with ZERO filling defect/thrombosis and NO giant sigmoid sinus diverticulum."},
                    {"title": "Arterial TOF-MRA & Skull Base (Video01 + Films 02–04b)", "badge": "b-green", "tag": "No High-Flow dAVF", "desc": "513 axial 1.0-mm slices confirm normal carotid siphons, Circle of Willis, vertebrobasilar junction, aerated left mastoid cells, and ZERO dilated external carotid feeder (recognizing 1.5T TOF threshold cannot exclude a tiny slow Cognard Type I twig)."},
                    {"title": "Parenchyma, Orbits & Chiasm (Films 05, 07–10)", "badge": "b-blue", "tag": "CSVD + 2.59x Oil", "desc": "Negative DWI/ADC (zero acute stroke); CSVD Fazekas 1–2 + chronic right lentiform lacune (ZERO MS Dawson fingers); normal IACs/CPAs &amp; intact optic chiasm; bilateral ethmoiditis; prominent <strong>Left Intraocular Silicone Oil (2.59x signal)</strong> omitted on formal PDF."},
                ],
            },
            # Tab 3: Diagnostics, Labs & Imaging Gallery
            "t3": {
                "pendingHead": "ORDERED / PENDING DIAGNOSTIC WORKUP (DO NOT CONFUSE WITH COMPLETED TESTS ABOVE)",
                "pendingItems": [
                    {"title": "Modified Dark-Only VNG (Lab Safety Mandate)", "txt": "Written warning required: patient is legally blind (&lt; 1/60). CANCEL light-target saccade/pursuit/OPK calibration; perform <strong>Dark Positional + Bithermal Calorics ONLY</strong>."},
                    {"title": "Fresh 2026 Bilateral Ocular B-Scan + IOP + Gonioscopy", "txt": "Last B-Scan in file is May 2021! Order urgent 2026 B-Scan (OU) + Applanation IOP to confirm OS retina remains attached behind Qual 2 silicone opacity &amp; rule out emulsified oil glaucoma."},
                    {"title": "Selective Bedside Neck Compression & Auscultation", "txt": "Quiet mastoid auscultation comparing <strong>gentle anterior Lt IJV compression</strong> vs <strong>pinpoint Lt occipital artery pulse pressure</strong> before considering any 4D-MRA / CTA / DSA."},
                    {"title": "STAT Electrolytes & Renal Clearance Panel", "txt": "<code>K+, Na+, Mg2+, Ca2+, Creatinine/eGFR, Urine ACR</code> immediately (mandatory prior to stopping HCTZ/blocking Cidamex or any contrast imaging atop Metformin)."},
                    {"title": "14-Day CGM (03:00–09:00 AM) + Metabolic Panel", "txt": "Audit 80 IU/d insulin for occult morning hypoglycemia driving 36.29% AM PVCs + tremor, alongside <code>HbA1c, Fasting Lipids (LDL), TSH, CBC, Vit B12, ALT/AST</code>."},
                    {"title": "Cardiology Ischemia & Neurology UPDRS Challenge", "txt": "<strong>TTE + Pharmacological Stress Echo/SPECT</strong> (NO treadmill due to &lt;1/60 blindness!) for -2.1 mm CH3 ST depression + supervised <strong>off/on Sinemet MDS-UPDRS test</strong>."},
                ],
                "diagHead": "1. Complete Diagnostic Investigations Matrix (2011 Historical – Nov 2026 Current)",
                "investigations": [
                    {"hist": False, "title": "1.5T MRI / MRA / MRV Brain (2026-11-05)", "badge": "b-blue", "status": "CURRENT • Film01–10", "metrics": ["CSVD Fazekas 1–2 + chronic Rt BG, Lt frontal &amp; Rt parietal lacunes; ZERO acute DWI stroke; MS excluded.", "MRA: Normal Circle of Willis &amp; vertebrobasilar flow; intact IACs/CPAs &amp; normal optic chiasm (Film10).", "MRV: Attenuated Lt TS/Sigmoid/IJV (L/R 0.48), zero clot. Bilateral ethmoiditis + omitted 2.59x Lt Silicone Oil."]},
                    {"hist": False, "title": "Definitive Masked PTA & GSI-39 (2026-11-03)", "badge": "b-red", "status": "CURRENT • Audio Baseline", "metrics": ["Masked AC/BC matches cleanly (0 dB ABG) — Rt 3f PTA 40.0 dB (WRS 84% @ 65 dB); Lt 54.2 dB (WRS 76% @ 90 dB).", "GSI-39 Thermal Printout: Rt Peak 1.5 cm3 @ -5 daPa (Normal Type A — refutes handwritten 'Rt As').", "Lt Peak 3.2 cm3 @ 0 daPa (&gt;3.0 cm3 plot ceiling, Extreme Type Ad). Historical Metz +45 dB SL confirms cochlear focus."]},
                    {"hist": False, "title": "Current Ophthalmology: Topcon OCT, Zeiss HFA & CSO VEP/ERG (Jun–Jul 2026)", "badge": "b-red", "status": "CURRENT • 2026 Eye", "metrics": ["Topcon (06-01): OD Qual 10 clear media, fovea 259 um attached, <strong>severe Optic Atrophy RNFL 40 um</strong> (Inf 34 um); OS Qual 2 scatter (90 um printout is artifact!).", "Zeiss HFA 24-2 (06-27): VFI 18%, signed <strong>'Not reliable Test'</strong> by Dr. Alaa Khaled (&lt;1/60 fixation loss).", "CSO (07-08): <strong>ERG OPs collapsed OS 0.74 uV vs OD 10.00 uV</strong> &amp; Flicker OS 12.97 vs OD 21.19 uV, <strong>YET Flash VEP present OU (OD 10.29 uV / OS 7.99 uV)</strong>!"]},
                    {"hist": False, "title": "24-Hour Holter ECG — 29 Pages (2026-11-05)", "badge": "b-red", "status": "CURRENT • 95,454 Beats", "metrics": ["Avg HR 65 bpm (Min 51 @ 02:06, Max 99 @ 12:32); SDNN 101 ms; 100% normal QTc (max 326 ms).", "857 PVCs (0.898%), 22 V-pairs, 4 V-runs (up to 104 bpm @ 08:00 AM), 153 SVEs.", "36.29% AM cluster (311 PVCs 06–09 AM) + 160 min CH3 ST depression (max -2.1 mm @ 13:11)."]},
                    {"hist": False, "title": "Carotid & Vertebral Color Duplex (2026-11-07)", "badge": "b-amber", "status": "CURRENT • Bulb Plaque", "metrics": ["Carotid Intima-Media Thickness (IMT) 0.13 cm (+44.4% above 0.9 mm upper normal limit).", "Bilateral non-stenotic atheromatous plaques; largest <strong>5 mm plaque in Left Carotid Bulb</strong>.", "Direct embolic DSA catheter hazard along left carotid navigation route; normal vertebral velocities."]},
                    {"hist": True, "title": "Prolonged 2h EEG (9 Epochs) & Non-Contrast Brain CT (Jul 2024)", "badge": "b-amber", "status": "HISTORICAL • Jul 2024", "metrics": ["IMC 2h EEG (2024-07-17): Symmetric 9 Hz alpha rhythm, normal HV/photic, ZERO epileptiform discharges.", "Dense fronto-temporal EMG tremor artifacts prove somatic/subcortical tremor (NOT cortical seizures).", "El-Zohor CT (2024-07-27): Age-related involutional changes; proves Lt silicone oil existed 27 mos before 2026 MRI."]},
                    {"hist": True, "title": "Historical Ophthalmology Series (2014 – 2021 OCT, Referral & B-Scans)", "badge": "b-amber", "status": "HISTORICAL • 2014–2021", "metrics": ["2014–2016 Optovue OCT: Early glaucomatous/ischemic RNFL cupping in OS (78.36 -> 71.41 um) + OD CME/RPE disruption.", "2019-08-20 Optopol OCT: Severe OS RNFL thinning (64 um red) BEFORE Lt RD vitrectomy; 2019-08-19 MOH referral notes prior Rt RD surgery.", "2019–2021 B-Scans: Both OD (emulsified oil, Apr 2021) &amp; OS (Dec 2019, Apr/May 2021) had silicone oil &amp; attached retina; Sept 2021 MOH &lt;1/60 decree."]},
                    {"hist": True, "title": "Historical Lower Limb Color Doppler (2011-06-21)", "badge": "b-amber", "status": "HISTORICAL • Jun 2011", "metrics": ["Altadamon Hospital (Dr. Helmy Rashad): Normal triphasic/biphasic lower-limb arterial waveforms (NO peripheral arterial occlusion).", "Patent bilateral SFV, popliteal &amp; tibial deep veins (NO DVT).", "Mild left calf perforator incompetence &amp; superficial varicosities only — valuable historical vascular baseline."]},
                ],
                "chartsHead": "2. Quantitative Telemetry Visualizations (Holter 24h & Definitive 2026-11-03 Audiology)",
                "holterChartTitle": "24-Hour Holter Circadian PVC Surge (857 Total • 36.29% Cluster 06–09 AM) & HR Profile",
                "holterLegend1": "Hourly PVCs (857 total; peak 164 @ 08:00 = 3.78%)",
                "holterLegend2": "Mean Hourly HR (Min 51 bpm @ 02:00 / Max 99 bpm)",
                "holterSurgeLbl": "36.29% AM SURGE (80 IU/d Insulin)",
                "holterStLbl": "CH3 -2.1 mm ST @ 13:11",
                "holterFoot": "Summary: 95,454 beats | 857 VEs (22 pairs, 4 runs) | 153 SVEs | SDNN 101 ms | Max QTc 326 ms",
                "audioChartTitle": "Definitive 2026-11-03 Masked Audiogram (0 dB ABG) & GSI-39 Peak Comparison (2024 vs 2026)",
                "audioAxisFreq": "Frequency (Hz) / GSI-39 Tymp",
                "audioRtLbl": "Rt Masked AC=BC (3f 40.0 dB | WRS 84%)",
                "audioLtLbl": "Lt Masked AC=BC (3f 54.2 dB | WRS 76%)",
                "audioTympHead": "GSI-39 Compliance (cm3)",
                "audioTympSub": "2024 vs 2026-11-03 Peak",
                "audioLimitLbl": "3.0 cm3 Plot Limit",
                "audioRefute": "1.5 refutes handwritten 'Rt As'",
                "roiHead": "3. Chronological Clinical Imaging & Auto-Cropped Ophthalmology / Vascular Plates (Click Any Plate to Zoom)",
                "galFilters": [
                    {"grp": "ALL", "lbl": "All Plates"},
                    {"grp": "NEURO_AUDIO_2026", "lbl": "2026 Neuro / Audio / Duplex"},
                    {"grp": "OPHTH_2026", "lbl": "2026 Current Ophthalmology"},
                    {"grp": "OPHTH_HIST", "lbl": "⚠️ Historical Ophthalmology (2014–2021)"},
                    {"grp": "VASC_HIST", "lbl": "⚠️ Historical 2011 Leg Doppler"},
                    {"grp": "HIST_2024", "lbl": "⚠️ Historical 2024 CT/EEG & 2021 MOH"},
                ],
                "roiBtn": "Inspect High-Res Zoom",
                "modalBtns": {"z1": "100%", "z2": "160%", "z3": "240%", "open": "Open Original", "close": "Close"},
                "filesHead": f"4. Complete Workspace File Index ({file_count} Verified Semantic Files)",
                "searchPh": "Filter by keyword, date, eye, Topcon, CSO VEP, ERG, Doppler, Holter, EEG...",
                "fileFilters": [
                    {"grp": "ALL", "lbl": "All Files"},
                    {"grp": "OPHTH_2026", "lbl": "2026 Ophthalmology"},
                    {"grp": "OPHTH_HIST", "lbl": "⚠️ Hist. Ophth (2014–21)"},
                    {"grp": "VASC_LL", "lbl": "⚠️ 2011 Leg Doppler"},
                    {"grp": "RAD_MRI_2026", "lbl": "2026 MRI/MRA/MRV"},
                    {"grp": "AUD_2026", "lbl": "2026 Audiology"},
                    {"grp": "CARD", "lbl": "24h Holter"},
                    {"grp": "VASC", "lbl": "Carotid Duplex"},
                    {"grp": "RAD_CT_2024", "lbl": "⚠️ 2024 CT Brain"},
                    {"grp": "NEURO_EEG_2024", "lbl": "⚠️ 2024 2h EEG"},
                    {"grp": "CLIN", "lbl": "Notes/MOH"},
                ],
                "fileCols": ["#", "Cat", "Date / Status", "Original Raw Name", "Renamed Semantic File (Click to Open)", "Size", "One-Line Clinical Summary"],
                "fileFoot": f"All {file_count} structured primary files across 2011–2026 reside in 01_Structured_and_Renamed_Inputs/ with untouched originals safely archived in 00_Raw_Backup_Archive/ and pre-2026 exams flagged ⚠️ HISTORICAL.",
            },
            # Tab 4: Prior Consultants & Clinical Conflict
            "t4": {
                "head": "Prior External Consultations & Forensic Diagnostic Conflict Resolution",
                "sub": "Direct side-by-side audit of why prior external evaluations diverged over Catheter DSA, Vertigo testing, Tremor etiology, Tympanometry, and Eye Prognosis — and the four hidden forensic pitfalls exposed by the MDT Board.",
                "priorCards": [
                    {
                        "role": "Prior Neurosurgery Consultation",
                        "doctor": "Prof. Dr. Ayman Elsayed Ali Galhom (2026-11-02 & 2026-11-03, No. 25334)",
                        "badge": "b-amber",
                        "tag": "Pro-DSA",
                        "stance": "Suspected Occult Low-Flow Occipital dAVF -> Recommended DSA + Extensive Vascular Workup; Documented Essential Tremor",
                        "points": [
                            "<strong>Occipital dAVF Hypothesis:</strong> Posterior neck/suboccipital manual pressure abolished the left-sided pulsatile bruit, raising suspicion for a Cognard Type I dural AVF missed by routine 1.5T MRI/TOF MRA.",
                            "<strong>Workup Ordered (2026-11-02):</strong> Ordered MRI Brain, MRV, MRA, Carotid/Vertebral Color Duplex, and 24h Holter ECG.",
                            "<strong>Movement Exam &amp; Rx (2026-11-03):</strong> Logged <code>Essential tremor + Titubation</code> (negative lower-limb involvement) alongside DM/HTN; prescribed <code>Inderal 10 mg BID</code>, <code>Ciprapra 10 mg</code> 1/2 tab, <code>Cobal F</code>, and <code>Brain-Ox</code>.",
                        ],
                    },
                    {
                        "role": "Prior Neuro-Interventional Opinion",
                        "doctor": "External Interventional Radiology Assessment",
                        "badge": "b-blue",
                        "tag": "Anti-DSA",
                        "stance": "Advised AGAINST Immediate Catheter Angiography (DSA)",
                        "points": [
                            "<strong>Imaging &amp; Vascular Rationale:</strong> Cited unremarkable 3D-TOF MRA (no dilated external carotid feeders or cortical venous ectasia), right-dominant venous asymmetry, and diabetic procedural morbidity.",
                            "<strong>Clinical Gap Left Unresolved:</strong> Could not reconcile distressing pulse-synchronous tinnitus or response to posterior neck pressure without a structured non-invasive gatekeeper protocol.",
                        ],
                    },
                    {
                        "role": "Prior ENT / Otology Consultation",
                        "doctor": "Dr. Mohamed Salem Abdel-Maksoud (2026-11-03 Prescription)",
                        "badge": "b-purple",
                        "tag": "ENT / VNG",
                        "stance": "Suspected Vascular Cause Behind Left Ear; Prescribed Meclizine + Ginkgo & Requested VNG",
                        "points": [
                            "<strong>Clinical Localization:</strong> Correctly suspected a retrotympanic/vascular abnormality behind the left ear driving pulsatile tinnitus and imbalance.",
                            "<strong>Hazardous Order Gaps:</strong> Prescribed daily <code>Meclizine qPM x 1 month</code> + <code>Cetrovit Ginkgo qAM x 2 months</code> and ordered routine optical VNG without flagging <code>&lt; 1/60</code> legal blindness or concurrent <code>Aggrex 75 mg</code>.",
                        ],
                    },
                    {
                        "role": "Disconnected Ophthalmology Records (2014–2026)",
                        "doctor": "Multiple External Eye Centers + Rology MRI Report Omission",
                        "badge": "b-red",
                        "tag": "Topcon Qual 2 Trap",
                        "stance": "Unlinked 2019–2021 Silicone Oil History, Rology Omission & Misleading '90 um' OS Topcon Noise",
                        "points": [
                            "<strong>MRI Report Blind Spot:</strong> Formal 2026-11-05 Rology PDF completely omitted prominent <code>2.59x</code> Left Intraocular Silicone Oil &amp; optic atrophy.",
                            "<strong>Quality-2 Segmentation Illusion:</strong> June 2026 Topcon prints <code>90 um</code> green/amber ONH noise for OS at <code>Qual 2</code> when true OS RNFL was already <code>64 um</code> back in 2019 &amp; July 2026 CSO ERG OPs collapsed to <code>0.74 uV</code> — yet CSO Flash VEP remains measurable (<code>7.99 uV</code>)!",
                        ],
                    },
                ],
                "bridgeTitle": "Four Forensic Pitfalls Uncovered Exclusively by the 7-Consultant MDT Synthesis",
                "bridgeCards": [
                    {
                        "title": "1. The Dual-Vessel Neck Compression Pitfall",
                        "cls": "alert-amber",
                        "txt": "Pressing deeply over suboccipital tissues compresses <strong>BOTH the superficial occipital artery feeder AND the underlying Internal Jugular Vein (IJV)</strong> simultaneously. Selective light anterior IJV pressure vs pinpoint occipital artery pressure is required.",
                    },
                    {
                        "title": "2. Hidden Procedural & Metabolic Landmines Behind DSA",
                        "cls": "alert-red",
                        "txt": "Prior DSA debates predated the <strong>5 mm Left Carotid Bulb Plaque</strong> right on the left ECA catheter route, <strong>-2.1 mm CH3 ST depression + 3.78% AM PVC surge</strong>, and unknown eGFR on <strong>Metformin (Gliptus Plus)</strong>.",
                    },
                    {
                        "title": "3. The Blindness-VNG & Pharmacologic Trap",
                        "cls": "alert-blue",
                        "txt": "Unreported legal blindness (<code>&lt; 1/60</code>) renders optical VNG invalid. Daily <code>Meclizine</code> + <code>Brain-Ox</code> + triple platelet blockade (<code>Aggrex + Ginkgo + Ciprapra</code>) + non-selective <code>Inderal</code> on <code>80 IU/d</code> insulin creates fall, bleed &amp; hypoglycemia risks.",
                    },
                    {
                        "title": "4. Topcon Qual 2 '90 um' Illusion vs Preserved CSO Flash VEP",
                        "cls": "alert-amber",
                        "txt": "Never read June 2026 OS Topcon <code>Qual 2</code> noise (90 um / 400.3 um) as healthy eye structure when July 2026 <code>ERG OPs = 0.74 uV</code>; conversely never give up completely because <code>Flash VEP = 7.99 uV OS / 10.29 uV OD</code> proves live optic nerve transmission requiring IOP/2026 B-Scan protection!",
                    },
                ],
            },
            # Tab 5: 7-Consultant MDT Board (Conclusions & Action Plans)
            "t5": {
                "head": "7-Consultant MDT Board: Conclusions, Immediate Actions & Master Answers",
                "sub": "Every consultant strictly follows the 3-block clinical protocol (1. Conclusion, 2. Immediate Action, 3. Detailed Action Plan), anchored by direct answers to the 3 decisive patient/family questions.",
                "allBtn": "Show All 7 Consultants",
                "blkConcLbl": "1. Conclusion",
                "blkImmLbl": "2. Immediate Action",
                "blkPlanLbl": "3. Detailed Action Plan",
                "tieLbl": "Cross-System Integration:",
                "qaHead": "Definitive MDT Board Answers to the 3 Central Clinical Questions",
                "qaCards": [
                    {
                        "cls": "qa-blue",
                        "badge": "b-blue",
                        "tag": "Hearing Loss Solution",
                        "title": "A. Solving Asymmetric Cochlear SNHL & Pulsatile Tinnitus",
                        "bullets": [
                            "<strong>Diagnosis:</strong> Definitive <code>2026-11-03</code> Masked PTA confirms <strong>0 dB ABG pure cochlear SNHL</strong> (Rt <code>40.0 dB / WRS 84%</code>, Lt <code>54.2 dB / WRS 76%</code>) + GSI-39 proves Rt <code>1.5 cm3</code> Type A (refuting handwritten 'Rt As') and Lt <code>3.2 cm3</code> Extreme Type Ad amplifying venous hum (<code>L/R 0.48</code>).",
                            "<strong>Exact Prescription:</strong> Fit <strong>Bilateral Open-Fit WDRC RIC Hearing Aids</strong> with <strong>soft vented / open domes</strong> (prevents occlusion amplification of venous hum), strict <strong>Left OSPL90 &lt;= 95–100 dB SPL</strong> honoring positive Metz loudness recruitment (<code>+45 dB SL</code>), and low-level tinnitus sound enrichment.",
                            "<strong>Entwined Medical Care:</strong> Stop <code>Meclizine</code> &amp; <code>Ginkgo</code> today; treat bilateral ethmoiditis with <code>Mometasone 50 mcg</code> + saline; binaural fitting acts as vital spatial safety navigation under <code>&lt; 1/60</code> blindness.",
                        ],
                    },
                    {
                        "cls": "qa-red",
                        "badge": "b-red",
                        "tag": "Left Eye Hope?",
                        "title": "B. Honest Visual Prognosis: Is There Hope for the Left Eye?",
                        "bullets": [
                            "<strong>The Candid Reality:</strong> Across 12 years (<code>2014 -> 2016 -> 2019 RNFL 64 um -> 2021 siliconized B-Scans</code>), functional reading or independent navigation recovery in <strong>OS is very unlikely</strong>. June 2026 Topcon <code>Qual 2</code> ('90 um') is scatter artifact, July 2026 CSO <strong>ERG OPs collapsed to 0.74 uV (vs 10.0 uV OD)</strong>, and routine OS oil removal after ~7 yrs risks hypotony/re-detachment.",
                            "<strong>Where Real Actionable Hope Exists:</strong> <code>2026-07-08</code> <strong>CSO Flash VEP is MEASURABLE BILATERALLY (OS N1-P1 7.99 uV / P2 132.4 ms; OD 10.29 uV / P2 125.4 ms)</strong> — optic-nerve-to-cortex wiring is NOT extinguished! MRI <code>Film10</code> optic chiasm is normal.",
                            "<strong>Immediate Eye Plan:</strong> Order an urgent <strong>Fresh 2026 Bilateral Ocular B-Scan + Applanation IOP + Slit-Lamp/Gonioscopy</strong> to confirm OS retina stays attached and rule out treatable <strong>emulsified silicone oil glaucoma</strong> (topical drops if needed; avoid systemic <code>Cidamex</code>!); guard disc perfusion (<code>BP &gt;= 115/70</code>) and focus low-vision support on <strong>OD (clear Qual 10 media, 259 um fovea, 40 um RNFL)</strong>.",
                        ],
                    },
                    {
                        "cls": "qa-amber",
                        "badge": "b-amber",
                        "tag": "Neurology & DSA",
                        "title": "C. Neurology, Mixed Tremor & Catheter DSA Deferral",
                        "bullets": [
                            "<strong>MS &amp; Epilepsy Excluded:</strong> Zero McDonald 2017 Dawson fingers (MRI shows hypertensive-diabetic CSVD Fazekas 1–2 + chronic Rt BG lacune); <code>2024 2h EEG (9 epochs)</code> proves 9 Hz alpha + somatic tremor EMG only, ZERO spikes.",
                            "<strong>Mixed Tremor Driver:</strong> Combine <code>Essential Tremor + Titubation</code> (Prof. Galhom), vascular BG lacune, and <strong>suspected morning hypoglycemia on 80 IU/d insulin</strong> matching the <code>36.29%</code> (<code>311/857</code>) Holter PVC cluster at <code>06:00–08:59 AM</code>! Run off/on <code>Sinemet MDS-UPDRS</code> + <strong>14-day CGM</strong>.",
                            "<strong>Defer Direct DSA:</strong> Crossing the <strong>5 mm Left Carotid Bulb plaque</strong> alongside <code>-2.1 mm</code> silent CH3 ischemia and unchecked eGFR on Metformin is unsafe. Enforce the <strong>3-Step Gatekeeper</strong> (selective IJV vs occipital compression -> 4D-MRA / HRCT+CTA after eGFR + 48h Metformin hold -> DSA only if objective shunt persists).",
                        ],
                    },
                ],
                "rankHead": "Ranked Multidisciplinary Consensus Diagnosis (Priority 01 to 09)",
                "rankedDx": [
                    {"rank": "01", "cls": "b-blue", "cat": "Cerebrovascular", "title": "Hypertensive-Diabetic Cerebral Small Vessel Disease (CSVD, Fazekas 1–2) & Chronic Right Lentiform Lacune", "evidence": "Proven on Film07/09/10 with ZERO acute diffusion restriction on Film05 DWI/ADC. MS &amp; cortical epilepsy are definitively EXCLUDED."},
                    {"rank": "02", "cls": "b-red", "cat": "Cardiovascular", "title": "High-Risk Carotid Atherosclerosis (IMT 0.13 cm, 5 mm Lt Bulb Plaque) + Silent Myocardial Ischemia (-2.1 mm ST)", "evidence": "Duplex (2026-11-07) + Holter CH3 (160 ST min, max -2.1 mm @ 13:11). Establishes shared endothelial microangiopathy linking heart, brain, retina &amp; inner ear."},
                    {"rank": "03", "cls": "b-red", "cat": "Ophthalmology", "title": "Bilaterally Vitrectomized Eye Disease (&lt; 1/60 Blindness): OD Optic Atrophy (40 um, Qual 10) & OS Siliconized Ischemia (OPs 0.74 uV) + Preserved Flash VEP", "evidence": "12-year longitudinal archive (2014–2021) + June–July 2026 Topcon/Zeiss/CSO testing proves inner-retinal ischemic collapse OS with preserved cortical VEP transmission OU (OD 10.29 uV / OS 7.99 uV) requiring urgent 2026 B-Scan + IOP check."},
                    {"rank": "04", "cls": "b-red", "cat": "Arrhythmia / Metabolic", "title": "Circadian Morning Ventricular Ectopy Surge (857 PVCs, 4 V-Runs) & 153 SVEs on 80 IU/d Insulin", "evidence": "36.29% of all 24h PVCs cluster 06:00–08:59 AM (peaking at 164 PVCs/hr = 3.78% burden &amp; 104 bpm V-run @ 08:00 AM), tightly suspicious for occult morning hypoglycemia and/or HCTZ K+ depletion."},
                    {"rank": "05", "cls": "b-amber", "cat": "Venous / Neuro-Otology", "title": "Left Pulsatile Venous Hum (Attenuated Lt TS/Sigmoid/IJV, L/R 0.48) Amplified by 3.2 cm3 Type Ad Eardrum", "evidence": "Zero dural sinus thrombosis on 3D-PCA MRV. Post-PVC pulse surges traverse a narrowed venous conduit beside an ultra-flaccid healed TM (&gt;3.0 cm3 plot ceiling); occult low-flow dAVF kept on conditional Gatekeeper hold."},
                    {"rank": "06", "cls": "b-red", "cat": "Balance / Multi-Sensory", "title": "'4-Hit' Multi-Sensory Vestibular Failure & High Fall-Risk Instability", "evidence": "Concurrence of (1) Lt inner-ear microvascular injury, (2) MOH &lt;1/60 blindness, (3) CSVD + chronic daily Meclizine sedation, and (4) glycemic/arrhythmic dips."},
                    {"rank": "07", "cls": "b-blue", "cat": "Audiology", "title": "Bilateral Asymmetric Pure Cochlear SNHL (0 dB ABG) — Resolved 'Rt As' Artifact", "evidence": "2026-11-03 Masked PTA (Rt 40.0 dB / WRS 84%; Lt 54.2 dB / WRS 76%) + historical Metz +45 dB SL. Primary GSI-39 proves Rt Peak 1.5 cm3 Normal Type A, overturning handwritten 'Rt As'."},
                    {"rank": "08", "cls": "b-purple", "cat": "Movement", "title": "Mixed Tremor Syndrome (Essential Tremor + Titubation + BG Lacune + Adrenergic Dips)", "evidence": "Reconciles prior PD/Sinemet prescribing with Prof. Galhom's Essential Tremor + Titubation exam and benign 2024 2h EEG (EMG tremor artifacts only), awaiting off/on Sinemet MDS-UPDRS verification."},
                    {"rank": "09", "cls": "b-green", "cat": "Sinus & Peripheral Vasc", "title": "Bilateral Ethmoid Sinusitis & Historical Lower-Limb Perforator Incompetence (2011 — PAD/DVT Free)", "evidence": "MRI Films 07/09/10 confirm ethmoiditis; 2011-06-21 Lower Limb Doppler confirms normal lower-limb arteries and deep veins historically."},
                ],
                "dilHead": "Scientific Resolution of the 4 Core Clinical Dilemmas",
            },
            # Tab 6: Master Treatment & Medication Plan
            "t6": {
                "head": "Master Treatment Plan, Pharmacotherapy Audit, Eye/Ear Protocols & Shareable PDFs",
                "resetBtn": "Reset Checklist",
                "checkHead": "1. Interactive Phase 1–2 Diagnostic & Clinical Workup Checklist (Saved Locally)",
                "doneWord": "Completed",
                "haHead": "2. Hearing Aid, Blindness-VNG & Optic Disc Preservation Guardrails",
                "haPoints": [
                    "<strong>Bilateral Open-Fit WDRC RIC Fitting:</strong> Mandatory bilateral amplification (spatial navigation under &lt;1/60 blindness) with <strong>soft vented / open silicone domes</strong> (never sealed molds) + <strong>Left OSPL90 &lt;= 95–100 dB SPL</strong> (Metz +45 dB SL).",
                    "<strong>Written VNG Safety Mandate:</strong> Notify vestibular lab explicitly of bilateral &lt;1/60 blindness — CANCEL saccade/pursuit/OPK light tests and perform <strong>dark positional + bithermal calorics ONLY</strong>.",
                    "<strong>Urgent Ophthalmology &amp; Perfusion Guardrails:</strong> Order <strong>Fresh 2026 Ocular B-Scan + Applanation IOP + Gonioscopy</strong>; use topical drops ONLY if IOP raised (NEVER systemic <code>Cidamex</code> atop HCTZ!); keep systemic BP &gt;= <code>115/70 mmHg</code> and avoid nocturnal hypoglycemia to protect the <code>40 um</code> optic nerve; check IOP 4–6 wks after starting nasal Mometasone.",
                ],
                "auditHead": "3. Pharmacotherapy & Drug-Interaction Safety Audit (8 High-Risk Nodes)",
                "auditFilters": [
                    {"tag": "ALL", "lbl": "All (8)"},
                    {"tag": "STOP", "lbl": "Stop / Avoid"},
                    {"tag": "CAUTION", "lbl": "High Caution / Switch"},
                    {"tag": "UPGRADE", "lbl": "Upgrade / Add"},
                ],
                "auditCols": ["Medication / Combination", "Action", "Pharmacological Mechanism", "Patient-Specific Hazard", "MDT Resolution"],
                "rxHead": "4. Modified Prescription & Multidisciplinary Strategy Matrix (All 6 Domains)",
                "rxCols": ["Clinical Domain", "Modified Regimen / Intervention", "Timing & Monitoring", "Clinical Rationale & Safety Guardrails"],
                "pdfSectionHead": "5. Shareable Consultant PDF Reports & Master Documentation",
                "pdfSectionSub": "Updated 7-Consultant board deliverables incorporating all 69 verified files (2011–2026), auto-cropped Ophthalmology/Doppler plates, and 3-block action plans.",
            },
        },

        "ar": {
            "pageTitle": "دراسة الحالة السريرية متعددة التخصصات — آمال مختار محمد إبراهيم (63.5 سنة)",
            "brandTitle": "لجنة الاستشاريين السبعة | تقييم الأوعية المخية، الأذن والاتزان، والشبكية والعصب البصري",
            "brandSub": f"المريضة: آمال مختار محمد إبراهيم • أنثى 63.5 سنة • الرقم القومي: 26304090300188 • {file_count} ملفاً موثقاً (2011–2026)",
            "btnThemeDark": "الوضع الليلي",
            "btnThemeLight": "الوضع النهاري",
            "btnPrintTab": "طباعة القسم",
            "btnPrintAll": "الملف الكامل (PDF)",
            "navLabel": "رحلة دراسة الحالة",
            "pdfLabel": "التقارير والفهرس القابل للمشاركة",
            "pdfLinks": [
                {"href": "../05_Shareable_PDF_Reports/01_MDT_Executive_Summary_and_Action_Plan.pdf", "txt": "01. الملخص التنفيذي صفحتين (PDF)"},
                {"href": "../05_Shareable_PDF_Reports/02_Consultant_7_Board_Audit_Reports.pdf", "txt": "02. تقارير الاستشاريين السبعة (PDF)"},
                {"href": "../05_Shareable_PDF_Reports/03_Patient_Case_Study_Full_Report.pdf", "txt": "03. الدراسة الشاملة الكاملة (PDF)"},
                {"href": "../04_Final_MDT_Deliverables/Master_MDT_Case_Study_Report.md", "txt": "التقرير المرجعي المجمع (MD)"},
                {"href": "../01_Structured_and_Renamed_Inputs/FILE_RENAMING_INDEX.md", "txt": f"فهرس الملفات الكامل ({file_count} ملفاً)"},
            ],
            "tabs": [
                {"id": "tab-profile", "num": "01", "label": "01. البروفايل والتاريخ المرضي", "count": "البروفايل"},
                {"id": "tab-videos", "num": "02", "label": "02. فيديوهات الرنين المغناطيسي", "count": "3 فيديو"},
                {"id": "tab-diagnostics", "num": "03", "label": "03. الفحوصات والإشعاعات والتحاليل", "count": str(file_count)},
                {"id": "tab-prior", "num": "04", "label": "04. آراء الأطباء السابقين والخلاف التشخيصي", "count": "الخلاف"},
                {"id": "tab-board", "num": "05", "label": "05. لجنة الاستشاريين السبعة", "count": "7 أطباء"},
                {"id": "tab-plan", "num": "06", "label": "06. الخطة العلاجية والأدوية", "count": "العلاج"},
            ],
            "histBadge": "فحص قديم / تاريخي — يُرجى عدم خلطه مع الحالة الراهنة في 2026",
            "histShortBadge": "فحص قديم / تاريخي",
            "currBadge": "فحص حالي 2026",
            "t1": {
                "jumpTitle": "جناح فيديوهات الرنين والشرايين والأوردة جاهز في القسم 02",
                "jumpSub": "مشغلات فيديو مدمجة لأوردة المخ MRV (نسبة الشمال 0.48) وشرايين MRA (513 شريحة) ورنين T2/FLAIR متاحة مباشرة في القسم 02.",
                "jumpBtn": "فتح فيديوهات الرنين المغناطيسي",
                "secHead": "البيانات الديموغرافية وعوامل الخطورة والتسلسل الزمني الكامل (2011 – نوفمبر 2026)",
                "kpis": [
                    {"cls": "accent", "lbl": "الهوية والعمر الموثق", "val": "أنثى 63.5 سنة", "sub": f"ميلاد 1963-04-09 (قومي ...00188)؛ مسجل إدارياً 27-03 على الفيلم؛ {file_count} ملفاً"},
                    {"cls": "danger", "lbl": "الإبصار وتاريخ 12 سنة", "val": "< 1/60 | يمين 40µm", "sub": "يمين عدسة صافية (جودة 10) وضمور عصب 40µm؛ شمال سيليكون (جودة 2) وOPs 0.74µV مع بقاء VEP"},
                    {"cls": "warn", "lbl": "الأيض والشرايين", "val": "80 وحدة/يوم | 0.13 سم", "sub": "سكري وضغط ودهون؛ بلاك 5 مم بالسباتي الشمال (دوبلر الساق 2011 خالٍ من الجلطات)"},
                    {"cls": "danger", "lbl": "الهولتر والسمعيات", "val": "857 PVC | 3.2 سم³", "sub": "طفرة 36.29% صباحاً وهبوط ST -2.1 مم؛ MRV 0.48؛ فجوة سمعية 0 ديسيبل"},
                ],
                "safetyTitle": "تنبيهات أمان قصوى غير قابلة للتجاوز قبل أي دواء أو أشعة",
                "safetyBullets": [
                    "<strong>بطلان فحص اتزان العين الضوئي (VNG):</strong> فقدان البصر الشديد (&lt; 1/60 بالعين الأفضل) وضمور العصب <code>40 µm</code> وزيت السيليكون بالشمال يلغون تتبع الضوء، لذا يجب اقتصار VNG على <strong>الوضعيات والحراري في الضلمة فقط</strong>.",
                    "<strong>تأجيل قسطرة المخ (DSA) عبر عنق السباتي الشمال:</strong> القسطرة الفورية تعبر <code>بلاك 5 مم</code> بالسباتي الشمال قبل فحص الكلى (مع الميتفورمين) وفي وجود قصور تاجي صامت <code>-2.1 مم ST</code>.",
                    "<strong>ممنوع إزالة زيت السيليكون الشمال بناءً على رقم التوبكون الخاطئ:</strong> جودة <code>Qual 2</code> في اليسرى تعطي ضوضاء <code>90 µm</code> كاذبة (بينما العصب كان <code>64 µm</code> من 2019)، وإزالة الزيت بعد ~7 سنوات تخاطر بالارتخاء/الانفصال؛ اطلب <strong>سونار B-Scan 2026 + ضغط عين</strong> أولاً.",
                    "<strong>دوامة الأنسولين الصباحي والأملاح:</strong> وقف <code>Meclizine</code> و<code>Ginkgo</code> فوراً، منع دمج <code>Cidamex</code> مع الثيازيد، فحص هبوط سكر <code>6–9 صباحاً</code> (<code>80 وحدة/يوم</code>) المحفز لـ 36.29% من الخفقات، وحماية ضغط العين (&gt;= 115/70).",
                ],
                "compHead": "1. نطاقات الشكوى الإكلينيكية والتشخيص المتكامل (5 محاور)",
                "complaints": [
                    {"title": "طنين نبضي ورا الأذن الشمال", "badge": "b-amber", "tag": "أولوية وعائية/صوتية", "desc": "صوت نبض مستمر يشتد ليلاً؛ ناتج عن اندفاع الدم في جيب وريدي ضيق خلقياً (<code>L/R 0.48</code>) مع تضخيم ميكانيكي عبر طبلية شمال مرنة جداً <code>3.2 سم³ (Type Ad)</code> ونبضات الهولتر المتفاوتة."},
                    {"title": "عدم اتزان ودوخة شديدة (متلازمة 4 ضربات)", "badge": "b-red", "tag": "انهيار متعدد الحواس", "desc": "تلاقي (1) ضعف اتزان أذن شمال مزمن، (2) كف بصر &lt; 1/60 موثق في 2026، (3) تصلب مادة بيضاء + التهاب أعصاب + تثبيط <code>Meclizine</code>، و(4) هبوط التروية من القلب والسكر."},
                    {"title": "ضعف سمع قوقعي غير متماثل مستقر", "badge": "b-blue", "tag": "تشخيص قوقعي", "desc": "قياس السمع المقنع الحديث (<code>2026-11-03</code>) يثبت <strong>فجوة 0 ديسيبل</strong> بالجهتين (يمين <code>40.0 ديسيبل، تمييز 84%</code>؛ شمال <code>54.2 ديسيبل، تمييز 76%</code>) مع حساسية صوت <code>Metz +45 dB SL</code> — علاجه <strong>سماعتان Open-Fit WDRC RIC</strong> بحد <code>&lt;= 95-100 dB</code>."},
                    {"title": "رعشة باليدين وهزة بالرأس (ET + Titubation)", "badge": "b-purple", "tag": "حركي وأيضي", "desc": "وثق أ.د. أيمن جلهوم (<code>2026-11-03</code>) وجود <code>Essential tremor + Titubation</code> مع سلبية الطرفين السفليين على عكس وصف السينيمت، وتخطيط مخ 2024 سجل ذبذبات عضلية للرعشة فقط دون صرع."},
                    {"title": "فقدان البصر الشديد وسؤال الأمل في العين الشمال", "badge": "b-red", "tag": "الشبكية والعصب البصري", "desc": "تطور ضمور عصب واعتلال شبكية من <code>2014–2019</code> إلى عمليات انفصال شبكي وزيت سيليكون بالعينين (<code>2019–2021</code>). فحوصات 2026 تثبت <code>40 µm RNFL</code> باليمين وانهيار <code>OPs 0.74 µV</code> بالشمال مع <strong>بقاء إشارة العصب في Flash VEP بالعينين</strong>."},
                ],
                "timelineHead": "2. التسلسل الزمني للأحداث والفحوصات الطبية (2011 – نوفمبر 2026)",
                "timeline": [
                    {"hist": True, "date": "2011-06-21", "src": "مستشفى التضامن (د. حلمي رشاد)", "detail": "فحص قديم: دوبلر شرايين وأوردة الساقين أثبت سلامة الشرايين تماماً (لا يوجد ضيق PAD) وسلامة الأوردة العميقة (لا جلطات DVT)، مع ارتجاع بسيط ودوالي سطحية بالساق الشمال فقط."},
                    {"hist": True, "date": "2014-04-23 -> 2016-02-16", "src": "أشعة Optovue OCT التاريخية (د. سناء حمام)", "detail": "فحص قديم: كشف وجود تآكل مبكر وضمور بالعصب البصري الشمال (<code>RNFL 78.36 µm</code> في 2014 ثم <code>71.41 µm</code> في 2016) وارتشاح وتلف مركز الإبصار باليمين قبل جراحات الانفصال الشبكي."},
                    {"hist": True, "date": "2019-08-19 -> 2019-12-15", "src": "خطاب التأمين وأشعة Optopol وسونار العين الشمال", "detail": "فحص قديم: خطاب التأمين وثّق عملية شبكية سابقة باليمين وطلب عملية الشمال؛ أشعة 2019 أثبتت ضموراً شديداً بالشمال (<code>64 µm</code>)، وسونار سبتمبر/ديسمبر 2019 تابع نزيفاً زجاجياً ثم حقن السيليكون بالشمال."},
                    {"hist": True, "date": "2021-04-07 -> 2021-09-19", "src": "سونار مستشفى الرمد وقرار القومسيون الطبي", "detail": "فحص قديم: سونار أبريل 2021 (د. ياسمين سنان) أثبت وجود زيت سيليكون بالعينين، وآخر سونار للشمال مايو 2021 (د. منار الجيار)؛ وقرار القومسيون سبتمبر 2021 أثبت رؤية <code>&lt; 1/60</code>."},
                    {"hist": True, "date": "2024-07-17 -> 2024-11-02", "src": "تخطيط مخ ساعتين ومقطعية 2024 وسمعيات 2024", "detail": "فحص قديم: تخطيط مخ 2024 (9 مقاطع) أظهر إيقاع ألفا 9 هيرتز وارتجافاً عضلياً دون صرع؛ مقطعية 2024 أثبتت زيت السيليكون الشمال؛ وسمعيات 2024 تم استبدالها بسمعيات 3 نوفمبر 2026 المقنعة."},
                    {"hist": False, "date": "2026-06-01 -> 2026-07-08", "src": "أشعة التوبكون 2026 والمجال البصري ورسم العصب والشبكية CSO", "detail": "حالي 2026: يمين صافية جودة 10 بضمور عصب <strong>40 µm</strong>؛ شمال جودة 2 رقم كاذب (90 µm) بسبب السيليكون. مجال الإبصار 18% (غير موثوق، د. آلاء خالد). رسم CSO أثبت انهيار استجابة الشمال <code>OPs 0.74 µV</code> <strong>لكن Flash VEP ما زال يقيس الإشارة بالعينين (10.29 / 7.99 µV)</strong>!"},
                    {"hist": False, "date": "2026-11-02 -> 2026-11-03", "src": "سمعيات المتابعة المؤكدة واستشارات المخ والأعصاب والأنف والأذن", "detail": "حالي 2026: فحص سمع مقنع (د. شاهندة): <code>0 dB ABG</code>، تمييز 84% يمين و76% شمال، وطبلية يمين <code>1.5 سم³ Type A</code> (يفند 'Rt As') وشمال <code>3.2 سم³ Type Ad</code>. د. جلهوم سجل رعشة أساس وهزة رأس؛ د. سالم طلب VNG ووصف ميكليزين وجينكو."},
                    {"hist": False, "date": "2026-11-05 -> 2026-11-07", "src": "رنين 1.5T للمخ والأوعية + هولتر 29 صفحة + دوبلر السباتي", "detail": "حالي 2026: تصلب شعيرات مخية (CSVD 1–2) وجلطة فجوية قديمة باليمين دون جلطات حادة، ضمور الجيب الوريدي الشمال (<code>0.48</code>)، وزيت سيليكون <code>2.59x</code>. هولتر: <code>857 PVC</code> (311 بين 6–9 صباحاً) وهبوط ST <code>-2.1 مم</code>. دوبلر: <code>IMT 0.13 سم</code> و<strong>بلاك 5 مم بالسباتي الشمال</strong>."},
                ],
                "medsHead": "3. جرد الأدوية الحالية والموصوفة مؤخراً مع تقييم الأمان",
                "medCols": ["الدواء والتركيز", "الجرعة والتوقيت", "المصدر / الطبيب", "الدور الإكلينيكي وحالة اللجنة", "القرار"],
                "currentMeds": [
                    {"drug": "Toujeo SoloStar (أنسولين قاعدي U300)", "dose": "30 وحدة مساءً", "src": "التأمين الصحى", "role": "ضمن جرعة أنسولين يومية كبيرة (80 وحدة) تستدعي فحص هبوط السكر الليلي/الصباحي.", "tag": "حذر", "badge": "b-amber"},
                    {"drug": "Apidra SoloStar (أنسولين سريع)", "dose": "50 وحدة/يوم قبل الوجبات", "src": "التأمين الصحى", "role": "حمل مرتفع (62.5% من الأنسولين)؛ المشتبه الأول في نوبات هبوط السكر والـ 311 خفقة صباحاً.", "tag": "حذر", "badge": "b-amber"},
                    {"drug": "Gliptus Plus (فيلداجليبتين 50 / ميتفورمين 1000)", "dose": "قرص مرتين يومياً", "src": "التأمين الصحى", "role": "ممنوع إذا قل eGFR عن 30؛ يجب إيقافه 48 ساعة قبل أي صبغة مقطعية أو قسطرة.", "tag": "فحص كلى", "badge": "b-amber"},
                    {"drug": "Angiopress Comp (كابتوبريل 50 / هيدروكلوروثيازيد 25)", "dose": "قرص يومياً", "src": "التأمين الصحى", "role": "مفعول الكابتوبريل قصير يترك الضغط يرتفع ليلاً، والثيازيد يستنزف البوتاسيوم ويشعل الخفقات.", "tag": "يُستبدل", "badge": "b-red"},
                    {"drug": "Cidamex (أسيتازولاميد 250 مجم فموي — تحذير)", "dose": "ممنوع بالفم", "src": "حاجز أمان اللجنة", "role": "يمنع دمجه مع الثيازيد لأنه يضاعف فقد البوتاسيوم والحماض والـ PVCs (القطرات الموضعية فقط عند اللزوم).", "tag": "يُمنع", "badge": "b-red"},
                    {"drug": "Zatorva 20 mg (أتورفاستاتين)", "dose": "20 مجم ليلاً", "src": "التأمين الصحى", "role": "جرعة غير كافية مع بلاك السباتي 5 مم وتصلب المخ وقصور القلب (-2.1 مم)؛ يُرفع لـ 40–80 مجم.", "tag": "رفع الجرعة", "badge": "b-blue"},
                    {"drug": "Aggrex 75 mg (أسبرين)", "dose": "75 مجم بعد الغداء", "src": "التأمين الصحى", "role": "أساس الوقاية من الجلطات؛ يستمر وحده بعد وقف الجينكو لمنع النزيف.", "tag": "يستمر", "badge": "b-green"},
                    {"drug": "Sinemet (كاربيدوبا / ليفودوبا)", "dose": "حسب الجرعة المعتادة", "src": "مخ وأعصاب سابق", "role": "يستمر مؤقتاً حتى إجراء تقييم سحب/إعطاء فعلي MDS-UPDRS لبيان مدى استفادتها الحقيقية.", "tag": "تقييم", "badge": "b-amber"},
                    {"drug": "Inderal 10 mg (بروبرانولول)", "dose": "قرص كل 12 ساعة", "src": "أ.د. جلهوم (03-11)", "role": "يخفي أعراض هبوط السكر (تسارع القلب) مع 80 وحدة أنسولين؛ يفضل التحويل للبيسوبرولول مع القلب.", "tag": "مراجعة", "badge": "b-amber"},
                    {"drug": "Ciprapra 10 mg (إسيتالوبرام)", "dose": "نصف قرص مساءً", "src": "أ.د. جلهوم (03-11)", "role": "يزيد سيولة الصفائح مع الأسبرين؛ تستمر الحاجة له فقط إن وُجد قلق/اكتئاب سريري واضح.", "tag": "تقييم", "badge": "b-amber"},
                    {"drug": "Cobal F (فيتامين B12)", "dose": "قرص صباحاً", "src": "أ.د. جلهوم (03-11)", "role": "داعم آمن للأعصاب الطرفية ضد نقص فيتامين B12 الناتج عن الميتفورمين.", "tag": "يستمر", "badge": "b-green"},
                    {"drug": "Brain-Ox 30 mg (فينكامين)", "dose": "كبسولة بعد الغداء", "src": "أ.د. جلهوم (03-11)", "role": "موسع شعيرات قد يسبب هبوط ضغط مفاجئ يضر تروية العصب البصري والقلب (-2.1 مم).", "tag": "إيقاف", "badge": "b-red"},
                    {"drug": "Meclizine 25 mg", "dose": "قرص مساءً لمدة شهر", "src": "أنف وأذن (03-11)", "role": "مثبط يومي لجهاز الاتزان يمنع تعويض المخ في مريضة كفيفة ويزيد خطر السقوط.", "tag": "وقف فوري", "badge": "b-red"},
                    {"drug": "Cetrovit Ginkgo (جينكو بيلوبا)", "dose": "قرص صباحاً شهرين", "src": "أنف وأذن (03-11)", "role": "لا يعالج ضعف السمع القوقعي ويصنع تداخلاً نزفياً خطيراً مع الأسبرين والسيبراپرا.", "tag": "وقف فوري", "badge": "b-red"},
                ],
            },
            "t2": {
                "videoSuiteTitle": "جناح فيديوهات الرنين والشرايين والأوردة (عارض الفيديو التفاعلي)",
                "videoBadges": ["رنين 1.5T ومقاطع الشرايين والأوردة (05-11-2026)", "متوافق مع البث المحلي والويب"],
                "jumpDiagTitle": "الانتقال إلى الفحوصات الكاملة، فهرس الـ 69 ملفاً ومعرض الـ 41 لوحة طبية",
                "jumpDiagSub": "يضم رسومات الهولتر والسمعيات، ولوحات العيون والدوبلر المقصوصة (2026 + القديم 2011–2024)، ومربع الفحوصات المعلقة المطلوبة.",
                "jumpDiagBtn": "فتح القسم 03: الفحوصات والمعرض",
                "videos": [
                    {
                        "title": "البطاقة 1: فيديو أوردة المخ (3D-MRV Venography)",
                        "badge": "مقطع مستخلص",
                        "badgeCls": "b-amber",
                        "src": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4",
                        "dlText": "تحميل / فتح الفيديو",
                        "secText": "فتح مستقل",
                        "secHref": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4",
                        "secDownload": False,
                        "desc": "يوضح <strong>ضيق/ضمور الجيب الوريدي المستعرض والسيني والوداجي الشمال</strong> (نسبة <code>0.48</code> مقارنة باليمين السائد) عبر 18 زاوية 3D-PCA بدون جلطات وريدية (من فيلم <code>Film06</code>).",
                    },
                    {
                        "title": "البطاقة 2: فيديو شرايين المخ (3D-TOF MRA 513 شريحة)",
                        "badge": "المصدر الأساسي",
                        "badgeCls": "b-green",
                        "src": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil_WebH264.mp4",
                        "dlText": "تحميل / فتح الفيديو",
                        "secText": "ملف MOV الخام",
                        "secHref": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil.mov",
                        "secDownload": True,
                        "desc": "يعرض <strong>سلامة الشرايين المخية الرئيسية ودائرة ويليس</strong> عبر 513 شريحة محورية دقيقة 1.0 مم دون تمددات أو ناسور عالي التدفق، ويُظهر <strong>زيت السيليكون الساطع (2.59x) بالعين الشمال</strong>.",
                    },
                    {
                        "title": "البطاقة 3: فيديو مقاطع الرنين (Axial T2/FLAIR MRI)",
                        "badge": "مقطع مستخلص",
                        "badgeCls": "b-amber",
                        "src": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4",
                        "dlText": "تحميل / فتح الفيديو",
                        "secText": "فتح مستقل",
                        "secHref": "../01_Structured_and_Renamed_Inputs/01_Neuroimaging_MRI_MRA_MRV_Videos/MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4",
                        "secDownload": False,
                        "desc": "يعرض <strong>الجلطات الفجوية القديمة بالعقد القاعدية</strong> وتصلب الشعيرات الدموية (<code>CSVD Fazekas 1–2</code>) والتهاب الجيوب الغربالية وسلامة قنوات السمع والتصالب البصري.",
                    },
                ],
                "seqHead": "التحليل الكمي المجمع لأفلام الرنين الـ 10 وشرائح MRA الـ 513",
                "seqCards": [
                    {"title": "هندسة الأوردة MRV (Film06 — موثق بالعلامات)", "badge": "b-amber", "tag": "نسبة L/R 0.48", "desc": "سريان وريدي سائد بالجانب الأيمن؛ الجانب الشمال ضامر/رفيع خلقياً بنسبة 0.48 مع عدم وجود أي جلطة وريدية أو تمدد بارز بعظمة الماستويد."},
                    {"title": "شرايين قاع الجمجمة TOF-MRA (Video01 + Films 02–04b)", "badge": "b-green", "tag": "بلا ناسور كبير", "desc": "سلامة الشرايين السباتية والقاعدية وهوائية عظمة الماستويد، دون فروع شريانية متسعة (مع ملاحظة محدودية 1.5T تجاه الناسور البطيء الدقيق جدا)."},
                    {"title": "نسيج المخ والعيون والتصالب البصري (Films 05, 07–10)", "badge": "b-blue", "tag": "CSVD + زيت 2.59x", "desc": "انتشار DWI سليم (لا جلطة حادة)؛ بؤر CSVD بسيطة وجلطة قديمة بالعقد القاعدية (لا MS)؛ تصالب بصري وقنوات سمع سليمة؛ التهاب جيوب غربالية وزيت سيليكون بالعين الشمال."},
                ],
            },
            "t3": {
                "pendingHead": "فحوصات مطلوبة / معلّقة قادمة (لا تخلط بينها وبين الفحوصات المنجزة أدناه)",
                "pendingItems": [
                    {"title": "اختبار اتزان VNG معدّل في الضلمة (تحذير إلزامي للمعمل)", "txt": "كتابة تنبيه رسمي للمعمل بأن المريضة كفيفة (&lt; 1/60): إلغاء تتبع النقطة الضوئية تماماً والاكتفاء باختبار <strong>الوضعيات والاختبار الحراري بالضلمة فقط</strong>."},
                    {"title": "سونار عيون B-Scan حديث 2026 + ضغط عين IOP + فحص زاوية", "txt": "آخر سونار بالملف كان في مايو 2021! يجب عمل <strong>سونار 2026 جديد للعينين وقياس الضغط</strong> للتأكد من ثبات الشبكية الشمال خلف عتامة السيليكون واستبعاد مياه زرقاء متسحلبة."},
                    {"title": "اختبار الضغط الانتقائي بالرقبة والتسمّع بالسماعة", "txt": "مقارنة التسمع خلف الأذن مع <strong>ضغط أمامي خفيف بإصبع على الوريد الوداجي</strong> مقابل <strong>ضغط نقطي على نبض الشريان القذالي</strong> قبل طلب أي أشعة بالصبغة."},
                    {"title": "تحاليل عاجلة للأملاح والكلى (قبل تعديل الأدوية أو الصبغة)", "txt": "<code>K+, Na+, Mg2+, Ca2+, Creatinine/eGFR, Urine ACR</code> فوراً قبل إيقاف الثيازيد أو إعطاء أي صبغة بجانب الميتفورمين."},
                    {"title": "منحنى سكر 14 يوماً (CGM 3–9 صباحاً) وتحاليل الأيض", "txt": "كشف هبوط السكر الصباحي على 80 وحدة أنسولين والمسبب لـ 36.29% من الخفقات، مع <code>HbA1c، دهون صائمة، CBC، TSH، B12، إنزيمات كبد</code>."},
                    {"title": "تقييم تروية القلب وتحدي السينيمت (MDS-UPDRS)", "txt": "<strong>إيكو قلب + مسح ذري/إيكو دوائي</strong> (ممنوع السير الكهربائي بسبب البصر!) بسبب هبوط ST -2.1 مم + اختبار <code>Sinemet</code> قبل/بعد الجرعة."},
                ],
                "diagHead": "1. مصفوفة الفحوصات الاستقصائية الشاملة (من التاريخي 2011 إلى الحالي 2026)",
                "investigations": [
                    {"hist": False, "title": "رنين المخ والشرايين والأوردة 1.5T (2026-11-05)", "badge": "b-blue", "status": "حالي • Film01–10", "metrics": ["CSVD Fazekas 1–2 وجلطة فجوية مزمنة باليمين؛ DWI سلبي (لا جلطة حادة، نفي MS).", "MRA: شرايين طبيعية وقنوات سمع وتصالب بصري سليم (Film10).", "MRV: ضيق الوريد الشمال L/R 0.48 بلا تجلط + التهاب جيوب غربالية + زيت سيليكون 2.59x بالشمال."]},
                    {"hist": False, "title": "السمعيات والطبلية المؤكدة المقنعة (2026-11-03)", "badge": "b-red", "status": "حالي • مرجع السمع", "metrics": ["التوصيل الهوائي والعظمي متطابق 0 dB ABG: يمين 40.0 dB (تمييز 84%) وشمال 54.2 dB (تمييز 76%).", "شريط GSI-39 يثبت يمين 1.5 سم³ Type A طبيعي (يكذب كتابة اليد 'Rt As').", "الشمال 3.2 سم³ Type Ad متطرف فوق سقف الورقة + Metz +45 dB SL يؤكد ضعفاً قوقعياً."]},
                    {"hist": False, "title": "فحوصات العيون الحديثة 2026: Topcon OCT و HFA و CSO VEP/ERG", "badge": "b-red", "status": "حالي • عيون 2026", "metrics": ["التوبكون (06-01): اليمين صافية جودة 10 وضمور عصب شديد RNFL 40 µm؛ الشمال جودة 2 ضوضاء 90 µm كاذبة.", "مجال الإبصار Zeiss (06-27): VFI 18% بتوقيع 'Not reliable Test' د. آلاء خالد لعدم ثبات التثبيت.", "فحص CSO (07-08): انهيار استجابة شبكية الشمال OPs 0.74 مقابل 10.0 µV يمين، مع بقاء Flash VEP بالعينين (10.29 / 7.99 µV)!"]},
                    {"hist": False, "title": "هولتر القلب 24 ساعة — 29 صفحة (2026-11-05)", "badge": "b-red", "status": "حالي • 95,454 نبضة", "metrics": ["متوسط النبض 65 ن/د (51 لـ 99)؛ SDNN 101 مللي ثانية؛ QTc طبيعي تماماً (أقصى 326 مللي ثانية).", "857 PVC (0.898%)، 22 زوجاً، 4 نوبات V-runs (حتى 104 ن/د الساعة 8 ص)، و153 SVE.", "تركز 36.29% صباحاً (311 خفقة 6–9 ص) + 160 دقيقة هبوط ST بالقناة 3 (ذروة -2.1 مم الساعة 1:11 ظ)."]},
                    {"hist": False, "title": "دوبلر شرايين الرقبة والسباتي (2026-11-07)", "badge": "b-amber", "status": "حالي • بلاك 5 مم", "metrics": ["زيادة سمك جدار السباتي IMT إلى 0.13 سم (+44.4% فوق الحد الطبيعي 0.09).", "لويحات متصلبة على الجانبين؛ أكبرها بلاك بطول 5 مم في عنق الشريان السباتي الشمال.", "خطورة صدمات قسطرة مباشرة عبر السباتي الشمال؛ تدفق طبيعي بالشرايين الفقارية."]},
                    {"hist": True, "title": "تخطيط المخ 2h (9 مقاطع) ومقطعية المخ (يوليو 2024)", "badge": "b-amber", "status": "قديم / تاريخي • 2024", "metrics": ["تخطيط المخ (2024-07-17): إيقاع ألفا منتظم 9 هيرتز وغياب تام لأي بؤر صرعية.", "وجود ذبذبات عضلية أمامية-صدغية تثبت رعشة جسمية وليست صرعاً قشرياً.", "مقطعية الزهور (2024-07-27): تغيرات سنية بسيطة وتؤكد وجود زيت السيليكون الشمال قبل الرنين بـ 27 شهراً."]},
                    {"hist": True, "title": "سجل العيون التاريخي (2014–2021 Optovue/Optopol/B-Scan)", "badge": "b-amber", "status": "قديم / تاريخي • 2014–21", "metrics": ["أوبتوڤيو 2014–2016: تآكل قديم بعصب الشمال (78.36 -> 71.41 µm) وارتشاح وتلف ماكولة اليمين.", "أوبتوبول أغسطس 2019: ضمور شديد بالشمال (64 µm) قبل جراحة الانفصال + خطاب تأمين يؤكد عملية اليمين سابقاً.", "سونار 2019–2021: توثيق زيت السيليكون واستقرار الشبكية بالعينين + شهادة القومسيون سبتمبر 2021 (&lt; 1/60)."]},
                    {"hist": True, "title": "دوبلر شرايين وأوردة الساقين (2011-06-21)", "badge": "b-amber", "status": "قديم / تاريخي • 2011", "metrics": ["مستشفى التضامن (د. حلمي رشاد): نبض شرياني ثلاثي وثنائي سليم (خالٍ من قصور الشرايين PAD).", "الأوردة العميقة الفخذية والمأبضية والظنبوبية مفتوحة تماماً (خالية من الجلطات DVT).", "ارتجاع بسيط بالثواقب ودوالي سطحية بالساق الشمال فقط — مرجع وعائي تاريخي قيّم."]},
                ],
                "chartsHead": "2. الرسوم البيانية الكمية (الهولتر 24 ساعة وسمعيات 3 نوفمبر 2026)",
                "holterChartTitle": "توزيع الخفقات البطينية عبر 24 ساعة (857 PVC • 36.29% صباحاً) ومنحنى النبض",
                "holterLegend1": "الخفقات لكل ساعة (الإجمالي 857؛ ذروة 164 الساعة 8 ص)",
                "holterLegend2": "متوسط النبض (الأدنى 51 ن/د الساعة 2 ص / الأقصى 99)",
                "holterSurgeLbl": "فورة الصباح 36.29% (80 وحدة أنسولين)",
                "holterStLbl": "هبوط ST -2.1 مم الساعة 1:11 ظ",
                "holterFoot": "الملخص: 95,454 نبضة | 857 PVC (22 زوجاً، 4 نوبات) | 153 SVE | SDNN 101 ms | أقصى QTc 326 ms",
                "audioChartTitle": "مخطط السمع المقنع الحديث 03-11-2026 (فجوة 0 dB) ومقارنة ضغط الطبلية",
                "audioAxisFreq": "التردد (Hz) / طبلية GSI-39",
                "audioRtLbl": "اليمين مقنّع AC=BC (متوسط 40.0 dB | تمييز 84%)",
                "audioLtLbl": "الشمال مقنّع AC=BC (متوسط 54.2 dB | تمييز 76%)",
                "audioTympHead": "مرونة الطبلية GSI-39 (سم³)",
                "audioTympSub": "مقارنة 2024 مع 03-11-2026",
                "audioLimitLbl": "سقف الورقة 3.0 سم³",
                "audioRefute": "1.5 سم³ تفند كتابة 'Rt As'",
                "roiHead": "3. معرض اللوحات الإشعاعية والعيون والدوبلر المقصوصة زمنياً (اضغط للتكبير)",
                "galFilters": [
                    {"grp": "ALL", "lbl": "كل اللوحات"},
                    {"grp": "NEURO_AUDIO_2026", "lbl": "2026 مخ وأعصاب وسمع وشرايين"},
                    {"grp": "OPHTH_2026", "lbl": "2026 عيون حديثة"},
                    {"grp": "OPHTH_HIST", "lbl": "⚠️ عيون قديمة (2014–2021)"},
                    {"grp": "VASC_HIST", "lbl": "⚠️ دوبلر ساق 2011 قديم"},
                    {"grp": "HIST_2024", "lbl": "⚠️ مقطعية وتخطيط 2024 وقومسيون 2021"},
                ],
                "roiBtn": "فحص عالي الدقة",
                "modalBtns": {"z1": "100%", "z2": "160%", "z3": "240%", "open": "فتح الأصل", "close": "إغلاق"},
                "filesHead": f"4. الفهرس الكامل لملفات المشروع الموثقة ({file_count} ملفاً)",
                "searchPh": "ابحث بالكلمة أو التاريخ أو العين أو Topcon أو VEP أو ERG أو دوبلر أو هولتر...",
                "fileFilters": [
                    {"grp": "ALL", "lbl": "كل الملفات"},
                    {"grp": "OPHTH_2026", "lbl": "عيون 2026"},
                    {"grp": "OPHTH_HIST", "lbl": "⚠️ عيون قديمة (14–21)"},
                    {"grp": "VASC_LL", "lbl": "⚠️ دوبلر 2011"},
                    {"grp": "RAD_MRI_2026", "lbl": "رنين 2026"},
                    {"grp": "AUD_2026", "lbl": "سمعيات 2026"},
                    {"grp": "CARD", "lbl": "هولتر 24h"},
                    {"grp": "VASC", "lbl": "دوبلر سباتي"},
                    {"grp": "RAD_CT_2024", "lbl": "⚠️ مقطعية 2024"},
                    {"grp": "NEURO_EEG_2024", "lbl": "⚠️ تخطيط مخ 2024"},
                    {"grp": "CLIN", "lbl": "روشتات وقومسيون"},
                ],
                "fileCols": ["#", "الفئة", "التاريخ والحالة", "الاسم الأصلي", "الملف المنظم الجديد (اضغط للفتح)", "الحجم", "الملخص الطبي السريع"],
                "fileFoot": f"جميع الملفات الأولية المنظمة ({file_count} ملفاً من 2011 إلى 2026) محفوظة في 01_Structured_and_Renamed_Inputs/ مع نسخ الأصل في 00_Raw_Backup_Archive/ وتمييز فحوصات ما قبل 2026 بعلامة ⚠️.",
            },
            "t4": {
                "head": "مقارنة الآراء الطبية السابقة وحسم التضارب التشخيصي",
                "sub": "تدقيق مباشر لأسباب اختلاف الأطباء السابقين حول قسطرة المخ واختبار الاتزان وسبب الرعشة وقياس الطبلية وتوقعات البصر — والثغرات الأربع التي كشفتها اللجنة.",
                "priorCards": [
                    {
                        "role": "رأي جراحة المخ والأعصاب السابق",
                        "doctor": "أ.د. أيمن السيد علي جلهوم (02 و03-11-2026)",
                        "badge": "b-amber",
                        "tag": "مؤيد للقسطرة",
                        "stance": "اشتباه ناسور شرياني وريدي قذالي -> أوصى بـ DSA والفحوصات، ووثق Essential Tremor",
                        "points": [
                            "<strong>فرضية الناسور القذالي:</strong> اختفاء الطنين بالضغط خلف الرقبة أثار شكه في ناسور بطيء Cognard I قد يفوته الرنين التقليدي 1.5T.",
                            "<strong>الفحوصات المطلوبة:</strong> طلب رنين المخ والأوردة والشرايين ودوبلر السباتي وهولتر 24 ساعة (ما مكننا من حل اللغز الآن).",
                            "<strong>تقييم الرعشة والوصفة:</strong> سجل بوضوح <code>Essential tremor + Titubation</code> دون إصابة بالساقين، ووصف <code>Inderal 10</code> و<code>Ciprapra</code> و<code>Cobal F</code> و<code>Brain-Ox</code>.",
                        ],
                    },
                    {
                        "role": "رأي القسطرة المخية السابق",
                        "doctor": "تقييم الأشعة التداخلية الخارجي",
                        "badge": "b-blue",
                        "tag": "ضد القسطرة",
                        "stance": "نصح بعدم إجراء قسطرة مخية تشخيصية فورية (Anti-DSA)",
                        "points": [
                            "<strong>المبرر الإشعاعي:</strong> استند إلى سلامة شرايين MRA وغياب التمدد الوريدي السطحي وشيوع ضمور الجيب الوريدي الشمال خلقياً.",
                            "<strong>الفجوة المتبقية:</strong> لم يقدم تفسيراً عملياً لاستمرار الطنين النبضي المزعج أو خريطة فرز أمان متدرجة.",
                        ],
                    },
                    {
                        "role": "استشارة الأنف والأذن السابقة",
                        "doctor": "د. محمد سالم عبد المقصود (روشتة 03-11-2026)",
                        "badge": "b-purple",
                        "tag": "أنف وأذن / VNG",
                        "stance": "اشتبه في سبب وعائي خلف الأذن؛ وصف ميكليزين وجينكو وطلب VNG",
                        "points": [
                            "<strong>دقة التموضع المبدئي:</strong> توقع بحق وجود سبب وعائي خلف الطبلية اليسرى يسبب الطنين وعدم الاتزان.",
                            "<strong>ثغرات وصفية خطيرة:</strong> كتب <code>Meclizine</code> شهراً كاملاً + <code>Ginkgo</code> شهرين وطلب <code>VNG</code> قياسي دون تنبيه لفقدان البصر (&lt; 1/60) أو الأسبرين.",
                        ],
                    },
                    {
                        "role": "سجلات العيون المنفصلة وتقرير Rology",
                        "doctor": "مراكز عيون متعددة + تقرير رنين Rology",
                        "badge": "b-red",
                        "tag": "فخ جودة Topcon 2",
                        "stance": "سقوط تاريخ زيت السيليكون في تقرير الرنين، وخداع قراءة 90 µm بجهاز التوبكون",
                        "points": [
                            "<strong>ثغرة تقرير الرنين:</strong> تقرير <code>Rology</code> بتاريخ 05-11-2026 لم يذكر زيت السيليكون البارز بالشمال (2.59x) ولا ضمور العصب البصري.",
                            "<strong>وهم جودة 2 بالتوبكون:</strong> سجل التوبكون للشمال رقم <code>90 µm</code> كاذباً عند جودة <code>Qual 2</code> بينما كانت العصب بالفعل <code>64 µm</code> منذ 2019 وانهارت <code>OPs ERG</code> لـ <code>0.74 µV</code> — لكن <code>Flash VEP</code> بقي حياً (<code>7.99 µV</code>)!",
                        ],
                    },
                ],
                "bridgeTitle": "أربع ثغرات تشخيصية خفية لم تكشفها إلا لجنة الاستشاريين السبعة",
                "bridgeCards": [
                    {
                        "title": "1. فخ الضغط المزدوج خلف الرقبة",
                        "cls": "alert-amber",
                        "txt": "الضغط القوي خلف الأذن والرقبة يقفل <strong>الشريان القذالي والوريد الوداجي الداخلي معاً</strong>؛ لذلك يجب اختبار الضغط الأمامي الخفيف على الوريد الوداجي مقابل الضغط النقطي على الشريان القذالي.",
                    },
                    {
                        "title": "2. ألغام القسطرة (بلاك 5 مم + قصور تاجي + كلى)",
                        "cls": "alert-red",
                        "txt": "الجدال السابق سبق اكتشاف <strong>بلاك 5 مم بالسباتي الشمال</strong> في مسار القسطرة، وهبوط <strong>ST -2.1 مم</strong> بالهولتر، وعدم فحص الكلى قبل الميتفورمين.",
                    },
                    {
                        "title": "3. فخ كف البصر مع الـ VNG والتداخلات الدوائية",
                        "cls": "alert-blue",
                        "txt": "غياب تنبيه البصر (&lt; 1/60) يبطل فحص تتبع الضوء في الـ VNG؛ وجمع <code>Meclizine + Brain-Ox + Ginkgo + Aggrex + Ciprapra + Inderal</code> يرفع مخاطر السقوط والنزيف وهبوط السكر.",
                    },
                    {
                        "title": "4. خدعة جودة Topcon 2 مقابل إثبات بقاء إشارة Flash VEP",
                        "cls": "alert-amber",
                        "txt": "لا تقرأ <code>90 µm</code> على جودة 2 في العين الشمال كأنها عصب سليم مع انهيار شبكية <code>OPs 0.74 µV</code>، وفي المقابل لا تيأس كلياً لأن <code>Flash VEP 7.99 / 10.29 µV</code> يثبت توصيل العصب ويوجب حماية الضغط بـ <strong>B-Scan 2026</strong>!",
                    },
                ],
            },
            "t5": {
                "head": "لجنة الاستشاريين السبعة: الاستنتاج، الإجراء الفوري، وخطة العمل",
                "sub": "كل استشاري مقسّم بوضوح إلى 3 كتل سريرية (1. الاستنتاج، 2. الإجراء الفوري، 3. الخطة التفصيلية) مسبوقة بالإجابات القاطعة على الأسئلة الثلاثة المحورية.",
                "allBtn": "إظهار السبعة استشاريين",
                "blkConcLbl": "1. الاستنتاج (Conclusion)",
                "blkImmLbl": "2. الإجراء الفوري (Immediate Action)",
                "blkPlanLbl": "3. الخطة التفصيلية (Detailed Plan)",
                "tieLbl": "الربط بين التخصصات:",
                "qaHead": "إجابات اللجنة القاطعة على الأسئلة الثلاثة المحورية للمريضة والأسرة",
                "qaCards": [
                    {
                        "cls": "qa-blue",
                        "badge": "b-blue",
                        "tag": "حل ضعف السمع",
                        "title": "أ. الحل النهائي لضعف السمع الشمال والطنين النبضي",
                        "bullets": [
                            "<strong>التشخيص القاطع:</strong> سمعيات <code>03-11-2026</code> المقنعة تثبت <strong>ضعف سمع قوقعي نقي (0 dB فجوة)</strong> (يمين <code>40 dB / تمييز 84%</code>، شمال <code>54.2 dB / تمييز 76%</code>) وطبلية يمين سليمة <code>1.5 سم³ Type A</code> مقابل شمال <code>3.2 سم³ Type Ad</code> تضخم الصوت الوريدي (<code>0.48</code>).",
                            "<strong>الوصفة السمعية الدقيقة:</strong> تركيب <strong>سماعتين رقميتين Open-Fit WDRC RIC</strong> بـ<strong>قبة مفتوحة مهواة (Vented/Open Domes)</strong> لمنع كتم الأذن وتضخيم الطنين، مع سقف خرج صارم <strong>Left OSPL90 &lt;= 95–100 dB SPL</strong> لحماية الأذن من حساسية <code>Metz (+45 dB SL)</code>.",
                            "<strong>العلاج المصاحب:</strong> إيقاف <code>Meclizine</code> و<code>Ginkgo</code> اليوم، علاج الجيوب ببخاخ <code>Mometasone</code> + غسول ملح؛ السماعتان ضرورة أمان حتمية لتحديد الاتجاهات في كف البصر <code>&lt; 1/60</code>.",
                        ],
                    },
                    {
                        "cls": "qa-red",
                        "badge": "b-red",
                        "tag": "أمل العين الشمال؟",
                        "title": "ب. الحقيقة العلمية الصادقة: هل يوجد أمل لإبصار العين الشمال؟",
                        "bullets": [
                            "<strong>الواقع السريري الصريح:</strong> عبر 12 عاماً (<code>2014 -> 2016 -> 2019 ضمور 64 µm -> سونار السيليكون 2021</code>)، استعادة إبصار وظيفي للقراءة بالعين الشمال <strong>مستبعد جداً</strong>؛ رقم <code>90 µm</code> بجهاز التوبكون على جودة <code>2</code> مجرد تشويش، وانهيار <code>OPs 0.74 µV</code> يثبت تلفاً إقفارياً شديداً بالشبكية الداخلية، وإزالة السيليكون بعد ~7 سنوات قد تؤدي لضمور وهبوط ضغط العين.",
                            "<strong>أين يكمن الأمل الحقيقي القابل للحماية؟</strong> فحص <code>CSO Flash VEP</code> في يوليو 2026 أثبت أن <strong>مسار العصب البصري للقشرة المخية ما زال ينقل الإشارة بالعينين (الشمال 7.99 µV / اليمين 10.29 µV)</strong> مع سلامة التصالب البصري بالرنين!",
                            "<strong>خطة العيون العاجلة:</strong> طلب <strong>سونار B-Scan جديد 2026 للعينين + قياس ضغط العين + فحص الزاوية</strong> لاستبعاد مياه زرقاء متسحلبة وعلاجها <strong>بقطرات موضعية فقط عند ارتفاع الضغط (ممنوع Cidamex فموياً!)</strong>، مع حماية تروية العصب (ضغط <code>&gt;= 115/70</code>) وتأهيل ضعف البصر للعين اليمنى الصافية (<code>Qual 10، 40 µm</code>).",
                        ],
                    },
                    {
                        "cls": "qa-amber",
                        "badge": "b-amber",
                        "tag": "الأعصاب والقسطرة",
                        "title": "ج. حسم الرعشة وهزة الرأس وتأجيل القسطرة المخية (DSA)",
                        "bullets": [
                            "<strong>نفي التصلب المتعدد (MS) والصرع:</strong> صفر علامات MS بالرنين (الموجود تصلب شرايين دقيقة CSVD 1–2 وجلطة فجوية قديمة)؛ تخطيط المخ 2024 يثبت إيقاع ألفا 9 هيرتز وارتعاشاً عضلياً فقط دون صرع.",
                            "<strong>تشريح الرعشة:</strong> خليط من <code>Essential Tremor + Titubation</code> (د. جلهوم) + قابلية وعائية + <strong>هبوط سكر صباحي مشتبه مع 80 وحدة أنسولين</strong> يفسر فورة <code>36.29%</code> للـ PVCs الساعة <code>6–9 صباحاً</code>! المطلوب <code>MDS-UPDRS</code> مع <code>Sinemet</code> وتركيب <code>CGM 14 يوماً</code>.",
                            "<strong>تأجيل قسطرة DSA:</strong> عبور <strong>بلاك السباتي 5 مم</strong> مع قصور تاجي <code>-2.1 مم</code> وقصور فحص الكلى غير آمن؛ الالتزام ببروتوكول الفرز الثلاثي (ضغط خفيف أمامي للوريد مقابل القذالي -> <code>4D-MRA / HRCT+CTA</code> بعد فحص الكلى -> قسطرة فقط إذا ثبت ناسور مؤكد).",
                        ],
                    },
                ],
                "rankHead": "التشخيص النهائي المرتب متعددة التخصصات (من 01 إلى 09)",
                "rankedDx": [
                    {"rank": "01", "cls": "b-blue", "cat": "الأوعية المخية", "title": "تصلب الشرايين المخية الدقيقة CSVD (Fazekas 1–2) وجلطة فجوية قديمة بالعقد القاعدية اليمين", "evidence": "مثبت بأشعة التوبكون والرنين (Film07/09/10) مع خلو DWI تماماً من أي جلطة حادة؛ وتم نفي MS والصرع تماماً."},
                    {"rank": "02", "cls": "b-red", "cat": "القلب والشرايين", "title": "تصلب شرايين سباتية عالي الخطورة (IMT 0.13 سم، بلاك 5 مم) + قصور تاجي صامت (-2.1 مم ST)", "evidence": "دوبلر السباتي (07-11) + هولتر القناة 3 (160 دقيقة هبوط، ذروة -2.1 مم الساعة 1:11 ظ) يؤسس المحور الوعائي المشترك."},
                    {"rank": "03", "cls": "b-red", "cat": "العيون والأعصاب", "title": "مرض عيون ثنائي ما بعد الجراحة (&lt; 1/60): ضمور عصب يمين 40 µm وتلف شبكية سيليكون شمال (OPs 0.74 µV) مع بقاء VEP", "evidence": "توثيق 12 عاماً (2014–2021) + فحوصات 2026 تثبت انهيار الشبكية الداخلية بالشمال لكن استجابة VEP القشرية حية بالعينين."},
                    {"rank": "04", "cls": "b-red", "cat": "كهرباء القلب والأيض", "title": "فورة صباحية في الخفقات البطينية (857 PVC و4 نوبات V-runs) و153 SVE على 80 وحدة أنسولين", "evidence": "تركز 36.29% من الخفقات بين 6 و9 صباحاً (ذروة 164/ساعة ونوبة 104 ن/د) يرجح هبوط سكر صامتاً أو نقص بوتاسيوم الثيازيد."},
                    {"rank": "05", "cls": "b-amber", "cat": "الأوردة والأذن", "title": "طنين وريدي نبضي شمال (ضيق الجيب الوريدي L/R 0.48) مضخّم بطبلية مرنة جداً 3.2 سم³ Type Ad", "evidence": "انعدام التجلط الوريدي؛ اندفاع موجات ما بعد الـ PVCs عبر الجيب الضيق أمام طبلية فائقة المرونة، مع حجز فرضية dAVF بطيء للفرز المتدرج."},
                    {"rank": "06", "cls": "b-red", "cat": "الاتزان والسقوط", "title": "فشل الاتزان رباعي الضربات (4-Hit Equilibrium Failure) وخطر السقوط المرتفع", "evidence": "ضعف أذن داخلية شمال + كف بصر &lt; 1/60 + تصلب مادة بيضاء وتثبيط Meclizine + تذبذب التروية القلبية والسكرية."},
                    {"rank": "07", "cls": "b-blue", "cat": "السمعيات", "title": "ضعف سمع حسي عصبي قوقعي ثنائي غير متماثل (0 dB ABG) — تصحيح خطأ 'Rt As'", "evidence": "سمعيات 03-11 المقنعة (يمين 84% وشمال 76% وMetz +45 dB)؛ شريط GSI-39 أثبت يمين 1.5 سم³ Type A طبيعية."},
                    {"rank": "08", "cls": "b-purple", "cat": "اضطراب الحركة", "title": "متلازمة رعشة مختلطة (Essential Tremor + Titubation + جلطة عقد قاعدية + هبوط سكر أدرينالي)", "evidence": "تجمع بين فحص د. جلهوم وسلبية تخطيط 2024 للصرع، وتنتظر اختبار MDS-UPDRS للتحقق من فائدة السينيمت."},
                    {"rank": "09", "cls": "b-green", "cat": "الجيوب والساقان", "title": "التهاب جيوب غربالية ثنائي + ارتجاع وريدي قديم بالساق الشمال (2011 — بلا PAD أو DVT)", "evidence": "رنين 2026 يؤكد التهاب الجيوب؛ دوبلر ساق 2011 يؤكد خلو الشرايين والأوردة العميقة تاريخياً."},
                ],
                "dilHead": "الحل العلمي التفصيلي للمعضلات الإكلينيكية الأربع",
            },
            "t6": {
                "head": "الخطة العلاجية الرئيسية، تدقيق الأدوية، وبروتوكولات الأذن والعين والملفات PDF",
                "resetBtn": "إعادة ضبط القائمة",
                "checkHead": "1. قائمة التحقق التفاعلية للفحوصات والإجراءات (محفوظة تلقائياً)",
                "doneWord": "مكتمل",
                "haHead": "2. بروتوكولات الأمان الإلزامية للسماعة، اختبار VNG، والعصب البصري",
                "haPoints": [
                    "<strong>تركيب سماعتين Open-Fit WDRC RIC:</strong> إلزامي للناحيتين (لحفظ الاتجاه المكاني مع كف البصر &lt; 1/60) مع <strong>قمع مفتوح مهوّى</strong> (لا قوالب مسدودة) + <strong>سقف خرج أقصى &lt;= 95–100 dB SPL</strong> للأذن اليسرى (Metz +45 dB SL).",
                    "<strong>تنبيه كتابي لفحص VNG:</strong> إبلاغ معمل الاتزان رسمياً بفقدان البصر (&lt; 1/60) وإلغاء اختبارات تتبع الضوء تماماً والاعتماد فقط على <strong>الوضعيات والحراري في الضلمة</strong>.",
                    "<strong>بروتوكول العيون وتروية العصب البصري:</strong> طلب <strong>سونار 2026 حديث + قياس الضغط IOP</strong>؛ استعمال قطرات فقط عند ارتفاع الضغط (ممنوع <code>Cidamex</code> فموياً مع الثيازيد!)، حفظ ضغط الدم &gt;= <code>115/70</code> ومنع هبوط السكر الليلي، ومراجعة ضغط العين بعد 4–6 أسابيع من بخاخ الجيوب.",
                ],
                "auditHead": "3. تدقيق تفاعلات الأدوية وحماية المريضة (8 محاور حرجة)",
                "auditFilters": [
                    {"tag": "ALL", "lbl": "الكل (8)"},
                    {"tag": "STOP", "lbl": "إيقاف / منع"},
                    {"tag": "CAUTION", "lbl": "حذر / استبدال"},
                    {"tag": "UPGRADE", "lbl": "رفع / إضافة"},
                ],
                "auditCols": ["الدواء / التداخل", "القرار", "الآلية الدوائية", "الخطر على المريضة", "تعديل اللجنة"],
                "rxHead": "4. جدول الوصفات المعدّلة والاستراتيجية المتكاملة (جميع التخصصات الستة)",
                "rxCols": ["التخصص الطبي", "العلاج / التدخل المقترح", "التوقيت والمتابعة", "السبب العلمي وحواجز الأمان"],
                "pdfSectionHead": "5. تحميل تقارير اللجنة المحدّثة بصيغة PDF والملف المرجعي",
                "pdfSectionSub": "إصدارات اللجنة السباعية المحدثة والمتضمنة الـ 69 ملفاً (2011–2026) ولوحات العيون والدوبلر وخطط العمل الثلاثية.",
            },
        },
    }

    # Bilingual lists (Consultants [1..7], Dilemmas, ROI, MedAudit, RxTable, Checklist)
    consultants = {
        "en": [
            {
                "id": 1,
                "role": "Consultant 1 — Diagnostic Head, Neck & Neuroradiology",
                "badge": "b-blue",
                "conclusion": "Right-dominant dural venous drainage with attenuated Left Transverse-Sigmoid-IJV conduit (L/R = 0.48, Film06) and ZERO thrombosis; Circle of Willis, IACs, CPAs, and optic chiasm are normal (excluding vestibular schwannoma or compressive pituitary mass). Brain parenchyma demonstrates hypertensive-diabetic CSVD (Fazekas 1–2) + chronic Rt basal ganglia lacune (zero DWI stroke; stable vs 2024 CT) + bilateral ethmoiditis. Forensically corrected external Rology omission of prominent 2.59x Left Intraocular Silicone Oil & bilateral optic thinning, corroborating 2019–2021 B-Scans & 2026 Topcon Qual 10 OD 40um.",
                "immediate_action": "Do NOT order immediate unscreened Catheter DSA or contrast CT/MRV today; first obtain Serum Creatinine/eGFR, verify 48h Metformin hold instructions, obtain Cardiology clearance (-2.1 mm ST, 857 PVCs), perform bedside light IJV vs focal occipital compression, and file an official addendum noting Left Intraocular Silicone Oil & bilateral optic atrophy.",
                "detailed_plan": [
                    "<strong>Gatekeeper Non-Invasive Protocol (If Arterial Bruit Suspected or Tinnitus Refractory at 4–6 wks):</strong> Order 0.5-mm Temporal Bone HRCT + Contrast CTA/CTV OR 4D Time-Resolved Contrast MRA (4D-TRAK) strictly if eGFR >= 45 and Metformin held 48h — achieves 100% bone dehiscence & ~90–95% Cognard I dAVF detection without touching the 5 mm Lt Carotid Bulb plaque.",
                    "<strong>Correlative Ocular Guidance:</strong> Defer repeat orbital MRI/CT because Film08 & 2024 CT confirm persistent silicone oil; pair instead with fresh 2026 B-Scan ocular ultrasound.",
                    "<strong>Sinusitis & Carotid Tracking:</strong> Treat bilateral ethmoiditis medically (Mometasone + saline) and monitor the 5 mm Lt bulb plaque via non-invasive annual Carotid Duplex under High-Intensity Statin.",
                ],
                "tie": "Links the 0.48 venous bottleneck, -2.1 mm Holter pulse surges, and 3.2 cm3 TM acoustics while unifying MRI/CT/Ophthalmology findings.",
            },
            {
                "id": 2,
                "role": "Consultant 2 — Multiple Sclerosis (MS) & White Matter Disorders",
                "badge": "b-green",
                "conclusion": "Multiple Sclerosis, NMOSD, and CNS vasculitis are definitively EXCLUDED (2017 McDonald & STRIVE-2 negative: zero Dawson fingers, intact corpus callosum & infratentorium, negative DWI). White-matter foci strictly reflect Hypertensive-Diabetic CSVD (Fazekas 1–2) + chronic lacunar infarcts. Crucially, July 2026 CSO Full-Field ERG (OPs OD 10.0 vs OS collapsed 0.74 uV; dark/flicker suppression) plus 2014–2021 retinopathy proves optic atrophy (OD 40 um / OS 64 um 2019) is vitreoretinal-glaucomatous/ischemic, NOT demyelinating optic neuritis.",
                "immediate_action": "Reassure the patient and family that she does NOT have Multiple Sclerosis; cancel/avoid Lumbar Puncture (CSF OCB), AQP4/MOG antibodies, IV pulse steroids, or MS immunosuppressants (unnecessary and hazardous atop 80 IU/d insulin). Focus all urgency on microvascular endothelial control.",
                "detailed_plan": [
                    "<strong>Microvascular CSVD Halt:</strong> Intensive HbA1c control free of morning hypoglycemic swings, smooth 24h BP control (125–135/75–80 mmHg avoiding nocturnal <115 systolic hypoperfusion), High-Intensity Statin (LDL <55 mg/dL), and single antiplatelet Aggrex 75–81 mg OD.",
                    "<strong>Cardio-Cerebral Reperfusion Shield:</strong> Stabilize -2.1 mm silent coronary ischemia & 857 PVCs to prevent repetitive white-matter watershed hypoperfusion.",
                    "<strong>Rational Surveillance:</strong> Repeat non-contrast Brain MRI only at 12–18 months or with new acute focal deficits; routine MS gadolinium protocols are not indicated.",
                ],
                "tie": "Prevents invasive lumbar puncture & steroid harm while uniting CSVD, cochlear SNHL, Holter ischemia, and ischemic retinopathy under one microvascular axis.",
            },
            {
                "id": 3,
                "role": "Consultant 3 — Medical Neurology, Vascular & Movement Disorders",
                "badge": "b-amber",
                "conclusion": "Movement disorder reflects a triple/quadruple overlap reconciling prior Sinemet vs Prof. Galhom's 2026-11-03 log of Essential Tremor + Head Titubation (negative lower limbs): (1) ET-Plus substantiated by 2024 2h EEG (9 epochs showing EMG tremor bursts on normal 9 Hz alpha, zero epilepsy), (2) chronic Rt basal ganglia lacune + frontal CSVD vascular tremor vulnerability, and (3) insulin adrenergic tremors (80 IU/d matching the 06:00–08:59 AM 36.29% Holter PVC surge & 104 bpm V-run). 2011 Leg Doppler confirms no PAD/DVT (perforator incompetence only), contrasting high-risk systemic atherosclerosis now (5 mm bulb plaque, IMT 0.13 cm, CSVD, -2.1 mm ST).",
                "immediate_action": "Order STAT Labs today (K+, Na+, Mg2+, Ca2+, Cr/eGFR, HbA1c, Fasting Lipids, CBC, TSH, Vit B12), launch a 14-day CGM / 7-pt SMBG targeting 03:00–09:00 AM hypoglycemia, STOP Cetrovit Ginkgo & daily Meclizine immediately, and keep Sinemet unchanged ONLY until formal in-person response testing.",
                "detailed_plan": [
                    "<strong>Movement & Tremor Clarification:</strong> Execute off vs 60-min on Sinemet MDS-UPDRS Part III; taper Sinemet slowly over 2–4 wks if response <20–30%. With Cardiology, prefer transition from unmonitored Inderal (which masks hypoglycemia warnings on 80 IU/d insulin) to cardioselective Bisoprolol 2.5–5 mg OD.",
                    "<strong>High-Risk Vascular Regimen:</strong> Upgrade Zatorva 20 -> Atorvastatin 40–80 mg nocte (LDL <55); keep Aggrex 75–81 mg (+ PPI if dyspeptic); deprescribe Brain-Ox & verify Ciprapra indication; transition Angiopress Comp -> 24h Perindopril/Telmisartan (+/- Amlodipine).",
                    "<strong>Metabolic & Peripheral Venous Care:</strong> Endocrinology insulin de-escalation (+/- SGLT2i post-eGFR), Mometasone sinus spray, and daytime 15–20 mmHg stockings + ankle pumps once diabetic pedal pulses are verified intact.",
                ],
                "tie": "Connects morning insulin dips to both adrenergic tremor and the 08:00 AM Holter PVC peak while fixing triple-bleeding & hypokalemic drug hazards.",
            },
            {
                "id": 4,
                "role": "Consultant 4 — Senior Consultant in Neurosurgery",
                "badge": "b-green",
                "conclusion": "Zero surgical intracranial tumor, aneurysm, AVM, hydrocephalus, or Chiari compression across MRI/CT/MRA. Coronal MRI (Film10) confirms a completely intact optic chiasm & pituitary stalk, ruling out compressive pituitary/suprasellar tumor as the cause of severe bilateral vision loss. No high-risk cortical venous reflux (Cognard IIb–IV) exists on 513 TOF slices, while -2.1 mm silent ischemia, 4 V-runs, 5 mm Lt Carotid Bulb plaque, and unverified Cr/Metformin make immediate blind DSA neurosurgically unjustifiable.",
                "immediate_action": "HOLD any neurosurgical craniotomy, mastoid bone resurfacing, or upfront catheter DSA scheduling; enforce bedside selective IJV vs isolated occipital pulse compression + Cr/eGFR + Cardiology review before any invasive step.",
                "detailed_plan": [
                    "<strong>Staged Gatekeeper Pathway:</strong> If bedside light anterior IJV compression extinguishes tinnitus & no arterial bruit exists, pursue 4–6 wks medical/sinus/Holter therapy (zero surgery); otherwise order 4D-MRA or 0.5-mm Temporal Bone HRCT + CTA/CTV (post-eGFR >= 45 & 48h Metformin hold).",
                    "<strong>Narrow Surgical / Endovascular Threshold:</strong> Reserve Onyx/NBCA embolization strictly for confirmed symptomatic dAVF/CVR on gatekeeper imaging or bone reconstruction for sleep-destroying dehiscence.",
                    "<strong>Non-Negotiable Right Venous Protection:</strong> Because MRV (0.48 ratio) proves the Right Transverse-Sigmoid-IJV carries nearly all brain venous drainage, NEVER sacrifice or compress the Right dural venous system.",
                ],
                "tie": "Rules out sellar/chiasmal vision compression on MRI Film10 and enforces anatomical preservation of the dominant Right venous sinus.",
            },
            {
                "id": 5,
                "role": "Consultant 5 — Neuro-Interventional Radiology (Catheter DSA Arbitration)",
                "badge": "b-red",
                "conclusion": "Resolves the Pro-DSA vs Anti-DSA deadlock cleanly: while 1.5T TOF MRA excludes large/high-flow Cognard IIb–IV dAVF, it cannot definitively dismiss micro-shunt Cognard Type I ECA twigs. However, firm posterior suboccipital compression indiscriminately collapses BOTH the Occipital Artery AND the mastoid emissary/upper IJV. Combined with an attenuated 0.48 left sinus directly beside a flaccid 3.2 cm3 Type Ad left TM amplifying 857 PVCs, crossing a 5 mm Left Carotid Bulb plaque during unscreened DSA amid -2.1 mm silent ischemia & unchecked eGFR/Metformin is unsafe today.",
                "immediate_action": "Defer elective upfront Catheter DSA; perform quiet mastoid auscultation comparing light unilateral anterior-neck Left IJV pressure (never squeezing the 5 mm carotid bulb plaque!) vs pinpoint Left Occipital Artery pressure 2–3 cm postero-inferior to mastoid tip, and draw Cr/eGFR + K+/Mg2+ + Cardiology eval first.",
                "detailed_plan": [
                    "<strong>Step 1 (4–6 Wk Conservative Trial if Venous Pattern):</strong> Normalize nocturnal BP (stop HCTZ/Captopril OD), treat ethmoiditis, control AM PVC surges, and sleep at 30 deg; remission avoids catheter angiography permanently.",
                    "<strong>Step 2 (Non-Invasive Gatekeeper CTA/4D-MRA):</strong> Triggered ONLY if isolated focal occipital bruit persists or refractory symptoms continue after Step 1 (requiring eGFR >= 45 + 48h Metformin hold + hydration).",
                    "<strong>Step 3 (Strict Conditional DSA):</strong> Only for positive Step 2 dAVF/CVR or disabling objective bruit, under Cardiology ischemia clearance, anticoagulation, radial/plaque-sparing technique, and 48h Metformin hold.",
                ],
                "tie": "Uses Duplex (5 mm bulb plaque), Holter (-2.1 mm ST), and Tymp (3.2 cm3 amplifier) to settle the foremost clinical conflict safely.",
            },
            {
                "id": 6,
                "role": "Consultant 6 — ENT, Neurotology & Clinical Audiology",
                "badge": "b-blue",
                "conclusion": "Definitive Masked PTA (2026-11-03) supersedes 2024 baseline: Pure Bilateral Asymmetric Cochlear SNHL with ZERO Air-Bone Gap (Rt 3f PTA 40.0 dB / WRS 84%; Lt stable 54.2 dB / WRS 76%; historical Metz +45 dB SL). GSI-39 thermal strip exposes a critical clerical error: RIGHT Peak is 1.5 cm3 @ -5 daPa (Normal Type A — refuting handwritten 'Rt As'), while LEFT Peak hits 3.2 cm3 @ 0 daPa (>3.0 cm3 plot ceiling, Extreme Type Ad monomeric drum acting as a low-frequency biological microphone for the 0.48 venous sinus & 857 PVCs). Confirms '4-Hit' Balance Failure worsened by 2026-documented <1/60 blindness and daily Meclizine.",
                "immediate_action": "STOP daily Meclizine & Cetrovit Ginkgo today; issue a written VNG Lab Safety Mandate warning of <1/60 blindness (CANCEL light-target saccade/pursuit/OPK calibration — run DARK positional & bithermal calorics ONLY); start Mometasone 50 mcg nasal spray (2 puffs qAM) + saline irrigation.",
                "detailed_plan": [
                    "<strong>Exact Hearing Loss Prescription (Specific Answer A):</strong> Fit Bilateral Digital Open-Fit Receiver-In-Canal (RIC) Hearing Aids with soft open/vented domes (mandatory binaural spatial cues for <1/60 blindness without trapping internal pulsatile somatosounds) + WDRC compression + hard Left OSPL90 ceiling <= 95–100 dB SPL (Metz protection) + optional bedside sound masking + 30 deg head elevation.",
                    "<strong>Non-Visual Vestibular Rehabilitation (VRT):</strong> Daily supervised somatosensory/proprioceptive balance physiotherapy, home fall-proofing, podiatry check, and white-cane mobility cues.",
                    "<strong>Otologic Follow-Up:</strong> Otomicroscopy of the monomeric 3.2 cm3 left TM + Epley maneuver if Dix-Hallpike positive on dark VNG.",
                ],
                "tie": "Refutes the handwritten 'Rt As' error, integrates July 2026 vision loss into VNG safety, and delivers exact Open-Fit WDRC aid parameters.",
            },
            {
                "id": 7,
                "role": "Consultant 7 — Ophthalmology (Vitreoretinal & Neuro-Ophthalmology)",
                "badge": "b-red",
                "conclusion": "12-year longitudinal reconciliation (2014–2026): Bilateral severe optic/retinal compromise predated vitrectomy (2014 OS RNFL 78.36 -> 2016 71.41 -> 2019 64 um; OD CME/RPE disruption), progressing to bilateral RD surgeries + silicone oil (2019–2021). Current 2026 Topcon (OD clear Qual 10, attached fovea 259 um, profound OD RNFL 40 um [Inf 34 um]; OS Qual 2 prints false 90 um scatter artifact!), Zeiss HFA (VFI 18% 'Not reliable Test'), and CSO ERG/VEP resolve Left Eye Hope (Specific Answer B): functional OS recovery is very unlikely (OS ERG OPs collapsed 0.74 vs 10.0 uV OD; ~7-yr oil removal risks hypotony/re-detachment), YET BILATERAL FLASH VEP REMAINS MEASURABLE (OD 10.29 uV, OS 7.99 uV) alongside normal MRI optic chiasm!",
                "immediate_action": "Order an urgent Fresh 2026 Bilateral Ocular B-Scan + Applanation IOP + Slit-Lamp/Gonioscopy (last B-Scan on file is May 2021!), do NOT schedule routine OS silicone oil removal without specialist surgical justification, and STRICTLY PROHIBIT systemic Acetazolamide (Cidamex) atop HCTZ/Angiopress Comp (use topical IOP drops ONLY if pressure is raised).",
                "detailed_plan": [
                    "<strong>Preserving Measurable Neural Reserve (OD Priority + OS Pressure Shield):</strong> Check & control secondary emulsified silicone oil glaucoma topically; protect fragile 40 um optic nerve head perfusion by keeping systemic BP >= 115/70 mmHg, upgrading statin & eliminating nocturnal 80 IU/d insulin hypoglycemia.",
                    "<strong>Cross-System Steroid & Cardiac Coupling:</strong> Recheck IOP 4–6 weeks after initiating ENT Mometasone nasal spray; alert neurology that chronic Propranolol can mildly drop IOP whereas stopping HCTZ safely avoids systemic glaucoma diuretic hypokalemia.",
                    "<strong>Low-Vision & Spatial Rehabilitation:</strong> Focus optical/magnification aids on better-media OD (Qual 10), integrate tactile/talking glucose meters & high-contrast home safety alongside bilateral RIC hearing aids for indoor orientation.",
                ],
                "tie": "Eliminates the Topcon Qual 2 '90 um' false reading, proves live VEP transmission (7.99 uV OS), and ties IOP/perfusion safety into ENT, Neurology & Cardiology.",
            },
        ],
        "ar": [
            {
                "id": 1,
                "role": "استشاري 1 — الأشعة التشخيصية للرأس والرقبة والأعصاب",
                "badge": "b-blue",
                "conclusion": "التصريف الوريدي للمخ معتمد على الجانب الأيمن مع ضمور خلقي بالجيب الوريدي المستعرض والسيني والوداجي الشمال (نسبة L/R = 0.48 في Film06) بلا أي جلطات وريدية؛ وشرايين دائرة ويليس وقنوات السمع والتصالب البصري سليمة تماماً. نسيج المخ يظهر تصلب شعيرات دقيقة (CSVD 1–2) وجلطة فجوية قديمة بالعقد القاعدية اليمين (DWI سلبي للجلطات الحادة؛ ومطابق لمقطعية 2024) والتهاب جيوب غربالية ثنائي. رصد بالرنين إغفال تقرير Rology لزيت السيليكون البارز بالعين الشمال (2.59x إضاءة) وضمور العصب البصري، متطابقاً مع سونار 2019–2021 وتوبكون 2026 (يمين جودة 10، 40 µm).",
                "immediate_action": "عدم إجراء قسطرة مخية (DSA) أو أشعة بالصبغة فوراً اليوم؛ يجب أولاً قياس الكرياتينين/eGFR وتجهيز إيقاف الميتفورمين 48 ساعة، أخذ موافقة القلب (-2.1 مم ST و857 PVC)، إجراء اختبار الضغط الأمامي اللطيف للوريد مقابل القذالي بالسماعة، وإرفاق ملحق رسمي يوثق زيت السيليكون بالشمال وضمور العصب البصري.",
                "detailed_plan": [
                    "<strong>بروتوكول البوابة غير التداخلي (عند الاشتباه الشرياني أو استمرار الطنين بعد 4–6 أسابيع):</strong> طلب أشعة مقطعية عالية الدقة لعظمة الأذن HRCT 0.5mm + CTA/CTV بالصبغة أو رنين ديناميكي 4D-MRA بشرط eGFR >= 45 ووقف الميتفورمين 48 ساعة؛ لتأكيد أو نفي التآكل العظمي والناسور البطيء بنسبة 90–95% دون لمس بلاك السباتي الـ 5 مم.",
                    "<strong>متابعة العين بدون إشعاع إضافي:</strong> لا حاجة لإعادة رنين الحجاج لأن Film08 ومقطعية 2024 يثبتان وجود السيليكون؛ يُستبدل بسونار B-Scan 2026 للعينين.",
                    "<strong>الجيوب ومتابعة البلاك:</strong> علاج التهاب الجيوب الغربالية بموميتازون ومحلول ملح، ومتابعة بلاك السباتي الـ 5 مم بدوبلر سنوي مع ستاتين عالي الكثافة.",
                ],
                "tie": "يربط ضيق الجيب الوريدي 0.48 باندفاعات نبض الهولتر والطبلية 3.2 سم³، ويوحد قراءات الرنين والمقطعية وأشعة العيون.",
            },
            {
                "id": 2,
                "role": "استشاري 2 — التصلب المتعدد (MS) وأمراض المادة البيضاء",
                "badge": "b-green",
                "conclusion": "نفي قاطع للتصلب المتعدد (MS) وNMOSD والتهاب الأوعية (سلبية معايير ماكدونالد 2017: صفر أصابع داوسون، سلامة الجسم الجاسي والمخيخ وجذع المخ، وسلبية DWI). البؤر البيضاء تمثل Strictly تصلب شرايين مخية دقيقة (CSVD Fazekas 1–2) وجلطات فجوية قديمة. والأهم أن تخطيط الشبكية الكهربائي CSO في يوليو 2026 (انهيار OPs الشمال إلى 0.74 مقابل 10.0 µV يمين) وتاريخ 2014–2021 يثبتان أن ضمور العصب البصري وعائي/شبكي/مرتفع الضغط وليس التهاباً مناعياً تبع MS!",
                "immediate_action": "طمأنة المريضة والأسرة بأنها غير مصابة بالـ MS؛ وإلغاء البزل القطني (CSF)، وتحاليل AQP4/MOG، أو أخذ كورتيزون وريدي أو أدوية مناعية (ضارة وغير مبررة مع 80 وحدة أنسولين). توجيه الجهود فوراً لضبط الشرايين الدقيقة.",
                "detailed_plan": [
                    "<strong>علاج جذري لتصلب المادة البيضاء CSVD:</strong> ضبط السكر التراكمي بدون نوبات هبوط صباحية، ضبط الضغط بسلاسة على مدار 24 ساعة (125–135/75–80 مم زئبق دون هبوط ليلي <115)، ستاتين عالي الجرعة (LDL <55)، وأسبرين 75–81 مجم يومياً منفرداً.",
                    "<strong>حماية التروية المخية-القلبية:</strong> علاج القصور التاجي الصامت (-2.1 مم) والـ 857 PVC لمنع تذبذب التروية المتكرر للمادة البيضاء.",
                    "<strong>متابعة الأشعة المنطقية:</strong> رنين مخ عادي بدون صبغة فقط بعد 12–18 شهراً أو عند حدوث عرض عصبي بؤري جديد.",
                ],
                "tie": "يحمي المريضة من بزل النخاع والكورتيزون ويربط CSVD وضعف القوقعة والقلب والشبكية في خط وعائي ميكروسكوبي واحد.",
            },
            {
                "id": 3,
                "role": "استشاري 3 — الباطنة العصبية والأوعية الدموية واضطرابات الحركة",
                "badge": "b-amber",
                "conclusion": "الاضطراب الحركي مزيج ثلاثي/رباعي يفسر تضارب السينيمت أمام فحص أ.د. جلهوم (Essential Tremor + Titubation وسلبية الساقين): (1) رعشة أساس مؤكدة بتخطيط مخ 2024 (9 مقاطع: ذبذبات EMG عضلية على خلفية ألفا 9 هيرتز بلا صرع)، (2) قابلية رعشة/تيبّس وعائي من جلطة العقد القاعدية اليمين وCSVD، و(3) رعشة هبوط سكر أدرينالية من 80 وحدة أنسولين متزامنة مع فورة الـ PVCs الصباحية (36.29% بين 6 و9 صباحاً مع نوبة 104 ن/د). كما يثبت دوبلر الساق 2011 خلوها من PAD/DVT تاريخياً مقابل تصلب شرايين جهازي عالي الخطورة الآن (بلاك 5 مم، IMT 0.13 سم، -2.1 مم ST).",
                "immediate_action": "طلب تحاليل اليوم فوراً (K+, Na+, Mg2+, Ca2+, Cr/eGFR, HbA1c، دهون، صورة دم، غدة TSH، B12)، تركيب سنسور سكر CGM 14 يوماً أو قياس 7 مرات مع التركيز على 3–9 صباحاً، وقف Cetrovit Ginkgo وMeclizine فوراً، وعدم قطع Sinemet فجأة قبل فحص الاستجابة.",
                "detailed_plan": [
                    "<strong>حسم نوع الرعشة والأدوية:</strong> إجراء فحص MDS-UPDRS مع وبدون سينيمت (سحبه تدريجياً خلال 2–4 أسابيع إن كان التحسن <20–30%). التنسيق مع القلب للتحويل من Inderal (الذي يخفي أعراض هبوط السكر!) إلى Bisoprolol 2.5–5 mg يومياً إذا سمحت الرعشة.",
                    "<strong>تطوير الوقاية الوعائية:</strong> رفع Zatorva 20 -> Atorvastatin 40–80 mg ليلاً (هدف LDL <55)؛ استمرار Aggrex 75–81 mg؛ مراجعة Brain-Ox وCiprapra؛ استبدال Angiopress Comp بدواء طويل المفعول (Perindopril/Telmisartan +/- Amlodipine).",
                    "<strong>الأيض والجيوب والأوردة:</strong> مراجعة جرعة الأنسولين (+/- SGLT2i بعد فحص الكلى) + بخاخ موميتازون؛ جوارب ضاغطة خفيفة 15–20 مم زئبق نهاراً فقط بعد فحص نبض القدم السكري.",
                ],
                "tie": "يفسر ذروة الخفقات والرجفة الصباحية بهبوط السكر الأدرينالي ويحسم تفاعلات النزيف ونقص البوتاسيوم.",
            },
            {
                "id": 4,
                "role": "استشاري 4 — جراحة المخ والأعصاب",
                "badge": "b-green",
                "conclusion": "انعدام تام لأي ورم بالمخ أو زاوية المخيخ والجسر أو تمدد شرياني أو استسقاء. والأهم أن المقطع الإكليلي Film10 أثبت سلامة التصالب البصري والغدة النخامية تماماً، مما ينفي وجود ورم ضاغط على العصب البصري كسبب لكف البصر. كما تنفي شرائح TOF أي ارتجاع وريدي قشري خطير (Cognard IIb–IV)، وفي وجود قصور تاجي -2.1 مم وبلاك سباتي 5 مم وغياب فحص الكلى تصبح القسطرة المباشرة أو أي جراحة غير مبررة الآن.",
                "immediate_action": "عدم تحديد موعد لأي جراحة جمجمة أو ترقيع عظمي أو قسطرة مخية DSA فورية؛ البدء باختبار الضغط الانتقائي بالسماعة وفحص الكلى ومراجعة القلب أولاً.",
                "detailed_plan": [
                    "<strong>خوارزمية الفرز المتدرجة:</strong> إذا ثبت الطنين الوريدي بالضغط اللطيف دون لغط شرياني وتحسن بالعلاج الدوائي 4–6 أسابيع -> لا جراحة ولا قسطرة نهائياً؛ وبخلاف ذلك يُجرى 4D-MRA أو مقطعية عالية الدقة بالصبغة بعد التأكد من الكلى ووقف الميتفورمين 48 ساعة.",
                    "<strong>مؤشرات التدخل الضيقة:</strong> قصر القسطرة العلاجية على ثبوت ناسور حقيقي وارتجاع وريدي خطير، أو الترميم الجراحي على التآكل العظمي المدمر للنوم.",
                    "<strong>ميثاق الأمان الوريدي المطلق:</strong> نظراً لضيق الجيب الشمال (0.48) واعتماد تصريف المخ على الجيب الوريدي الأيمن السائد، يُمنع منعاً باتاً إغلاق أو التضحية بأي وريد جيب أيمن مستقبلاً.",
                ],
                "tie": "ينفي ضغط الأورام على التصالب البصري ويفرض حماية كاملة لمسار تصريف الدم الوريدي الأيمن السائد.",
            },
            {
                "id": 5,
                "role": "استشاري 5 — القسطرة المخية والأشعة التداخلية (حسم جدل الـ DSA)",
                "badge": "b-red",
                "conclusion": "حسم التضارب بين طلب DSA ورفضها: رنين 1.5T TOF ينفي الناسور عالي التدفق Cognard IIb–IV لكنه لا يستبعد 100% ناصوراً دقيقاً بطيئاً من الشريان القذالي؛ ومع ذلك فإن الضغط القوي خلف الأذن يكتم الشريان القذالي والوريد الوداجي معاً! ومع تضخيم الطبلية 3.2 سم³ لاندفاعات 857 PVC عبر جيب وريدي ضيق 0.48، فإن تمرير قسطرة بجانب بلاك 5 مم في عنق السباتي الشمال مع -2.1 مم ST وعدم فحص الكلى/الميتفورمين يحمل خطراً غير مقبول حالياً.",
                "immediate_action": "تأجيل القسطرة المخية التشخيصية (DSA)؛ إجراء تسمع هادئ بالسماعة خلف الأذن ومقارنة ضغط أمامي رقيق بإصبع على الوريد الوداجي الشمال (دون عصر بلاك السباتي الـ 5 مم!) مقابل ضغط نقطي 2–3 سم خلف وتحت الماستويد على الشريان القذالي، مع تحليل الكلى والقلب أولاً.",
                "detailed_plan": [
                    "<strong>الخطوة 1 (تجربة علاجية 4–6 أسابيع للنمط الوريدي):</strong> تنظيم الضغط الليلي (وقف الثيازيد)، علاج الجيوب الغربالية، تهدئة الـ PVCs الصباحية، ورفع الرأس 30 درجة؛ الاستجابة تلغي القسطرة للأبد.",
                    "<strong>الخطوة 2 (بوابة التصوير غير التداخلي):</strong> فقط عند استمرار لغط شرياني نقطي قذالي أو فشل الخطوة 1 -> عمل 4D-MRA أو HRCT + CTA/CTV بشرط eGFR >= 45 ووقف الميتفورمين 48 ساعة.",
                    "<strong>الخطوة 3 (قسطرة DSA مشروطة):</strong> حصراً إذا أثبت الفحص الديناميكي ناصوراً أو تباطؤاً خطيراً، وبعد موافقة القلب والوصول بحذر بعيداً عن بلاك الـ 5 مم.",
                ],
                "tie": "يحسم الجدال الأهم بالدليل الكمي من دوبلر السباتي (5 مم) والهولتر (-2.1 مم) والطبلية (3.2 سم³).",
            },
            {
                "id": 6,
                "role": "استشاري 6 — أمراض الأذن والاتزان والسمعيات الإكلينيكية",
                "badge": "b-blue",
                "conclusion": "سمعيات 03-11-2026 المقنعة تحل محل 2024: ضعف سمع قوقعي نقي بالجهتين مع 0 dB فجوة (يمين 40.0 dB / تمييز 84%؛ شمال مستقر 54.2 dB / تمييز 76% مع حساسية Metz +45 dB SL). شريط الطبلية GSI-39 كشف خطأ خط اليد 'Rt As': اليمين 1.5 سم³ Type A طبيعية تماماً، بينما الشمال 3.2 سم³ Type Ad متطرف فوق سقف الورقة يعمل كمضخم ميكانيكي لصوت الجيب الضيق (0.48) واندفاعات 857 PVC. كما يؤكد فشل الاتزان الرباعي المفاقم بفقدان البصر &lt;1/60 ودواء Meclizine.",
                "immediate_action": "وقف Meclizine وCetrovit Ginkgo اليوم فوراً؛ إرسال تنبيه مكتوب لمعمل الـ VNG بشأن فقدان البصر (&lt; 1/60) لإلغاء تتبع الضوء والاكتفاء بالوضعيات والحراري في الضلمة فقط؛ وبدء بخاخ Mometasone 50 mcg + محلول ملح للجيوب.",
                "detailed_plan": [
                    "<strong>بروتوكول السماعات الدقيق (إجابة سؤال السمع):</strong> تركيب سماعتين رقميتين Open-Fit RIC بقبة مفتوحة مهواة (لحماية الإدراك المكاني في كف البصر دون سد القناة وحبس الطنين) مع ضغط WDRC وسقف خرج حاد OSPL90 &lt;= 95–100 dB SPL للأذن اليسرى احتراماً لحساسية Metz + برنامج مهدئ طنين + رفع الرأس 30 درجة.",
                    "<strong>تأهيل الاتزان غير البصري (VRT):</strong> جلسات علاج طبيعي حسية-عضلية بالاعتماد على الأرضية والكاحل وتأمين المنزل من السقوط والعناية بقدمي السكري.",
                    "<strong>متابعة الأذن:</strong> فحص الطبلية المترققة بالميكروسكوب وإجراء مناورة Epley إن ثبت BPPV في الـ VNG بالضلمة.",
                ],
                "tie": "يصحح خطأ الطبلية اليمين، يحمي فحص الـ VNG من خطأ الإبصار، ويصف أدق سماعة تناسب الطنين وMetz وكف البصر.",
            },
            {
                "id": 7,
                "role": "استشاري 7 — طب وجراحة العيون (الشبكية والجسم الزجاجي وأعصاب العين)",
                "badge": "b-red",
                "conclusion": "مصالحة سجل 12 سنة (2014–2026): بدأ ضمور العصب والتلف الشبكي قبل عمليات الانفصال (2014 شمل RNFL 78.36 -> 2016 71.41 -> 2019 64 µm وارتشاح ماكولة اليمين) وصولاً لجراحات السيليكون بالعينين 2019–2021. فحوصات 2026 الحالية (التوبكون: يمين صافية جودة 10، شبكية مثبتة 259 µm، وضمور عصب شديد 40 µm؛ بينما الشمال جودة 2 تطبع رقم 90 µm كاذب بسبب السيليكون!) ومجال HFA 18% ('غير موثوق') ورسم CSO تحسم سؤال الأمل بالعين الشمال: استعادة القراءة بالشمال مستبعد جداً (انهيار OPs لـ 0.74 مقابل 10.0 µV يمين، ومخاطر إزالة السيليكون بعد ~7 سنوات عالية)، لكن إشارة العصب لمركز الإبصار في المخ ما زالت حية (Flash VEP الشمال 7.99 µV واليمين 10.29 µV) مع تصالب بصري سليم!",
                "immediate_action": "طلب سونار B-Scan 2026 حديث للعينين + قياس ضغط العين بالمستوى المسطح + فحص الزاوية فوراً (آخر سونار بالملف مايو 2021!)، وعدم تحديد موعد روتيني لإزالة زيت السيليكون الشمال دون فحص تخصصي دقيق، وحظر Cidamex الفموي نهائياً مع الثيازيد (قطرات موضعية فقط إذا ارتفع الضغط).",
                "detailed_plan": [
                    "<strong>حماية الرصيد العصبي المتبقي (أولوية اليمين + أمان الشمال):</strong> الكشف المبكر وعلاج ارتفاع ضغط السيليكون المتسحلب بالقطرات الموضعية عند اللزوم؛ الحفاظ على ضغط الدم &gt;= 115/70 وضبط السكر دون هبوط ليلي لحماية العصب البصري الضامر 40 µm بالعين اليمنى.",
                    "<strong>ربط الأدوية الأنفية والعامة بالعين:</strong> إعادة فحص ضغط العين بعد 4–6 أسابيع من بخاخ Mometasone، وحماية كهرباء القلب بمنع الأسيتازولاميد الفموي مع الثيازيد.",
                    "<strong>تأهيل ضعف البصر والتوجيه المكاني:</strong> تركيز المعينات البصرية على العين اليمنى الصافية (Qual 10) مع أجهزة قياس سكر ناطقة وتنسيق المكفوفين بالتكامل مع سماعتي Open-Fit RIC.",
                ],
                "tie": "يفند قراءة التوبكون الكاذبة (90 µm)، يثبت بقاء توصيل العصب البصري بـ VEP، ويمنع كارثة Cidamex القلبي الكلوي.",
            },
        ],
    }

    dilemmas = {
        "en": [
            {
                "title": "1. Left Pulsatile Tinnitus & Catheter Angiography (DSA) Deferral",
                "left_lbl": "Clinical Conflict: Pro-DSA (Galhom) vs Anti-DSA (IR) + Neck Pressure",
                "left": "Prof. Galhom raised Cognard Type I occipital dAVF after firm posterior neck pressure silenced tinnitus, while External IR rejected DSA based on normal 3D-TOF MRA. Neither resolved the nocturnal throbbing or checked underlying procedural risk.",
                "right_lbl": "MDT Board Consensus: Acoustic Amplifier + 3-Step Gatekeeper Shield",
                "right": "Firm posterior pressure collapses BOTH Occipital Artery and IJV/emissary veins simultaneously! Paired with MRV Left attenuation (L/R 0.48), Holter post-PVC stroke surges (857 PVCs), and a 3.2 cm3 Type Ad left eardrum amplifier, immediate DSA across a 5 mm Left Carotid Bulb plaque + -2.1 mm ST ischemia + unscreened Metformin/eGFR is contraindicated. Enforce: (1) light anterior IJV vs pinpoint occipital compression + auscultation -> (2) medical/sinus/BP fix -> (3) conditional 4D-MRA / HRCT+CTA after eGFR clearance.",
            },
            {
                "title": "2. White Matter & Movement: CSVD vs MS; ET-Plus vs Vascular PD vs Insulin Dips",
                "left_lbl": "Prior Uncertainty: MS Fears, Unproven Sinemet & Epileptic Tremor Fears",
                "left": "White-matter T2 lesions prompted MS fears, chronic Sinemet conflicted with documented 'Essential tremor + Titubation', and morning shakiness went unmeasured.",
                "right_lbl": "MDT Board Consensus: CSVD Proven, MS/Epilepsy Excluded, CGM + UPDRS Needed",
                "right": "Zero McDonald 2017 Dawson fingers + clean corpus callosum/brainstem + July 2026 ERG retinopathy collapse (0.74 uV OPs) prove optic atrophy and brain spots are diabetic-hypertensive CSVD (Fazekas 1–2) + chronic Rt BG lacune, NOT MS (do NOT do LP!). 2024 2h EEG proves 9 Hz alpha + EMG tremor only (ZERO epilepsy). Audit 80 IU/d insulin for AM hypoglycemia fueling ET-Plus & the 06–09 AM 36.29% PVC surge via 14-day CGM, and test Sinemet via off/on MDS-UPDRS.",
            },
            {
                "title": "3. '4-Hit' Multi-Sensory Vertigo, Bilateral Blindness & VNG Calibration Mandate",
                "left_lbl": "Hidden Risk: Standard Optical VNG Ordered + Chronic Meclizine Sedation",
                "left": "Routine VNG was requested without warning of legal blindness (<1/60), alongside 1-month daily Meclizine + Vincamine (Brain-Ox) suppressing CNS compensation and risk of falls.",
                "right_lbl": "MDT Board Consensus: Adapted Dark VNG + STOP Meclizine + Non-Visual VRT",
                "right": "2021 MOH blindness + 2026 Topcon 40 um / ERG collapse remove ~95% visual balance backup aboard left peripheral vestibulopathy, CSVD/neuropathy, and Holter dips. STOP Meclizine & Brain-Ox immediately; mandate Dark-Only Positional & Bithermal Caloric VNG (CANCEL saccades/pursuit/OPK); institute somatosensory non-visual physiotherapy + home fall-proofing.",
            },
            {
                "title": "4. Solving Hearing Loss (Open-Fit WDRC RIC) & Left Eye Hope (VEP vs OPs) + 'Rt As' Fix",
                "left_lbl": "Prior Clerical & Interpretation Traps: Handwritten 'Rt As' & '90 um' OS Noise",
                "left": "2026-11-03 handwriting falsely stated 'Rt As'; sealed molds risk trapping tinnitus; June 2026 OS Topcon Qual 2 prints artifactual '90 um' RNFL despite chronic blindness.",
                "right_lbl": "MDT Board Consensus: Bilateral Open-Fit RIC (<=100 dB) + Fresh 2026 Eye B-Scan",
                "right": "GSI-39 strip proves Right is Normal Type A (1.5 cm3) vs Extreme Type Ad Left (3.2 cm3, 0 dB ABG, WRS 76% Lt / 84% Rt, Metz +45 dB SL) -> prescribe Bilateral Open-Fit WDRC RIC (vented domes, Left OSPL90 <= 95–100 dB SPL). For OS vision: inner-retinal ischemia is severe (OPs 0.74 uV) making reading recovery unlikely, YET preserved Flash VEP (7.99 uV OS / 10.29 uV OD) demands fresh 2026 B-Scan + IOP protection without systemic Cidamex!",
            },
        ],
        "ar": [
            {
                "title": "1. الطنين النبضي الشمال وحسم جدل القسطرة المخية (DSA)",
                "left_lbl": "التعارض السابق: طلب قسطرة (جلهوم) ضد رفضها (الأشعة التداخلية)",
                "left": "اشتبه أ.د. جلهوم في ناسور قذالي بعد اختفاء الطنين بالضغط خلف الرقبة، بينما رفضت الأشعة التداخلية القسطرة بناءً على سلامة MRA فقط دون خطة بديلة للطنين أو فحص الأمان.",
                "right_lbl": "حسم اللجنة: الفيزياء الصوتية وبروتوكول الفرز الثلاثي الآمن",
                "right": "الضغط القوي خلف الأذن يقفل الشريان القذالي والوريد الوداجي معاً! ومع ضيق الجيب الشمال (0.48) وتضخيم الطبلية 3.2 سم³ لضربات الهولتر (857 PVC)، يمنع عبور بلاك السباتي الـ 5 مم بقسطرة وسط قصور تاجي -2.1 مم وغياب فحص الكلى؛ نبدأ بضغط أمامي خفيف على الوريد vs القذالي + السماعة، ثم العلاج الدوائي، ثم 4D-MRA / HRCT عند اللزوم.",
            },
            {
                "title": "2. البؤر البيضاء والرعشة: CSVD مقابل MS، ورعشة الأساس مقابل الشلل الرعاش وهبوط السكر",
                "left_lbl": "التشويش السابق: خوف من التصلب المتعدد MS والصرع وعدم مراجعة السينيمت",
                "left": "بؤر الرنين أثارت خوفاً من MS، بينما صرف السينيمت تعارض مع تشخيص Essential Tremor + Titubation ولم يُربط السكر بالرجفة الصباحية.",
                "right_lbl": "حسم اللجنة: تصلب شعيرات CSVD فقط (لا MS ولا صرع) + سنسور سكر وتقييم MDS-UPDRS",
                "right": "انعدام أصابع داوسون وانهيار شبكية CSO 2026 (OPs 0.74 µV) يثبتان أن المخ والعيون مصابان بتصلب شعيرات السكر والضغط (ممنوع بذل النخاع!). تخطيط 2024 يثبت ارتعاشاً عضلياً فقط دون صرع. مطلوب CGM لـ 80 وحدة أنسولين (تفسر طفرة 36.29% PVCs 6–9 ص) وفحص MDS-UPDRS للسينيمت.",
            },
            {
                "title": "3. دوخة الضربات الأربع وكف البصر وحماية اختبار الاتزان VNG",
                "left_lbl": "الخطر الصامت: طلب VNG ضوئي عادي وإعطاء Meclizine يومياً شهراً كاملاً",
                "left": "وُصفت الميكليزين والبراين أوكس وطلب VNG دون الانتباه لفقدان البصر &lt; 1/60 مما يعرض المريضة للسقوط ويفشل الاختبار.",
                "right_lbl": "حسم اللجنة: وقف الميكليزين فوراً، تحذير كتابي لـ VNG في الضلمة، وتأهيل حسي",
                "right": "فقدان 95% من تعويض النظر بضمور العصب 40 µm والسيليكون يضاعف أثر ضعف الأذن الداخلية وCSVD وهبوط النبض؛ لذلك نوقف Meclizine وBrain-Ox فوراً، نقتصر في VNG على الوضعيات والحراري في الضلمة، ونطبق تأهيلاً حسياً غير بصري وتأميناً منزلياً.",
            },
            {
                "title": "4. حل السمعيات (Open-Fit WDRC RIC) + حقيقة أمل العين الشمال + تصحيح خطأ 'Rt As'",
                "left_lbl": "أخطاء القراءة القديمة: خط اليد 'Rt As' ووهم 90 µm على جودة التوبكون 2",
                "left": "ملف 03-11 كتب خطأ 'Rt As'، والقوالب المغلقة تحبس الطنين، وجودة التوبكون 2 تعطي رقماً كاذباً لسمك عصب العين الشمال.",
                "right_lbl": "حسم اللجنة: سماعتان Open-Fit RIC بحد &lt;= 100 dB + سونار 2026 وضغط عين",
                "right": "شريط GSI-39 يثبت سلامة الطبلية اليمين 1.5 سم³ Type A ومرونة الشمال 3.2 سم³ (فجوة 0 dB وتمييز 84%/76%)؛ فتُركب سماعتان Open-Fit WDRC RIC بقبة مهواة وسقف &lt;= 95–100 dB. وللعين الشمال: رغم ضعف أمل القراءة لانهيار OPs 0.74 µV، يثبت بقاء Flash VEP (7.99 / 10.29 µV) حياة العصب ويوجب سونار 2026 وحماية ضغط العين بلا Cidamex.",
            },
        ],
    }

    roi_cards = {
        "en": [
            # Historical 2011 Leg Doppler (5)
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Report_DrHelmyRashad_LtCalfPerforatorIncompetence.png", "title": "2011-06-21 Leg Doppler Report (Dr. Helmy Rashad)", "desc": "HISTORICAL: Confirms patent lower-limb arteries (NO PAD) and deep veins (NO DVT) with mild left calf perforator incompetence only."},
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Strip01_AltadamonHospital.png", "title": "2011-06-21 Leg Doppler Strip 01 (Altadamon Hospital)", "desc": "HISTORICAL: Baseline arterial triphasic spectral Doppler printout showing intact lower extremity inflows."},
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Strip02_NormalArteriesDeepVeins.png", "title": "2011-06-21 Leg Doppler Strip 02 (Patent Arteries/Deep Veins)", "desc": "HISTORICAL: Auto-cropped bilateral femoral/popliteal arterial & venous compressibility study excluding DVT."},
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Strip03_Waveforms_Upright.png", "title": "2011-06-21 Leg Doppler Strip 03 (Corrected Upright Waveforms)", "desc": "HISTORICAL: Auto-rotated upright arterial/venous spectral traces confirming absence of chronic PAD occlusion."},
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Strip04_DeepFemoralVenousFlow.png", "title": "2011-06-21 Leg Doppler Strip 04 (Deep Femoral Venous Flow)", "desc": "HISTORICAL: Patent deep femoral/popliteal venous phasic flow trace serving as 15-year vascular baseline."},
            # Historical 2014-2021 Ophthalmology (13)
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT/OPHTH_2014-04-23_Optovue_Macula_OD_CystoidEdema_RPEDisruption_HISTORICAL.png", "title": "2014-04-23 Optovue Macula OD (CME & RPE Disruption)", "desc": "HISTORICAL: Pre-2019 diabetic cystoid macular edema & RPE breakdown in Right eye long predating current workup."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT/OPHTH_2014-04-23_Optovue_MaculaMap_EMM5_OU_RPE_Disruption_OD280_OS100_HISTORICAL.png", "title": "2014-04-23 Optovue EMM5 Macula Map (OD 280 / OS 100)", "desc": "HISTORICAL: Bilateral macular map documenting asymmetric diabetic maculopathy and outer retinal thinning in 2014."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT/OPHTH_2014-04-23_Optovue_OpticNerve_OS_RNFL78.36um_CDVert0.83_HISTORICAL.png", "title": "2014-04-23 Optovue ONH OS (RNFL 78.36 um | C/D 0.83)", "desc": "HISTORICAL: Early advanced vertical cupping (0.83) and inferotemporal RNFL loss in Left eye 5 years prior to 2019 RD surgery."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2016_02_16_Historical_Optovue_ONH/OPHTH_2016-02-16_Optovue_OpticNerve_OS_RNFL71.41um_CDVert0.74_HISTORICAL.png", "title": "2016-02-16 Optovue ONH OS (Progressed RNFL 71.41 um)", "desc": "HISTORICAL: Progressive OS nerve fiber loss from 78.36 -> 71.41 um between 2014 and 2016 before vitrectomy."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-08-19_MOH_HealthInsurance_Referral_LtCataract_RD_PriorRtRD_HISTORICAL.png", "title": "2019-08-19 MOH Insurance Referral (Prior Rt RD -> Lt Surgery)", "desc": "HISTORICAL: Official referral confirming Right cataract+RD surgery had ALREADY taken place prior to 2019 Lt surgery request."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-08-20_Optopol_OCT_Report_OU_DrOmaimaMohamed_MarkedTSNIT_HISTORICAL.png", "title": "2019-08-20 Optopol OCT OU Summary (Dr. Omaima Mohamed)", "desc": "HISTORICAL: Annotated bilateral RNFL & GCC report revealing profound asymmetrical OS disc damage in August 2019."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-08-20_Optopol_OCT_OD_Spectral_MeanTSNIT260_CD0.24_HISTORICAL.png", "title": "2019-08-20 Optopol OD Spectral Scan (Pre-Atrophy 2019)", "desc": "HISTORICAL: Right eye scan prior to severe 2026 atrophy showing preserved 2019 nerve structure (contrasting 40 um in 2026)."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-08-20_Optopol_OCT_OS_SevereRNFLThinning_MeanTSNIT64_HISTORICAL.png", "title": "2019-08-20 Optopol OS Severe RNFL Thinning (64 um Red)", "desc": "HISTORICAL: Crucial proof that OS Mean TSNIT was ALREADY severely thinned (64 um) in 2019 — disproving 2026 '90 um' Topcon noise!"},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-09-11_BScan_OS_VitreousHemorrhage_NoRetinalDetachment_DrOmaimaSharara_HISTORICAL.png", "title": "2019-09-11 Ocular B-Scan OS (Vitreous Hemorrhage)", "desc": "HISTORICAL: Sept 2019 B-Scan documenting dense proliferative diabetic vitreous hemorrhage in Left eye prior to tamponade."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-12-15_BScan_OS_SiliconizedEye_RetinaInPlace_DrHaithamKhalil_HISTORICAL.png", "title": "2019-12-15 Ocular B-Scan OS (Post-Op Siliconized Retina Attached)", "desc": "HISTORICAL: Dec 2019 ultrasound confirming postoperative Left intraocular silicone oil fill with attached retina (Dr. Haitham Khalil)."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans/OPHTH_2021-04-07_BScan_OD_EmulsifiedSiliconeOil_RetinaInPlace_DrYasminSinan_HISTORICAL.png", "title": "2021-04-07 Ocular B-Scan OD (Emulsified Silicone Oil)", "desc": "HISTORICAL: Proves Right eye also underwent silicone tamponade (later cleared to Qual 10 by 2026) with retina in place."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans/OPHTH_2021-04-07_BScan_OS_SiliconFilledEye_RetinaInPlace_DrYasminSinan_HISTORICAL.png", "title": "2021-04-07 Ocular B-Scan OS (Silicon-Filled Eye)", "desc": "HISTORICAL: April 2021 Left eye ultrasound showing complete silicone oil tamponade cavity and attached posterior retina."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans/OPHTH_2021-05-29_BScan_OS_SiliconFilledEye_RetinaInPlace_DrManarElGayar_HISTORICAL.png", "title": "2021-05-29 Last Ocular B-Scan OS (Dr. Manar El-Gayar)", "desc": "HISTORICAL: Latest ocular ultrasound on file (5+ years old) -> mandates fresh 2026 B-Scan + IOP today!"},
            # Historical 2021-2024 Records (2)
            {"grp": "HIST_2024", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/08_Clinical_Notes_and_Prescriptions/CLIN_2021-09-01_MOH_MedicalCouncil_LegalBlindness_LessThan1of60_RetinalOpticAtrophy.jpg", "title": "2021-09-01 MOH Council Legal Blindness (<1/60)", "desc": "HISTORICAL: Official Ministry of Health decree confirming <1/60 vision from retinal & optic atrophy (ID 26304090300188)."},
            {"grp": "HIST_2024", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI5b_2024_CT_Brain_and_ProlongedEEG_TremorArtifacts.png", "title": "July 2024 Brain CT & 2h Prolonged EEG (9 Hz Alpha / Zero Spikes)", "desc": "HISTORICAL: El-Zohor CT confirms persistent Lt silicone oil in 2024; 2h EEG shows 9 Hz alpha + EMG tremor artifact only (zero epilepsy)."},
            # Current 2026-06 / 2026-07 Ophthalmology (12)
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-01_Topcon_OCT_Macula_Radial_OD_Avg221.9um_Qual10.png", "title": "2026-06-01 Topcon Macula OD (Qual 10 | Fovea 259 um)", "desc": "CURRENT 2026: Clear OD optical media (Quality 10/10, silicone removed), attached macula (Avg 221.9 um) with mild foveal attenuation."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OD_RNFL40um_SevereAtrophy_Qual9.png", "title": "2026-06-01 Topcon 3D Disc OD (Qual 9 | Severe RNFL 40 um)", "desc": "CURRENT 2026: Reliable high-quality scan confirming profound OD Optic Atrophy (Total RNFL 40 um, Inf 34 um) — guard perfusion >= 115/70!"},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-01_Topcon_OCT_Macula_Radial_OS_LowQual2_SiliconeMediaOpacity.png", "title": "2026-06-01 Topcon Macula OS (Low Qual 2 | Silicone Opacity)", "desc": "CURRENT 2026: Severe optical scatter from retained OS silicone oil/cataract producing degraded OCT signal (Qual 2, false thickness noise)."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OS_Qual2_MediaArtifact.png", "title": "2026-06-01 Topcon 3D Disc OS (Qual 2 | Artifact '90 um' Trap!)", "desc": "CURRENT 2026: CRITICAL TRAP — Qual 2 prints misleading '90 um' segmentation noise when true OS RNFL was already 64 um in 2019!"},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-27_Zeiss_VisualField_HFA_VFI18pct_UnreliableTest_DrAlaaKhaled.png", "title": "2026-06-27 Zeiss HFA 24-2 Field (VFI 18% | 'Not Reliable Test')", "desc": "CURRENT 2026: Hand-signed 'Not reliable Test' by Dr. Alaa Khaled due to severe <1/60 fixation losses; confirms optical testing limits."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_Flash_VEP_OD_Present_N1P1_10.29uV_P2_125.39ms.png", "title": "2026-07-08 CSO Flash VEP OD (Present | N1-P1 10.29 uV)", "desc": "CURRENT 2026: Clear measurable cortical visual evoked response from Right Eye (P2 125.39 ms) confirming active optic-nerve conduction."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_Flash_VEP_OS_Present_N1P1_7.99uV_P2_132.42ms.png", "title": "2026-07-08 CSO Flash VEP OS (Present | N1-P1 7.99 uV!)", "desc": "CURRENT 2026: PIVOTAL FINDING — Measurable 7.99 uV cortical response proves Left optic-nerve fibers are NOT extinguished despite 0.74 uV OPs!"},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_OscillatoryPotentials_OD10.0uV_OS0.74uV_SevereOSIschemia.png", "title": "2026-07-08 CSO ERG Oscillatory Potentials (OD 10.0 vs OS 0.74 uV)", "desc": "CURRENT 2026: Extreme inner-retinal ischemic collapse in OS (0.74 uV, 92.6% lower than OD 10.0 uV) explaining poor OS acuity potential."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_LightAdapted_3.0_Flicker_OD21.2uV_OS13.0uV.png", "title": "2026-07-08 CSO ERG 30Hz Flicker (OD 21.2 / OS 13.0 uV)", "desc": "CURRENT 2026: Reduced cone-pathway flicker amplitudes OU (worse in OS) excluding demyelinating optic neuritis as primary etiology."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_DarkAdapted_0.01_OU_OD39.1uV_OS35.8uV.png", "title": "2026-07-08 CSO ERG Dark-Adapted 0.01 (OD 39.1 / OS 35.8 uV)", "desc": "CURRENT 2026: Markedly attenuated rod bipolar response OU confirming widespread diabetic-ischemic retinal photoreceptor loss."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_DarkAdapted_3.0_OU_OD94.4uV_OS76.8uV.png", "title": "2026-07-08 CSO ERG Dark-Adapted 3.0 (OD 94.4 / OS 76.8 uV)", "desc": "CURRENT 2026: Mixed rod-cone scotopic b-wave suppression OU confirming severe retinal-vascular disease across both eyes."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_LightAdapted_3.0_OU_OD37.1uV_OS30.4uV.png", "title": "2026-07-08 CSO ERG Light-Adapted 3.0 (OD 37.1 / OS 30.4 uV)", "desc": "CURRENT 2026: Photopic cone single-flash attenuation consistent with chronic bilateral panretinal photocoagulation & ischemia."},
            # Current 2026-11 Neuro / Audio / Holter / Carotid (9)
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI5a_Audiology_2026-11-03_MaskedPTA_and_GSI39_Tymp_Zoom.png", "title": "ROI5a: 2026-11-03 Masked PTA (0 dB ABG) & GSI-39 (Rt 1.5 / Lt 3.2 cm3)", "desc": "CURRENT 2026: Proves Rt WRS 84% / Lt 76%, Rt 1.5 cm3 Type A (overturning 'Rt As') and Lt 3.2 cm3 Extreme Type Ad amplifier."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1a_MRV_Left_Transverse_Sigmoid_IJV_CLAHE_Zoom.png", "title": "ROI1a: 3D-PCA MRV Zoom (Lt Hypoplastic Sinus L/R 0.48)", "desc": "CURRENT 2026: Marker-verified venous drainage; right-dominant drainage with patent attenuated left sinus (zero thrombus)."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1b_MRA_3D_TOF_SourceSlices_SkullBase_Mastoid_Occipital_CLAHE.png", "title": "ROI1b: 3D-TOF Source Slices (Skull Base & Carotid Canal)", "desc": "CURRENT 2026: 1.0-mm axial arterial slices excluding high-flow dAVF or carotid bifurcation aneurysm."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1c_Left_Mastoid_Sigmoid_Occipital_MicroZoom_dAVF_Audit.png", "title": "ROI1c: Left Mastoid-Occipital Micro-Zoom (dAVF Audit)", "desc": "CURRENT 2026: Clean left mastoid air cells and absent dilated occipital cortical venous reflux (Cognard IIb-IV ruled out)."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI2_NeuroOtological_IAC_CPA_Orbits_Ethmoid_CLAHE.png", "title": "ROI2: IAC, CPA, Optic Chiasm, Ethmoiditis & Lt 2.59x Silicone Oil", "desc": "CURRENT 2026: Symmetric IACs/CPAs (no acoustic neuroma), normal optic chiasm (Film10), ethmoiditis, and bright Lt silicone oil."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI3_BasalGanglia_WhiteMatter_LacunarInfarcts_CLAHE.png", "title": "ROI3: Basal Ganglia Lacune & CSVD (Fazekas 1–2, MS Excluded)", "desc": "CURRENT 2026: Hypertensive-diabetic lacunes & CSVD without McDonald Dawson fingers or acute DWI stroke."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/02_MRA_Intracranial_Arteries/MRA_2026-11-05_Video01_48Slice_ZStack_ContactSheet.png", "title": "MRA Z-Stack: 48 Representative Slices Across 513-Slice TOF", "desc": "CURRENT 2026: Full vertex-to-foramen magnum vascular sweep confirming patent Circle of Willis and posterior circulation."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI4_Clinical_Handwriting_and_Holter_KeyStrips_Zoom.png", "title": "ROI4: Galhom Rx/ET Note + Holter 08:00 AM Surge & -2.1 mm ST", "desc": "CURRENT 2026: Prof. Galhom's Essential Tremor log alongside 104 bpm V-run, 164 PVCs/hr AM peak, and CH3 -2.1 mm ischemia."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/05_Vascular_Carotid_Duplex/VASC_2026-11-07_Carotid_Vertebral_ColorDuplex_IMT0.13cm_LtBulbPlaque5mm.jpg", "title": "2026-11-07 Carotid Duplex (IMT 0.13 cm | 5 mm Lt Bulb Plaque)", "desc": "CURRENT 2026: Critical DSA gatekeeper plaque in Left Carotid Bulb + systemic atherosclerosis marker."},
        ],
        "ar": [
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Report_DrHelmyRashad_LtCalfPerforatorIncompetence.png", "title": "تقرير دوبلر الساقين 21-06-2011 (د. حلمي رشاد)", "desc": "فحص قديم: يثبت سلامة شرايين الساقين (لا PAD) والأوردة العميقة (لا DVT) مع ارتجاع بسيط بالساق الشمال فقط."},
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Strip01_AltadamonHospital.png", "title": "شريط دوبلر الساقين 01 — مستشفى التضامن (2011)", "desc": "فحص قديم: موجات تدفق شرياني ثلاثية الطور طبيعية في الساقين."},
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Strip02_NormalArteriesDeepVeins.png", "title": "شريط دوبلر الساقين 02 — سلامة الشرايين والأوردة (2011)", "desc": "فحص قديم: فحص أوردة الفخذ والمأبض خالٍ تماماً من الجلطات العميقة DVT."},
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Strip03_Waveforms_Upright.png", "desc": "فحص قديم: موجات طيفية معدولة الوضع توثق كفاءة التروية الشريانية الطرفية تاريخياً.", "title": "شريط دوبلر الساقين 03 — الموجات الطيفية المعدولة (2011)"},
            {"grp": "VASC_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/10_Peripheral_Vascular_LowerLimb_Doppler/VASC_2011-06-21_LowerLimb_ColorDoppler_Strip04_DeepFemoralVenousFlow.png", "title": "شريط دوبلر الساقين 04 — الوريد الفخذي العميق (2011)", "desc": "فحص قديم: سريان وريدي طبيعي متناسق مع التنفس كمرجع وعائي منذ 15 عاماً."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT/OPHTH_2014-04-23_Optovue_Macula_OD_CystoidEdema_RPEDisruption_HISTORICAL.png", "title": "أوبتوڤيو ماكولة اليمين 23-04-2014 (ارتشاح وتلف RPE)", "desc": "فحص قديم: يثبت وجود اعتلال ماكولة سكري وارتشاح باليمين قبل جراحات الانفصال بـ 5 سنوات."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT/OPHTH_2014-04-23_Optovue_MaculaMap_EMM5_OU_RPE_Disruption_OD280_OS100_HISTORICAL.png", "title": "خريطة ماكولة Optovue للعينين (2014)", "desc": "فحص قديم: يوثق تلف الطبقات الخارجية ومركز الإبصار بالعينين (OD 280 / OS 100) منذ 2014."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT/OPHTH_2014-04-23_Optovue_OpticNerve_OS_RNFL78.36um_CDVert0.83_HISTORICAL.png", "title": "عصب العين الشمال Optovue 2014 (RNFL 78.36 | C/D 0.83)", "desc": "فحص قديم: يثبت بداية تقعر وتآكل العصب البصري بالشمال من 2014 قبل جراحة الشبكية."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2016_02_16_Historical_Optovue_ONH/OPHTH_2016-02-16_Optovue_OpticNerve_OS_RNFL71.41um_CDVert0.74_HISTORICAL.png", "title": "تراجع عصب العين الشمال 16-02-2016 (71.41 µm)", "desc": "فحص قديم: انخفاض تدريجي مستمر في ألياف العصب البصري من 78.36 إلى 71.41 ميكرون."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-08-19_MOH_HealthInsurance_Referral_LtCataract_RD_PriorRtRD_HISTORICAL.png", "title": "إحالة التأمين الصحى 19-08-2019 (انفصال سابق باليمين وطلب الشمال)", "desc": "فحص قديم: وثيقة تاريخية تثبت سبق إجراء عملية الشبكية والكتاركت باليمين قبل أغسطس 2019."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-08-20_Optopol_OCT_Report_OU_DrOmaimaMohamed_MarkedTSNIT_HISTORICAL.png", "title": "تقرير أوبتوبول الشامل للعينين 20-08-2019 (د. أميمة محمد)", "desc": "فحص قديم: مقارنة التآكل الشديد بالعصب الشمال قبل العملية مع حفظ نسبي باليمين وقتها."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-08-20_Optopol_OCT_OD_Spectral_MeanTSNIT260_CD0.24_HISTORICAL.png", "title": "أوبتوبول اليمين 20-08-2019 (قبل ضمور 2026)", "desc": "فحص قديم: يظهر عصب اليمين بمستوى محفوظ نسبياً عام 2019 مقارنة بضموره الشديد الآن (40 µm)."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-08-20_Optopol_OCT_OS_SevereRNFLThinning_MeanTSNIT64_HISTORICAL.png", "title": "أوبتوبول الشمال 20-08-2019 (ضمور شديد 64 µm بالأحمر)", "desc": "فحص قديم: دليل حاسم أن العصب الشمال كان 64 ميكرون في 2019 — يكذب قراءة التوبكون 2026 الكاذبة (90 µm)!"},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-09-11_BScan_OS_VitreousHemorrhage_NoRetinalDetachment_DrOmaimaSharara_HISTORICAL.png", "title": "سونار العين الشمال 11-09-2019 (نزيف جسم زجاجي)", "desc": "فحص قديم: توثيق النزيف الزجاجي السكري بالشمال قبل التدخل الجراحي."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan/OPHTH_2019-12-15_BScan_OS_SiliconizedEye_RetinaInPlace_DrHaithamKhalil_HISTORICAL.png", "title": "سونار العين الشمال 15-12-2019 (بعد حقن السيليكون، الشبكية في مكانها)", "desc": "فحص قديم: يؤكد امتلاء العين الشمال بزيت السيليكون واستقرار الشبكية بعد الجراحة (د. هيثم خليل)."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans/OPHTH_2021-04-07_BScan_OD_EmulsifiedSiliconeOil_RetinaInPlace_DrYasminSinan_HISTORICAL.png", "title": "سونار العين اليمين 07-04-2021 (سيليكون مستحلب)", "desc": "فحص قديم: يثبت وجود سيليكون سابق بالعين اليمنى أيضاً قبل إزالته وصفاء عدستها في 2026."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans/OPHTH_2021-04-07_BScan_OS_SiliconFilledEye_RetinaInPlace_DrYasminSinan_HISTORICAL.png", "title": "سونار العين الشمال 07-04-2021 (مملوءة بالسيليكون)", "desc": "فحص قديم: استمرار زيت السيليكون بالشمال مع بقاء الشبكية مثبتة."},
            {"grp": "OPHTH_HIST", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans/OPHTH_2021-05-29_BScan_OS_SiliconFilledEye_RetinaInPlace_DrManarElGayar_HISTORICAL.png", "title": "آخر سونار للشمال 29-05-2021 (د. منار الجيار)", "desc": "فحص قديم: أحدث سونار متاح عمره الآن أكثر من 5 سنوات — مما يوجب عمل سونار 2026 جديد فوراً!"},
            {"grp": "HIST_2024", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/08_Clinical_Notes_and_Prescriptions/CLIN_2021-09-01_MOH_MedicalCouncil_LegalBlindness_LessThan1of60_RetinalOpticAtrophy.jpg", "title": "قرار القومسيون الطبي 01-09-2021 (كف بصر &lt; 1/60)", "desc": "فحص قديم: يوثق قانونياً ضعف البصر الشديد (&lt; 1/60) من ضمور الشبكية والعصب البصري."},
            {"grp": "HIST_2024", "historical": True, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI5b_2024_CT_Brain_and_ProlongedEEG_TremorArtifacts.png", "title": "مقطعية المخ وتخطيط الـ 2h في يوليو 2024 (لا صرع)", "desc": "فحص قديم: مقطعية 2024 تثبت وجود زيت السيليكون الشمال؛ وتخطيط المخ يثبت إيقاع ألفا 9 هيرتز وارتعاشاً عضلياً فقط."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-01_Topcon_OCT_Macula_Radial_OD_Avg221.9um_Qual10.png", "title": "التوبكون ماكولة اليمين 01-06-2026 (جودة 10 | فجوة 259 µm)", "desc": "حالي 2026: عدسة صافية (أُزيل السيليكون، جودة 10/10) وماكولة مثبتة متوسطة السمك 221.9 µm."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OD_RNFL40um_SevereAtrophy_Qual9.png", "title": "توبكون عصب اليمين 01-06-2026 (جودة 9 | ضمور حاد 40 µm)", "desc": "حالي 2026: فحص عالي الدقة يثبت ضمور عصب بصري شديداً باليمين (RNFL 40 µm، الأسفل 34) يوجب منع هبوط الضغط والسكر!"},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-01_Topcon_OCT_Macula_Radial_OS_LowQual2_SiliconeMediaOpacity.png", "title": "توبكون ماكولة الشمال 01-06-2026 (جودة منخفضة 2 | عتامة السيليكون)", "desc": "حالي 2026: تشتت الضوء بزيت السيليكون والكتاركت يؤدي لانخفاض حاد بدقة القياس بالعين الشمال."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OS_Qual2_MediaArtifact.png", "title": "فخ التوبكون لعصب الشمال 01-06-2026 (جودة 2 | ضوضاء 90 µm كاذبة!)", "desc": "حالي 2026: تحذير — رقم 90 µm هنا مجرد تشويش آلي على جودة 2، بينما العصب الحقيقي كان 64 µm من 2019!"},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField/OPHTH_2026-06-27_Zeiss_VisualField_HFA_VFI18pct_UnreliableTest_DrAlaaKhaled.png", "title": "مجال إبصار زايس 27-06-2026 (VFI 18% | غير موثوق — د. آلاء خالد)", "desc": "حالي 2026: وقع عليه الطبيب 'Not reliable Test' لكثرة فقد التثبيت البصري نتيجة ضعف الإبصار الشديد."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_Flash_VEP_OD_Present_N1P1_10.29uV_P2_125.39ms.png", "title": "رسم العصب الضوئي Flash VEP يمين 08-07-2026 (10.29 µV)", "desc": "حالي 2026: استجابة قشرية واضحة ومقروءة من العين اليمنى تثبت سلامة التوصيل العصبي للقشرة."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_Flash_VEP_OS_Present_N1P1_7.99uV_P2_132.42ms.png", "title": "دليل بقاء توصيل العصب الشمال: Flash VEP 08-07-2026 (7.99 µV!)", "desc": "حالي 2026: اكتشاف فارق — العصب البصري الشمال ما زال يرسل إشارة مقاسة للمخ (7.99 µV) رغم تضرر الشبكية!"},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_OscillatoryPotentials_OD10.0uV_OS0.74uV_SevereOSIschemia.png", "title": "انهيار شبكية الشمال ERG OPs 08-07-2026 (0.74 مقابل 10.0 µV يمين)", "desc": "حالي 2026: انهيار بنسبة 92.6% في استجابة الشبكية الداخلية بالشمال يفسر لماذا استعادة القراءة فيها مستبعدة."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_LightAdapted_3.0_Flicker_OD21.2uV_OS13.0uV.png", "title": "وميض المخاريط 30Hz Flicker بالـ CSO 08-07-2026", "desc": "حالي 2026: ضعف واضح لاستجابة المخاريط بالعينين يثبت أن المشكلة شبكية وعائية لا التهاب عصب MS."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_DarkAdapted_0.01_OU_OD39.1uV_OS35.8uV.png", "title": "رسم شبكية الإظلام Dark 0.01 (39.1 يمين / 35.8 شمال)", "desc": "حالي 2026: هبوط استجابة العصي الشبكية بالعينين نتيجة السكري التكاثري وجلسات الليزر والجراحات."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_DarkAdapted_3.0_OU_OD94.4uV_OS76.8uV.png", "title": "رسم الشبكية المختلط Dark 3.0 (94.4 يمين / 76.8 شمال)", "desc": "حالي 2026: ضعف عام متوقع متمشٍ مع تاريخ الليزر الشامل وانفصال الشبكية السكري."},
            {"grp": "OPHTH_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG/OPHTH_2026-07-08_CSO_ERG_LightAdapted_3.0_OU_OD37.1uV_OS30.4uV.png", "title": "رسم الشبكية المضيء Light 3.0 (37.1 يمين / 30.4 شمال)", "desc": "حالي 2026: يؤكد تلف المستقبلات الضوئية المزمن بالعينين مع تفوق الشمال في التلف الإقفاري."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI5a_Audiology_2026-11-03_MaskedPTA_and_GSI39_Tymp_Zoom.png", "title": "ROI5a: سمعيات 03-11-2026 المقنعة وشريط GSI-39 (1.5 / 3.2 سم³)", "desc": "حالي 2026: يثبت فجوة 0 dB وتمييز 84%/76% ويمين 1.5 Type A وشمال 3.2 سم³ Type Ad."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1a_MRV_Left_Transverse_Sigmoid_IJV_CLAHE_Zoom.png", "title": "ROI1a: تكبير أوردة المخ MRV (ضمور الشمال بنسبة 0.48)", "desc": "حالي 2026: يوثق سيادة الجانب الأيمن وتضيق الجيب المستعرض والسيني الشمال دون أي جلطات."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1b_MRA_3D_TOF_SourceSlices_SkullBase_Mastoid_Occipital_CLAHE.png", "title": "ROI1b: شرائح MRA الدقيقة لقاع الجمجمة والماستويد", "desc": "حالي 2026: مقاطع محورية 1.0 مم تستبعد أي تمدد شرياني أو ناسور عالي التدفق."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI1c_Left_Mastoid_Sigmoid_Occipital_MicroZoom_dAVF_Audit.png", "title": "ROI1c: التكبير الدقيق للماستويد والمنطقة القذالية", "desc": "حالي 2026: هوائية ممتازة بالماستويد الشمال وغياب أي احتقان وريدي قشري Cognard IIb–IV."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI2_NeuroOtological_IAC_CPA_Orbits_Ethmoid_CLAHE.png", "title": "ROI2: قنوات السمع IAC والتصالب البصري وزيت السيليكون 2.59x", "desc": "حالي 2026: سلامة عصب السمع والتصالب البصري (Film10)، التهاب جيوب غربالية، وزيت السيليكون بالشمال."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI3_BasalGanglia_WhiteMatter_LacunarInfarcts_CLAHE.png", "title": "ROI3: جلطات العقد القاعدية الفجوية وتصلب المادة البيضاء CSVD", "desc": "حالي 2026: جلطات لاكونار مزمنة وبؤر CSVD Fazekas 1–2 مع استبعاد تام للـ MS."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/02_MRA_Intracranial_Arteries/MRA_2026-11-05_Video01_48Slice_ZStack_ContactSheet.png", "title": "لوحة MRA Z-Stack: 48 شريحة من فيديو 3D-TOF الـ 513 شريحة", "desc": "حالي 2026: مسح شرياني كامل من قاع الجمجمة لقمته يؤكد سلامة الشرايين المخية."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA/ROI4_Clinical_Handwriting_and_Holter_KeyStrips_Zoom.png", "title": "ROI4: روشتة وملاحظة أ.د. جلهوم + شرائط الهولتر", "desc": "حالي 2026: توثيق Essential Tremor بيد د. جلهوم بجانب نوبة 104 ن/د وهبوط ST -2.1 مم."},
            {"grp": "NEURO_AUDIO_2026", "historical": False, "img": "../01_Structured_and_Renamed_Inputs/05_Vascular_Carotid_Duplex/VASC_2026-11-07_Carotid_Vertebral_ColorDuplex_IMT0.13cm_LtBulbPlaque5mm.jpg", "title": "دوبلر شرايين الرقبة 07-11-2026 (IMT 0.13 سم | بلاك 5 مم)", "desc": "حالي 2026: البلاك المانع للقسطرة الفورية في عنق الشريان السباتي الشمال."},
        ],
    }

    med_audit = {
        "en": [
            {"tag": "STOP", "badge": "b-red", "med": "Angiopress Comp (Captopril 50 / HCTZ 25 OD) + Proposed Cidamex", "act": "STOP BOTH / Switch BP", "mech": "Short Captopril t1/2 + synergistic proximal/distal renal K+/Mg2+ wasting.", "hazard": "Leaves 16h nocturnal BP spikes fueling venous tinnitus while depleting K+ to ignite 857 PVCs & 4 V-runs.", "res": "Replace Angiopress Comp with 24h Perindopril/Telmisartan (+/- Amlodipine) after K+/Cr check; strictly prohibit oral Cidamex!"},
            {"tag": "STOP", "badge": "b-red", "med": "Aggrex (Aspirin 75 mg) + Cetrovit Ginkgo + Ciprapra 5 mg", "act": "STOP Ginkgo Today", "mech": "Irreversible COX-1 blockade + PAF antagonism + SSRI platelet 5-HT depletion.", "hazard": "Triple antiplatelet bleed risk (GI & cerebral microbleeds in CSVD) without proven cochlear benefit.", "res": "STOP Cetrovit Ginkgo immediately; maintain Aggrex 75 mg OD (+ PPI if GI symptoms); reassess Ciprapra need."},
            {"tag": "STOP", "badge": "b-red", "med": "Meclizine 25 mg qPM x 1 Month (ENT Order)", "act": "STOP Immediately", "mech": "Central H1 histamine & strong muscarinic anticholinergic sedation.", "hazard": "Blocks central brain vestibular compensation — catastrophic fall hazard under <1/60 blindness!", "res": "STOP regular daily Meclizine; unlock VRT compensation; reserve strictly <48h for severe acute spinning only."},
            {"tag": "STOP", "badge": "b-red", "med": "Brain-Ox 30 mg (Vincamine) + Antihypertensives", "act": "Deprescribe Brain-Ox", "mech": "Cerebral vasodilator stacking with antihypertensives & beta-blocker.", "hazard": "Risks nocturnal systemic hypotension & coronary/optic-nerve watershed hypoperfusion (40 um RNFL & -2.1 mm ST).", "res": "Discontinue Brain-Ox once smooth 24h antihypertensive & High-Intensity Statin targets are established."},
            {"tag": "CAUTION", "badge": "b-amber", "med": "Inderal 10 mg BID (Propranolol) + 80 IU/d Insulin (Toujeo 30 + Apidra 50)", "act": "CGM & Switch Review", "mech": "Non-selective beta1/beta2 blockade + heavy basal-bolus insulin load.", "hazard": "Masks hypoglycemic tachycardia & blunts hepatic glycogen rescue -> aligns directly with 06–09 AM 36.29% PVC cluster.", "res": "Start 14-day CGM (03–09 AM) & Endocrinology insulin titration; ask Cardiology to transition to cardioselective Bisoprolol 2.5–5 mg OD."},
            {"tag": "CAUTION", "badge": "b-amber", "med": "Gliptus Plus (Vildagliptin / Metformin 1000 mg) + Contrast Imaging", "act": "Check eGFR / 48h Hold", "mech": "Renally cleared biguanide paired with iodinated CT/MR/DSA contrast.", "hazard": "Metformin-Associated Lactic Acidosis (MALA) risk if eGFR <30 or contrast given unheld.", "res": "Verify Cr/eGFR urgently; STRICTLY HOLD Metformin 48 hours around any gatekeeper 4D-MRA / CTA / DSA."},
            {"tag": "UPGRADE", "badge": "b-blue", "med": "Zatorva 20 mg (Atorvastatin) Sub-Optimal Dose", "act": "Upgrade to 40–80 mg", "mech": "Moderate-intensity HMG-CoA reductase inhibition.", "hazard": "Insufficient plaque stabilization for 5 mm Lt bulb plaque, 0.13 cm IMT, CSVD & -2.1 mm silent ischemia (target LDL <55).", "res": "Escalate to Atorvastatin 40–80 mg nocte (or Rosuvastatin 20 mg) + check lipids & ALT/AST in 6–8 wks."},
            {"tag": "CAUTION", "badge": "b-amber", "med": "Topical Glaucoma Guardrail & Nasal Mometasone 50 mcg", "act": "B-Scan + 4–6w IOP Check", "mech": "Local silicone oil trabecular emulsification + minimal nasal steroid absorption.", "hazard": "Untreated emulsified oil IOP threatens last 7.99 uV OS VEP & 40 um OD nerve; oral Cidamex is hazardous!", "res": "Obtain Fresh 2026 Ocular B-Scan + Applanation IOP today; use topical drops ONLY if high & check IOP 4–6 wks post-Mometasone."},
        ],
        "ar": [
            {"tag": "STOP", "badge": "b-red", "med": "Angiopress Comp (كابتوبريل / HCTZ) + اقتراح Cidamex الفموي", "act": "إيقاف ومنع الدمج", "mech": "مفعول كابتوبريل قصير + فقدان مزدوج للبوتاسيوم والماغنيسيوم بالكلى.", "hazard": "يترك ارتفاعاً ليلياً للضغط يغذي الطنين النبضي ويشعل الـ 857 خفقة بطينية و4 نوبات V-runs.", "res": "الاستبدال بـ Perindopril أو Telmisartan طويل المفعول بعد فحص البوتاسيوم والكلى، وحظر Cidamex بالفم نهائياً!"},
            {"tag": "STOP", "badge": "b-red", "med": "Aggrex 75 + Cetrovit Ginkgo + Ciprapra 5 mg", "act": "وقف Ginkgo فوراً", "mech": "تثبيط COX-1 + PAF + تقليل السيروتونين بالصفائح الدموية معاً.", "hazard": "تداخل نزفي ثلاثي بالصفائح في مريضة CSVD دون أي دليل علمي على فائدة الجينكو لضعف القوقعة.", "res": "وقف Cetrovit Ginkgo اليوم، الاستمرار على Aggrex 75 (+ حماية معدة عند اللزوم)، ومراجعة الحاجة لـ Ciprapra."},
            {"tag": "STOP", "badge": "b-red", "med": "Meclizine 25 mg يومياً مساءً لمدة شهر", "act": "وقف فوري", "mech": "مضاد هيستامين وكولين مركزي مثبط للدماغ وجهاز الاتزان.", "hazard": "يمنع تعويض المخ للدوخة ويضاعف السقوط لدى مريضة فاقدة للبصر (&lt; 1/60).", "res": "إيقاف الميكليزين اليومي فوراً لفتح باب التأهيل الحسي (ويقتصر <48 ساعة للطوارئ الدوارية الحادة فقط)."},
            {"tag": "STOP", "badge": "b-red", "med": "Brain-Ox 30 mg (فينكامين) فوق أدوية الضغط", "act": "سحب Brain-Ox", "mech": "توسيع شرياني إضافي غير مثبت الفائدة في الجلطات الفجوية.", "hazard": "قد يسبب هبوط ضغط ليلي يضر العصب البصري (40 µm) والتروية التاجية الصامتة (-2.1 مم).", "res": "إيقاف Brain-Ox مع ضبط الضغط على مدار 24 ساعة ورفع جرعة الستاتين."},
            {"tag": "CAUTION", "badge": "b-amber", "med": "Inderal 10 BID + 80 وحدة أنسولين يومياً (Toujeo 30 + Apidra 50)", "act": "تركيب CGM ومراجعة", "mech": "حجب مستقبلات β1 و β2 مع جرعة أنسولين مرتفعة صباحاً/مساءً.", "hazard": "يخفي أعراض هبوط السكر المنذرة (النبض السريع) الموافقة لفورة 36.29% PVCs الساعة 6–9 صباحاً.", "res": "تركيب سنسور سكر CGM 14 يوماً + مراجعة الغدد لتخفيف الأنسولين، والتنسيق مع القلب للتحويل إلى Bisoprolol 2.5–5 mg."},
            {"tag": "CAUTION", "badge": "b-amber", "med": "Gliptus Plus (ميتفورمين 1000) + فحوصات الصبغة", "act": "فحص eGFR وإيقاف 48 س", "mech": "إخراج كلوي للميتفورمين مع صبغات اليود التشخيصية.", "hazard": "خطر الحماض اللاكتيكي (MALA) إذا كان eGFR منخفضاً أو أُجريت صبغة CTA/DSA دون توقف مسبق.", "res": "فحص Creatinine/eGFR فوراً وإيقاف الميتفورمين إلزامياً 48 ساعة قبل وبعد أي أشعة بالصبغة."},
            {"tag": "UPGRADE", "badge": "b-blue", "med": "Zatorva 20 mg (أتورفاستاتين) جرعة غير كافية", "act": "رفع لـ 40–80 مجم", "mech": "خفض كوليسترول متوسط القوة في حالة وعائية عالية الخطورة.", "hazard": "لا يحقق هدف LDL &lt; 55 المطلوب لبلاك السباتي 5 مم، تصلب المخ، وقصور القلب -2.1 مم.", "res": "الترقية إلى Atorvastatin 40–80 mg (أو Rosuvastatin 20 mg) ليلاً مع متابعة الدهون والإنزيمات بعد 6–8 أسابيع."},
            {"tag": "CAUTION", "badge": "b-amber", "med": "أمان العصب البصري مع السيليكون وبخاخ Mometasone", "act": "سونار B-Scan وقياس IOP", "mech": "احتمال تسحلب السيليكون بالعين مع متابعة امتصاص الكورتيكوستيرويد الأنفي.", "hazard": "ارتفاع ضغط العين الخفي يهدد ما تبقى في VEP الشمال (7.99 µV) وعصب اليمين 40 µm.", "res": "عمل سونار 2026 وقياس ضغط العين فوراً + قطرات موضعية فقط عند اللزوم + إعادة قياس الضغط بعد 4–6 أسابيع من البخاخ."},
        ],
    }

    rx_table = {
        "en": [
            {"domain": "Cardiology, BP & Anti-Ischemic Core", "drug": "<strong>Bisoprolol 2.5–5 mg OD</strong> (replacing Inderal after Cardiology review) + <strong>Perindopril 5 mg / Telmisartan 40–80 mg</strong> (+/- Amlodipine) + <strong>Atorvastatin 40–80 mg nocte</strong> + <strong>Aggrex 75 mg OD</strong>", "timing": "AM Beta/ACEi; Lunch Aspirin; Nocte Statin | Check K+/Cr/Echo/Stress", "rationale": "Treats -2.1 mm silent CH3 ischemia, blunts AM PVC surge, stops HCTZ K+ wasting & Captopril troughs, stabilizes 5 mm Lt bulb plaque (LDL <55), keeps smooth BP >= 115/70 for optic disc perfusion, and eliminates Ginkgo/Brain-Ox hazards."},
            {"domain": "Vitreoretinal, Glaucoma & Low-Vision Care", "drug": "Urgent <strong>2026 Ocular B-Scan OU + Applanation IOP + Gonioscopy</strong> -> <strong>Topical Anti-Glaucoma Drops ONLY</strong> if IOP high (NEVER oral Cidamex!) + Focus low-vision aids on <strong>OD (Qual 10)</strong>", "timing": "STAT Ophth visit + IOP recheck 4–6 wks after nasal spray", "rationale": "Confirms attached retina behind Qual 2 OS silicone oil (~7 yrs), guards surviving bilateral VEP conduction (7.99 / 10.29 uV) and severely thinned 40 um OD optic nerve without hypokalemic diuretics."},
            {"domain": "Diabetes, CGM & Hypoglycemia Fix", "drug": "<strong>Endocrinology Insulin De-Escalation</strong> (80 IU/d Toujeo+Apidra) + Verify <strong>Gliptus Plus (eGFR >= 45)</strong> +/- <strong>SGLT2i (Empagliflozin 10 mg)</strong>", "timing": "14-day CGM (03–09 AM focus); HOLD Metformin 48h pre-contrast", "rationale": "Breaks morning adrenergic hypoglycemia driving BOTH tremors & the 06–09 AM 36.29% PVC cluster; prevents nocturnally blind falls & protects optic disc."},
            {"domain": "ENT, Hearing Aid & Blindness-Adapted VRT", "drug": "<strong>Bilateral Open-Fit WDRC RIC Aids</strong> (open domes, Lt OSPL90 <= 95–100 dB) + <strong>STOP Meclizine & Ginkgo</strong> + <strong>Mometasone 50 mcg</strong> + saline spray + <strong>30 deg Bed Wedge</strong>", "timing": "Daily non-visual somatosensory VRT; written Dark-Only VNG order", "rationale": "Honors Metz (+45 dB SL), prevents mold occlusion trapping venous hum, unlocks CNS vestibular compensation, clears ethmoiditis, and cuts nighttime venous back-pressure."},
            {"domain": "Neurology & DSA Gatekeeper Algorithm", "drug": "Formal <strong>Off/On Sinemet MDS-UPDRS Test</strong> + Continue <strong>Cobal F</strong> (post-B12 lab) + Defer Catheter DSA pending <strong>Bedside Light IJV vs Focal Occipital Compression</strong>", "timing": "Neurology clinic visit (avoid sedating Primidone at night)", "rationale": "Verifies true Levodopa benefit vs Galhom's Essential Tremor + Titubation, avoids dangerous blind sedatives, and protects the 5 mm Lt Carotid Bulb plaque from unscreened catheter passage."},
            {"domain": "Peripheral Vascular (2011) & Fall Safety", "drug": "Check pedal arterial pulses + <strong>Daytime 15–20 mmHg Knee Compression Stockings</strong> (if pulses intact) + Home lighting/grab-bars/white-cane orientation", "timing": "Daytime only (off at night); Podiatry DM foot check", "rationale": "Manages chronic 2011 calf perforator varicosities safely under diabetes while preventing multi-sensory hip/head fractures indoors."},
        ],
        "ar": [
            {"domain": "القلب والضغط وحماية الشرايين", "drug": "<strong>Bisoprolol 2.5–5 mg صباحاً</strong> (بدلاً من Inderal بموافقة القلب) + <strong>Perindopril 5 mg أو Telmisartan 40–80 mg</strong> (+/- أملوديبين) + <strong>Atorvastatin 40–80 mg ليلاً</strong> + <strong>Aggrex 75 mg</strong>", "timing": "الضغط صباحاً؛ الأسبرين غداءً؛ الستاتين ليلاً | إيكو ومراجعة K+/كلى", "rationale": "يعالج القصور التاجي الصامت -2.1 مم، يهدئ الخفقات الصباحية، يمنع نقص بوتاسيوم HCTZ، يثبت بلاك الـ 5 مم (LDL <55)، يحفظ ضغط العصب البصري (>= 115/70)، ويزيل الجينكو وBrain-Ox."},
            {"domain": "الشبكية والجلوكوما وحماية البصر", "drug": "<strong>سونار B-Scan 2026 عاجل + قياس IOP + فحص زاوية</strong> -> <strong>قطرات موضعيه فقط</strong> عند ارتفاع الضغط (ممنوع Cidamex فموياً!) + تأهيل ضعاف البصر لـ <strong>العين اليمنى</strong>", "timing": "عيادة عيون عاجلة + إعادة قياس الضغط بعد 4–6 أسابيع من البخاخ", "rationale": "يطمئن على استقرار الشبكية خلف السيليكون، يحفظ توصيل VEP الحي بالعينين (7.99 / 10.29 µV) وعصب اليمين 40 µm دون تدمير أملاح القلب."},
            {"domain": "السكري ومنع هبوط السكر الصباحي", "drug": "<strong>إعادة معايرة جرعة الـ 80 وحدة أنسولين</strong> (Toujeo+Apidra) + فحص <strong>eGFR لـ Gliptus Plus</strong> +/- إضافة <strong>SGLT2i (إمباغليفلوزين 10)</strong>", "timing": "سنسور CGM 14 يوماً (3–9 ص)؛ وقف ميتفورمين 48 س قبل أي صبغة", "rationale": "يوقف هبوط السكر الصباحي المسبب للرعشة الأدرينالية وطفرة الـ 36.29% للـ PVCs الساعة 6–9 صباحاً."},
            {"domain": "الأذن والسمع والجيوب والاتزان", "drug": "<strong>سماعتان Open-Fit WDRC RIC</strong> (بقمع مهوّى وحد أقصى <= 95–100 dB بالشمال) + <strong>وقف Meclizine و Ginkgo</strong> + بخاخ <strong>Mometasone</strong> ومحلول ملح + مخدة 30 درجة", "timing": "علاج طبيعي حسي يومي للاتزان + VNG بالضلمة فقط", "rationale": "يحل ضعف السمع بأمان مع حساسية Metz، يمنع كتم الطنين، ينشط تعويض المخ، ويعالج الجيوب الغربالية وضغط الأوردة الليلي."},
            {"domain": "المخ والأعصاب وبوابة الـ DSA", "drug": "اختبار <strong>MDS-UPDRS</strong> مع وبدون Sinemet + استمرار <strong>Cobal F</strong> (بعد تحليل B12) + <strong>اختبار الضغط اللطيف للوريد vs القذالي</strong> قبل أي أشعة صبغة", "timing": "عيادة أعصاب (ممنوع المهدئات الليلية كالـ Primidone)", "rationale": "يفصل بين رعشة الأساس والباركنسون الوعائي، ويحمي بلاك السباتي الـ 5 مم من قسطرة غير مبررة."},
            {"domain": "أوردة الساقين وتأمين المنزل", "drug": "فحص نبض شرايين القدم + <strong>شراب دوالي خفيف 15–20 مم نهاراً</strong> (عند سلامة الشرايين) + تأمين الحمام والعتبات من السقوط", "timing": "نهاراً فقط؛ متابعة قدم سكري", "rationale": "يعالج ارتجاع الثواقب السطحي بأمان ويقي من كسور السقوط في مريضة فاقدة للبصر."},
        ],
    }

    checklist = {
        "en": [
            {"id": "c1", "txt": "STAT Blood Panel: K+, Na+, Mg2+, Ca2+, Serum Creatinine/eGFR, HbA1c, Fasting Lipids (LDL), CBC, TSH, Vitamin B12, ALT/AST."},
            {"id": "c2", "txt": "STOP daily Meclizine, Cetrovit Ginkgo & Brain-Ox immediately; PROHIBIT systemic oral Acetazolamide (Cidamex) alongside HCTZ."},
            {"id": "c3", "txt": "Urgent Ophthalmology Evaluation: Fresh 2026 Bilateral Ocular B-Scan + Applanation IOP + Gonioscopy (treat raised IOP with TOPICAL drops ONLY)."},
            {"id": "c4", "txt": "Written VNG Lab Safety Notice: State <1/60 blindness upfront — CANCEL optical saccade/pursuit/OPK calibration; run DARK positional & caloric tests ONLY."},
            {"id": "c5", "txt": "Execute Bedside Differential Vascular Test: Quiet mastoid auscultation + light anterior Left IJV pressure vs pinpoint Left Occipital Artery compression."},
            {"id": "c6", "txt": "Initiate 14-Day CGM (FreeStyle Libre) or 7-pt SMBG inspecting 03:00–09:00 AM glucose dips fueling 36.29% morning PVCs + Endocrinology insulin audit."},
            {"id": "c7", "txt": "Cardiology Consultation: TTE + Pharmacological Stress Echo/SPECT (NO treadmill!) for -2.1 mm ST depression + approval to switch Angiopress Comp / Inderal / Zatorva 20."},
            {"id": "c8", "txt": "Neurology In-Person Exam: Off vs On Sinemet MDS-UPDRS Part III testing to separate Essential Tremor/Titubation (Galhom) from Vascular Parkinsonism."},
            {"id": "c9", "txt": "Fit Bilateral Open-Fit WDRC RIC Hearing Aids (soft vented domes, strict Left OSPL90 <= 95–100 dB SPL) + Start Mometasone 50 mcg spray & 30 deg head-of-bed elevation."},
            {"id": "c10", "txt": "Begin Daily Non-Visual Somatosensory Balance Physiotherapy (VRT), Home Fall-Proofing, Podiatry Check & Recheck IOP 4–6 wks after starting Mometasone."},
        ],
        "ar": [
            {"id": "c1", "txt": "تحاليل دم عاجلة اليوم: البوتاسيوم K+، الصوديوم، الماغنيسيوم، الكالسيوم، وظائف الكلى Cr/eGFR، السكر التراكمي، الدهون الكاملة، صورة الدم، الغدة الدرقية TSH، وفيتامين B12."},
            {"id": "c2", "txt": "إيقاف Meclizine و Cetrovit Ginkgo و Brain-Ox فوراً اليوم؛ ومنع استخدام Cidamex (أسيتازولاميد) بالفم مع مدّر البول الثيازيد."},
            {"id": "c3", "txt": "فحص عيون عاجل: سونار B-Scan 2026 جديد للعينين + قياس ضغط العين + فحص الزاوية (وعلاج أي ضغط مرتفع بقطرات موضعية فقط)."},
            {"id": "c4", "txt": "تنبيه كتابي لمعمل الـ VNG: توثيق فقدان البصر &lt; 1/60 لإلغاء تتبع الضوء والاعتماد على الوضعيات والاختبار الحراري في الضلمة فقط."},
            {"id": "c5", "txt": "اختبار العيادة التفرقي: تسمع بالسماعة خلف الأذن ومقارنة ضغط الوريد الوداجي الأمامي الخفيف مقابل الضغط النقطي على الشريان القذالي."},
            {"id": "c6", "txt": "تركيب سنسور سكر CGM 14 يوماً (أو 7 قياسات) لفحص فترة 3–9 صباحاً ومراجعة الـ 80 وحدة أنسولين مع طبيب السكر والغدد الصماء."},
            {"id": "c7", "txt": "استشارة القلب: إيكو TTE + مسح ذري/إيكو دوائي (دون سير كهربائي!) لتقييم هبوط ST -2.1 مم والتحويل لـ Bisoprolol + Perindopril/Telmisartan + Atorvastatin 40-80."},
            {"id": "c8", "txt": "فحص أعصاب حضوري MDS-UPDRS مع وبدون السينيمت لحسم دوره مقابل رعشة الأساس (أ.د. جلهوم) والرعشة الوعائية."},
            {"id": "c9", "txt": "تركيب سماعتين رقميتين Open-Fit WDRC RIC بقبة مهواة وسقف خروج &lt;= 95–100 dB للأذن الشمال + بخاخ Mometasone ورفع الرأس 30 درجة وقت النوم."},
            {"id": "c10", "txt": "بدء علاج طبيعي حسي للاتزان (VRT غير بصري)، تأمين المنزل من السقوط، وإعادة قياس ضغط العين بعد 4–6 أسابيع من بخاخ الجيوب."},
        ],
    }

    return json.dumps(
        {
            "holter_hours": holter_hours,
            "files_index": files_index,
            "consultants": consultants,
            "dilemmas": dilemmas,
            "roi_cards": roi_cards,
            "med_audit": med_audit,
            "rx_table": rx_table,
            "checklist": checklist,
            "i18n": i18n,
        },
        ensure_ascii=False,
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" dir="ltr" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MDT Clinical Case Study — Amal Mokhtar Mohamed Ibrahim (63.5y F)</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%230F172A'/%3E%3Cpath d='M7 16h5l2.5-6 3.5 12 2.5-6H25' stroke='%230284C7' stroke-width='2.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&family=Tajawal:wght@500;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg-mesh: radial-gradient(circle at 12% 10%, rgba(14, 165, 233, 0.14), transparent 36%),
             radial-gradient(circle at 88% 18%, rgba(99, 102, 241, 0.12), transparent 38%),
             radial-gradient(circle at 50% 90%, rgba(20, 184, 166, 0.10), transparent 45%),
             #edf2f7;
  --glass-bg: rgba(255, 255, 255, 0.80);
  --glass-strong: rgba(255, 255, 255, 0.92);
  --glass-border: rgba(148, 163, 184, 0.34);
  --glass-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  --header-bg: rgba(15, 23, 42, 0.88);
  --text: #0f172a;
  --text-muted: #475569;
  --accent: #0284c7;
  --accent-dark: #0369a1;
  --accent-soft: rgba(2, 132, 199, 0.12);
  --red: #dc2626;
  --red-bg: rgba(254, 242, 242, 0.88);
  --red-border: rgba(239, 68, 68, 0.42);
  --amber: #b45309;
  --amber-bg: rgba(255, 251, 235, 0.88);
  --amber-border: rgba(245, 158, 11, 0.45);
  --green: #15803d;
  --green-bg: rgba(240, 253, 244, 0.88);
  --green-border: rgba(34, 197, 94, 0.42);
  --blue-bg: rgba(239, 246, 255, 0.88);
  --blue-border: rgba(59, 130, 246, 0.42);
  --code-bg: rgba(15, 23, 42, 0.06);
  --table-head: rgba(241, 245, 249, 0.92);
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-display: 'Plus Jakarta Sans', 'Inter', sans-serif;
}

html[lang="ar"] {
  --font-sans: 'Cairo', 'Tajawal', 'Segoe UI', Tahoma, sans-serif;
  --font-display: 'Cairo', 'Tajawal', sans-serif;
}

[data-theme="dark"] {
  --bg-mesh: radial-gradient(circle at 15% 10%, rgba(14, 165, 233, 0.18), transparent 36%),
             radial-gradient(circle at 85% 20%, rgba(99, 102, 241, 0.16), transparent 40%),
             #0b1120;
  --glass-bg: rgba(17, 24, 39, 0.78);
  --glass-strong: rgba(30, 41, 59, 0.90);
  --glass-border: rgba(148, 163, 184, 0.22);
  --glass-shadow: 0 12px 32px rgba(0, 0, 0, 0.42);
  --header-bg: rgba(9, 13, 24, 0.92);
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --accent-dark: #0ea5e9;
  --accent-soft: rgba(56, 189, 248, 0.16);
  --red: #f87171;
  --red-bg: rgba(69, 10, 10, 0.58);
  --red-border: rgba(248, 113, 113, 0.40);
  --amber: #fbbf24;
  --amber-bg: rgba(69, 26, 3, 0.55);
  --amber-border: rgba(251, 191, 36, 0.40);
  --green: #4ade80;
  --green-bg: rgba(5, 46, 22, 0.55);
  --green-border: rgba(74, 222, 128, 0.38);
  --blue-bg: rgba(23, 37, 84, 0.55);
  --blue-border: rgba(96, 165, 250, 0.38);
  --code-bg: rgba(255, 255, 255, 0.09);
  --table-head: rgba(15, 23, 42, 0.92);
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; overflow-x: clip; }
body {
  margin: 0;
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.6;
  color: var(--text);
  background: var(--bg-mesh);
  background-attachment: fixed;
  min-height: 100vh;
  overflow-x: clip;
}
h1, h2, h3, h4 {
  font-family: var(--font-display);
  margin: 0 0 8px 0;
  line-height: 1.3;
}
code {
  font-family: 'Cascadia Code', Consolas, monospace;
  font-size: 0.88em;
  background: var(--code-bg);
  padding: 2px 6px;
  border-radius: 5px;
  direction: ltr;
  unicode-bidi: isolate;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* STICKY GLASS HEADER */
.glass-header {
  position: sticky;
  top: 0;
  z-index: 60;
  background: var(--header-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  color: #f8fafc;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  padding: 12px 20px;
}
.header-inner {
  max-width: 1480px;
  margin: 0 auto;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.brand-box h1 {
  font-size: 17px;
  font-weight: 800;
  color: #f8fafc;
  margin-bottom: 2px;
}
.brand-sub {
  font-size: 12px;
  color: #cbd5e1;
}
.header-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

/* SEGMENTED EN | عربي مصري PILL TOGGLE */
.lang-pill {
  display: inline-flex;
  align-items: center;
  padding: 3px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.24);
  backdrop-filter: blur(10px);
}
.lang-opt {
  border: none;
  background: transparent;
  color: #cbd5e1;
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 700;
  padding: 5px 12px;
  border-radius: 999px;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
}
.lang-opt.active {
  background: linear-gradient(135deg, #0ea5e9, #2563eb);
  color: #ffffff;
  box-shadow: 0 2px 10px rgba(14, 165, 233, 0.45);
}

.ctrl-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 11px;
  border-radius: 9px;
  border: 1px solid rgba(255, 255, 255, 0.22);
  background: rgba(255, 255, 255, 0.10);
  color: #f8fafc;
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.18s;
  white-space: nowrap;
}
.ctrl-btn:hover { background: rgba(255, 255, 255, 0.20); }
.ctrl-btn.primary {
  background: linear-gradient(135deg, #0284c7, #0369a1);
  border-color: rgba(56, 189, 248, 0.45);
}

/* LAYOUT SHELL */
.shell {
  max-width: 1480px;
  margin: 0 auto;
  padding: 18px 20px 36px;
  display: grid;
  grid-template-columns: 276px minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}
.sidebar {
  position: sticky;
  top: 82px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.glass-card {
  background: var(--glass-bg);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  box-shadow: var(--glass-shadow);
  padding: 16px;
}
.nav-label {
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 10px;
}
.nav-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  text-align: start;
  padding: 10px 12px;
  margin-bottom: 6px;
  border-radius: 11px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 12.8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.16s ease;
}
.nav-btn:hover {
  background: var(--accent-soft);
  border-color: var(--glass-border);
}
.nav-btn.active {
  background: linear-gradient(135deg, #0284c7, #1d4ed8);
  color: #ffffff;
  box-shadow: 0 6px 16px rgba(2, 132, 199, 0.28);
}
.nav-pill {
  font-size: 10.5px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.22);
  color: inherit;
  white-space: nowrap;
}
.nav-btn.active .nav-pill {
  background: rgba(255, 255, 255, 0.24);
  color: #ffffff;
}

.pdf-list a {
  display: block;
  padding: 8px 10px;
  margin-bottom: 6px;
  border-radius: 9px;
  border: 1px solid var(--glass-border);
  background: var(--glass-strong);
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  transition: border-color 0.15s;
}
.pdf-list a:hover {
  border-color: var(--accent);
  color: var(--accent);
  text-decoration: none;
}

/* PANELS & TYPOGRAPHY */
main { min-width: 0; }
.tab-panel { display: none; animation: fadeIn 0.22s ease; }
.tab-panel.active { display: block; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

.section-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}
.section-head h2 {
  font-size: 19px;
  font-weight: 800;
  margin: 0;
}
.section-sub {
  font-size: 13px;
  color: var(--text-muted);
  margin: 4px 0 14px;
}

/* GRIDS */
.grid-4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 14px; }
.grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 14px; }

.kpi {
  background: var(--glass-strong);
  backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  padding: 14px;
  box-shadow: var(--glass-shadow);
  border-top: 3px solid var(--accent);
}
.kpi.danger { border-top-color: var(--red); }
.kpi.warn { border-top-color: #f59e0b; }
.kpi .lbl { font-size: 11.5px; font-weight: 700; color: var(--text-muted); margin-bottom: 4px; }
.kpi .val { font-size: 19px; font-weight: 800; font-family: var(--font-display); margin-bottom: 4px; }
.kpi .sub { font-size: 12px; color: var(--text-muted); line-height: 1.45; }

/* JUMP BANNER & ALERTS */
.jump-banner {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 14px 18px;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.90), rgba(30, 58, 138, 0.86));
  color: #f8fafc;
  border: 1px solid rgba(56, 189, 248, 0.35);
  box-shadow: var(--glass-shadow);
  margin-bottom: 16px;
}
.jump-cta {
  padding: 9px 16px;
  border-radius: 10px;
  border: none;
  background: linear-gradient(135deg, #38bdf8, #0284c7);
  color: #ffffff;
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
}
.jump-cta:hover { filter: brightness(1.08); }

.alert {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid var(--glass-border);
  backdrop-filter: blur(10px);
  margin-bottom: 16px;
  font-size: 13px;
}
.alert-red { background: var(--red-bg); border-color: var(--red-border); }
.alert-amber { background: var(--amber-bg); border-color: var(--amber-border); }
.alert-blue { background: var(--blue-bg); border-color: var(--blue-border); }
.alert-green { background: var(--green-bg); border-color: var(--green-border); }

/* BADGES & PILLS */
.badge {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
}
.b-blue { background: rgba(14, 165, 233, 0.16); color: #0369a1; }
.b-red { background: rgba(239, 68, 68, 0.16); color: #b91c1c; }
.b-amber { background: rgba(245, 158, 11, 0.18); color: #b45309; }
.b-green { background: rgba(34, 197, 94, 0.16); color: #15803d; }
.b-purple { background: rgba(139, 92, 246, 0.16); color: #6d28d9; }
[data-theme="dark"] .b-blue { color: #7dd3fc; }
[data-theme="dark"] .b-red { color: #fca5a5; }
[data-theme="dark"] .b-amber { color: #fcd34d; }
[data-theme="dark"] .b-green { color: #86efac; }
[data-theme="dark"] .b-purple { color: #c4b5fd; }

/* HERO VIDEO SUITE (TAB 2 TOP) */
.video-suite-hero {
  background: var(--glass-strong);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(56, 189, 248, 0.36);
  border-radius: 18px;
  padding: 18px;
  box-shadow: var(--glass-shadow);
  margin-bottom: 18px;
}
.video-suite-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}
.video-suite-head h2 { font-size: 18px; margin: 0; font-weight: 800; }
.video-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}
.video-card {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.video-frame-wrap {
  background: #020617;
  padding: 8px;
}
.video-frame-wrap video {
  width: 100%;
  height: 256px;
  object-fit: contain;
  border-radius: 8px;
  background: #000000;
  display: block;
}
.video-card-body {
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  flex: 1;
}
.video-card-body h4 { font-size: 14px; font-weight: 700; margin: 0; }
.video-card-body p { font-size: 12.6px; color: var(--text-muted); margin: 0; }
.video-actions {
  margin-top: auto;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--glass-border);
}
.btn-download {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 12px;
  border-radius: 8px;
  background: linear-gradient(135deg, #0284c7, #2563eb);
  color: #ffffff !important;
  font-size: 12px;
  font-weight: 700;
  text-decoration: none !important;
}
.btn-download:hover { filter: brightness(1.08); }

/* TABLES & FILTER BARS */
.filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  align-items: center;
  margin-bottom: 12px;
}
.btn-filter {
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid var(--glass-border);
  background: var(--glass-strong);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.btn-filter.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #ffffff;
}
.search-input {
  flex: 1 1 240px;
  padding: 8px 12px;
  border-radius: 10px;
  border: 1px solid var(--glass-border);
  background: var(--glass-strong);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 13px;
}
.table-wrap {
  overflow-x: auto;
  border-radius: 12px;
  border: 1px solid var(--glass-border);
  background: var(--glass-strong);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.6px;
}
th, td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--glass-border);
  text-align: start;
  vertical-align: top;
}
th {
  background: var(--table-head);
  font-weight: 700;
  font-size: 12px;
  white-space: nowrap;
}
tr:last-child td { border-bottom: none; }

/* TIMELINE */
.timeline-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.tl-item {
  display: grid;
  grid-template-columns: 155px minmax(0, 1fr);
  gap: 12px;
  padding: 11px 14px;
  border-radius: 12px;
  border: 1px solid var(--glass-border);
  background: var(--glass-strong);
}
.tl-date {
  font-weight: 800;
  color: var(--accent);
  font-size: 12.5px;
  font-family: 'Cascadia Code', Consolas, monospace;
  direction: ltr;
  text-align: start;
}

/* ROI GALLERY & MODAL */
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}
.roi-card {
  background: var(--glass-strong);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.roi-img-box {
  height: 196px;
  background: #0f172a;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: zoom-in;
  overflow: hidden;
}
.roi-img-box img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.25s ease;
}
.roi-card:hover .roi-img-box img { transform: scale(1.03); }
.roi-body { padding: 12px; display: flex; flex-direction: column; gap: 6px; flex: 1; }
.roi-body h4 { font-size: 13.5px; font-weight: 700; margin: 0; }
.roi-body p { font-size: 12px; color: var(--text-muted); margin: 0; flex: 1; }

.modal {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: rgba(2, 6, 23, 0.80);
  backdrop-filter: blur(8px);
}
.modal.open { display: flex; }
.modal-box {
  width: min(1160px, 96vw);
  max-height: 92vh;
  background: var(--glass-strong);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.modal-img-wrap {
  flex: 1;
  min-height: 300px;
  max-height: 70vh;
  overflow: auto;
  background: #020617;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.modal-img-wrap img {
  max-width: 100%;
  max-height: 68vh;
  transform-origin: center center;
  transition: transform 0.18s ease;
}

/* SVG CONTAINER */
.svg-wrap { width: 100%; overflow-x: auto; }
.svg-wrap svg { width: 100%; height: auto; display: block; }

/* CONSULTANT & RANKED CARDS */
.consultant-card {
  background: var(--glass-strong);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 18px;
  box-shadow: var(--glass-shadow);
  margin-bottom: 14px;
}
.verdict-box {
  padding: 10px 14px;
  border-radius: 10px;
  background: var(--accent-soft);
  border: 1px solid var(--glass-border);
  margin: 10px 0 12px;
  font-size: 13px;
}
.rank-card {
  display: grid;
  grid-template-columns: 54px minmax(0, 1fr);
  gap: 12px;
  align-items: start;
  padding: 14px;
  border-radius: 14px;
  border: 1px solid var(--glass-border);
  background: var(--glass-strong);
  margin-bottom: 10px;
}
.rank-num {
  height: 44px;
  border-radius: 12px;
  background: linear-gradient(135deg, #0284c7, #1d4ed8);
  color: #ffffff;
  font-weight: 800;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* CHECKLIST */
.check-row {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 9px 10px;
  border-bottom: 1px solid var(--glass-border);
  cursor: pointer;
}
.check-row:last-child { border-bottom: none; }
.check-row input { margin-top: 3px; width: 16px; height: 16px; accent-color: var(--accent); }
.check-done { text-decoration: line-through; color: var(--text-muted); }

@media (max-width: 1120px) {
  .shell { grid-template-columns: 1fr; }
  .sidebar { position: static; }
  .video-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 780px) {
  .grid-4, .grid-3, .grid-2, .video-grid, .gallery-grid { grid-template-columns: 1fr; }
  .tl-item { grid-template-columns: 1fr; gap: 4px; }
  .glass-header { padding: 10px 12px; }
  .shell { padding: 12px; }
}
@media print {
  .no-print, .sidebar, .header-controls { display: none !important; }
  .shell { display: block; padding: 0; }
  body.print-all .tab-panel { display: block !important; page-break-after: always; }
}

/* 7-Consultant 3-Block, Historical Warning, Pending & Q&A Upgrades */
.hist-warn-card {
  border: 2px solid #f59e0b !important;
  box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.25), var(--glass-shadow);
  background: linear-gradient(180deg, rgba(245, 158, 11, 0.09), var(--glass-bg) 38%);
}
.hist-banner {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 800;
  padding: 3px 9px;
  border-radius: 999px;
  background: rgba(245, 158, 11, 0.16);
  border: 1px solid #d97706;
  color: #b45309;
  margin-bottom: 6px;
}
[data-theme="dark"] .hist-banner {
  color: #fcd34d;
  background: rgba(245, 158, 11, 0.22);
}
.curr-banner {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 10.5px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(16, 185, 129, 0.14);
  border: 1px solid rgba(16, 185, 129, 0.45);
  color: var(--green);
  margin-bottom: 6px;
}
.pending-box {
  border: 2px dashed var(--accent);
  background: var(--blue-bg);
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 18px;
}
.pending-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.pending-item {
  background: var(--glass-strong);
  border: 1px solid var(--glass-border);
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 12.6px;
}
.qa-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.qa-card {
  border-radius: 14px;
  padding: 14px 16px;
  border: 1px solid var(--glass-border);
  background: var(--glass-strong);
  box-shadow: var(--glass-shadow);
}
.qa-blue { border-top: 4px solid #0284c7; }
.qa-red { border-top: 4px solid #dc2626; }
.qa-amber { border-top: 4px solid #d97706; }
.qa-card ul { margin: 8px 0 0; padding-inline-start: 18px; font-size: 12.6px; line-height: 1.55; }
.qa-card li { margin-bottom: 6px; }
.c-block {
  border-radius: 10px;
  padding: 10px 12px;
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.58;
  border: 1px solid var(--glass-border);
}
.c-block-conc { background: var(--blue-bg); border-inline-start: 4px solid #0284c7; }
.c-block-imm { background: var(--red-bg); border-inline-start: 4px solid #dc2626; }
.c-block-plan { background: var(--glass-bg); border-inline-start: 4px solid #16a34a; }
.c-block-lbl { display: block; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.4px; font-weight: 800; margin-bottom: 4px; color: var(--text); }
.c-block-plan ul { margin: 4px 0 0; padding-inline-start: 18px; }
.c-block-plan li { margin-bottom: 5px; }
.pdf-dl-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 10px;
}
.pdf-dl-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 8px;
  padding: 14px;
  border-radius: 12px;
  background: var(--glass-strong);
  border: 1px solid var(--glass-border);
  text-decoration: none;
  color: var(--text);
  transition: transform .15s ease, border-color .15s ease;
}
.pdf-dl-card:hover { transform: translateY(-2px); border-color: var(--accent); }
.tr-hist { background: rgba(245, 158, 11, 0.08); }
@media (max-width: 980px) {
  .pending-grid, .qa-grid, .pdf-dl-grid { grid-template-columns: 1fr; }
}

</style>
</head>
<body>

<header class="glass-header">
  <div class="header-inner">
    <div class="brand-box">
      <h1 id="brandTitle">MDT Clinical Case Study | Complex Neuro-Vascular &amp; Otological Evaluation</h1>
      <div class="brand-sub" id="brandSub"></div>
    </div>
    <div class="header-controls no-print">
      <div class="lang-pill" id="langSwitch" role="group" aria-label="Language switcher">
        <button type="button" class="lang-opt active" data-lang="en" onclick="setLang('en')">EN</button>
        <button type="button" class="lang-opt" data-lang="ar" onclick="setLang('ar')">عربي مصري</button>
      </div>
      <button type="button" class="ctrl-btn" id="btnTheme" onclick="toggleTheme()">🌙 <span id="lblTheme">Dark Mode</span></button>
      <button type="button" class="ctrl-btn" onclick="printSection()">🖨️ <span id="lblPrintTab">Print Tab</span></button>
      <button type="button" class="ctrl-btn primary" onclick="printFullDossier()">📄 <span id="lblPrintAll">Full Dossier (PDF)</span></button>
    </div>
  </div>
</header>

<div class="shell">
  <aside class="sidebar no-print">
    <nav class="glass-card" aria-label="Case study sections">
      <div class="nav-label" id="lblNavHeader">Case-Study Journey</div>
      <div id="tabNavHost"></div>
    </nav>
    <div class="glass-card pdf-list">
      <div class="nav-label" id="lblPdfHeader">Shareable Reports &amp; Index</div>
      <div id="pdfLinksHost"></div>
    </div>
  </aside>

  <main>
    <!-- TAB 1: 01. PATIENT PROFILE & CLINICAL HISTORY -->
    <section id="tab-profile" class="tab-panel active"></section>

    <!-- TAB 2: 02. NEUROIMAGING VIDEO SUITE (MRI / MRA / MRV) -->
    <section id="tab-videos" class="tab-panel">
      <section id="neuroimaging-video-suite" class="video-suite-hero" aria-label="Neuroimaging Video Suite">
        <div class="video-suite-head">
          <h2 id="videoSuiteHeading">Neuroimaging Video Suite (Interactive DICOM / MRI / MRV Viewer)</h2>
          <div id="videoBadgesHost" style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;"></div>
        </div>
        <div class="video-grid" id="videoGridHost"></div>
      </section>
      <div id="videoBodyHost"></div>
    </section>

    <!-- TAB 3: 03. DIAGNOSTICS, LABS & IMAGING GALLERY -->
    <section id="tab-diagnostics" class="tab-panel"></section>

    <!-- TAB 4: 04. PRIOR CONSULTANTS & CLINICAL CONFLICT -->
    <section id="tab-prior" class="tab-panel"></section>

    <!-- TAB 5: 05. 7-CONSULTANT MDT BOARD (CONCLUSIONS & ACTION PLANS) -->
    <section id="tab-board" class="tab-panel"></section>

    <!-- TAB 6: 06. MASTER TREATMENT & MEDICATION PLAN -->
    <section id="tab-plan" class="tab-panel"></section>
  </main>
</div>

<div class="modal" id="imgModal" onclick="if(event.target===this) closeModal()">
  <div class="modal-box">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">
      <strong id="modalTitle" style="font-size:15px;">Image Inspector</strong>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <button type="button" class="btn-filter" id="mZ1" onclick="setZoom(1)">100%</button>
        <button type="button" class="btn-filter" id="mZ2" onclick="setZoom(1.6)">160%</button>
        <button type="button" class="btn-filter" id="mZ3" onclick="setZoom(2.4)">240%</button>
        <a class="btn-download" id="modalRawLink" target="_blank">Open Original</a>
        <button type="button" class="btn-filter" id="mClose" onclick="closeModal()">Close</button>
      </div>
    </div>
    <div id="modalHistWrap"></div>
    <div class="modal-img-wrap"><img id="modalImg" src="" alt="Plate Preview"></div>
    <div id="modalCaption" style="font-size:12.6px;color:var(--text-muted);"></div>
  </div>
</div>

<script>
const MDT_DATA = __DATA_JSON__;

let currentLang = "en";
let activeTabId = "tab-profile";
let activeBoardFilter = 0; // 0 = all 7, or 1..7
let activeGalleryFilter = "ALL";
let activeFileFilter = "ALL";
let activeMedFilter = "ALL";
let videoCardsBuilt = false;

function getT() { return MDT_DATA.i18n[currentLang]; }

function setLang(lang) {
  currentLang = (lang === "ar") ? "ar" : "en";
  localStorage.setItem("mdt_lang_strict_v4", currentLang);
  const root = document.documentElement;
  root.setAttribute("lang", currentLang);
  root.setAttribute("dir", currentLang === "ar" ? "rtl" : "ltr");
  document.querySelectorAll(".lang-opt").forEach(b => {
    b.classList.toggle("active", b.getAttribute("data-lang") === currentLang);
  });
  videoCardsBuilt = false;
  renderAll();
}

function switchTab(id) {
  activeTabId = id;
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  const el = document.getElementById(id);
  if (el) el.classList.add("active");
  if (id === "tab-videos") {
    document.querySelectorAll("#videoGridHost video").forEach(v => {
      if (v.readyState === 0 && v.networkState === 0) { v.load(); }
    });
  }
  renderSidebarNav();
}

function jumpToVideoSuite() {
  switchTab("tab-videos");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function toggleTheme() {
  const root = document.documentElement;
  const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  localStorage.setItem("mdt_theme_v4", next);
  renderHeaderLabels();
}

function renderHeaderLabels() {
  const t = getT();
  document.title = t.pageTitle;
  document.getElementById("brandTitle").textContent = t.brandTitle;
  document.getElementById("brandSub").textContent = t.brandSub;
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  document.getElementById("lblTheme").textContent = isDark ? t.btnThemeLight : t.btnThemeDark;
  document.getElementById("lblPrintTab").textContent = t.btnPrintTab;
  document.getElementById("lblPrintAll").textContent = t.btnPrintAll;
  document.getElementById("lblNavHeader").textContent = t.navLabel;
  document.getElementById("lblPdfHeader").textContent = t.pdfLabel;
  document.getElementById("pdfLinksHost").innerHTML = t.pdfLinks.map(p =>
    `<a href="${p.href}" target="_blank" rel="noopener">${p.txt}</a>`
  ).join("");
}

function renderSidebarNav() {
  const t = getT();
  document.getElementById("tabNavHost").innerHTML = t.tabs.map(tb => `
    <button type="button" class="nav-btn ${tb.id === activeTabId ? "active" : ""}" onclick="switchTab('${tb.id}')">
      <span>${tb.label}</span>
      <span class="nav-pill">${tb.count}</span>
    </button>
  `).join("");
}

function renderTab1() {
  const t = getT();
  const c = t.t1;
  document.getElementById("tab-profile").innerHTML = `
    <div class="jump-banner">
      <div>
        <strong style="font-size:15px;">&#127909; ${c.jumpTitle}</strong>
        <div style="font-size:12.6px;color:var(--text-muted);margin-top:3px;">${c.jumpSub}</div>
      </div>
      <button type="button" class="jump-cta" onclick="jumpToVideoSuite()">${c.jumpBtn} &rarr;</button>
    </div>

    <h2 class="section-head">${c.secHead}</h2>
    <div class="grid-4" style="margin-bottom:14px;">
      ${c.kpis.map(k => `
        <div class="kpi ${k.cls}">
          <div class="lbl">${k.lbl}</div>
          <div class="val">${k.val}</div>
          <div class="sub">${k.sub}</div>
        </div>
      `).join("")}
    </div>

    <div class="alert alert-red" style="margin-bottom:16px;">
      <strong style="font-size:14.2px;">&#9888;&#65039; ${c.safetyTitle}</strong>
      <ul style="margin:8px 0 0;padding-inline-start:20px;font-size:13px;line-height:1.58;">
        ${c.safetyBullets.map(b => `<li style="margin-bottom:5px;">${b}</li>`).join("")}
      </ul>
    </div>

    <h3 class="section-head">${c.compHead}</h3>
    <div class="grid-2" style="margin-bottom:18px;">
      ${c.complaints.map(cp => `
        <div class="glass-card">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;">
            <strong style="font-size:14.5px;">${cp.title}</strong>
            <span class="badge ${cp.badge}">${cp.tag}</span>
          </div>
          <p style="margin:0;font-size:13px;color:var(--text-muted);line-height:1.55;">${cp.desc}</p>
        </div>
      `).join("")}
    </div>

    <h3 class="section-head">${c.timelineHead}</h3>
    <div class="glass-card" style="margin-bottom:18px;">
      <div class="timeline-list">
        ${c.timeline.map(ev => `
          <div class="tl-item ${ev.hist ? "hist-warn-card" : ""}" style="padding:10px 12px;border-radius:10px;">
            <div>
              ${ev.hist ? `<div class="hist-banner">&#9888;&#65039; ${t.histShortBadge}</div><br>` : `<div class="curr-banner">&#9989; ${t.currBadge}</div><br>`}
              <span class="tl-date">${ev.date}</span>
            </div>
            <div>
              <strong style="font-size:13.2px;">${ev.src}</strong>
              <p style="margin:4px 0 0;font-size:12.8px;color:var(--text-muted);">${ev.detail}</p>
            </div>
          </div>
        `).join("")}
      </div>
    </div>

    <h3 class="section-head">${c.medsHead}</h3>
    <div class="table-wrap glass-card">
      <table>
        <thead><tr>${c.medCols.map(col => `<th>${col}</th>`).join("")}</tr></thead>
        <tbody>
          ${c.currentMeds.map(m => `
            <tr>
              <td><strong>${m.drug}</strong></td>
              <td>${m.dose}</td>
              <td>${m.src}</td>
              <td>${m.role}</td>
              <td><span class="badge ${m.badge}">${m.tag}</span></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function buildOrUpdateVideoSuite() {
  const t2 = getT().t2;
  document.getElementById("videoSuiteHeading").innerHTML = "&#127909; " + t2.videoSuiteTitle;
  document.getElementById("videoBadgesHost").innerHTML = t2.videoBadges.map((b, i) =>
    `<span class="badge ${i === 0 ? "b-blue" : "b-green"}">${b}</span>`
  ).join("");

  if (!videoCardsBuilt) {
    document.getElementById("videoGridHost").innerHTML = t2.videos.map((v, idx) => `
      <div class="video-card">
        <div class="video-frame-wrap">
          <video id="neuroVid${idx}" controls preload="metadata" playsinline>
            <source src="${v.src}" type="video/mp4">
          </video>
        </div>
        <div class="video-card-body">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
            <strong style="font-size:13.8px;">${v.title}</strong>
            <span class="badge ${v.badgeCls}">${v.badge}</span>
          </div>
          <p style="margin:0;font-size:12.4px;color:var(--text-muted);line-height:1.5;">${v.desc}</p>
          <div class="video-actions">
            <a href="${v.src}" download class="btn-download" style="flex:1;justify-content:center;">&#11015;&#65039; ${v.dlText}</a>
            <a href="${v.secHref}" ${v.secDownload ? "download" : 'target="_blank"'} class="btn-filter">${v.secText}</a>
          </div>
        </div>
      </div>
    `).join("");
    videoCardsBuilt = true;
  }
}

function renderTab2Videos() {
  const t2 = getT().t2;
  buildOrUpdateVideoSuite();
  document.getElementById("videoBodyHost").innerHTML = `
    <h3 class="section-head" style="margin-top:16px;">${t2.seqHead}</h3>
    <div class="grid-3" style="margin-bottom:18px;">
      ${t2.seqCards.map(s => `
        <div class="glass-card">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;">
            <strong style="font-size:13.8px;">${s.title}</strong>
            <span class="badge ${s.badge}">${s.tag}</span>
          </div>
          <p style="margin:0;font-size:12.8px;color:var(--text-muted);line-height:1.55;">${s.desc}</p>
        </div>
      `).join("")}
    </div>

    <div class="jump-banner">
      <div>
        <strong style="font-size:15px;">&#128300; ${t2.jumpDiagTitle}</strong>
        <div style="font-size:12.6px;color:var(--text-muted);margin-top:3px;">${t2.jumpDiagSub}</div>
      </div>
      <button type="button" class="jump-cta" onclick="switchTab('tab-diagnostics'); window.scrollTo({top:0,behavior:'smooth'});">${t2.jumpDiagBtn} &rarr;</button>
    </div>
  `;
}

function renderHolterSVG() {
  const t3 = getT().t3;
  const H = MDT_DATA.holter_hours;
  const W = 740, Ht = 278, padL = 42, padR = 42, padT = 32, padB = 46;
  const cw = W - padL - padR, ch = Ht - padT - padB;
  const barW = cw / H.length;

  let bars = "", hrPts = "", xTicks = "";
  H.forEach((d, i) => {
    const x = padL + i * barW;
    const bh = (d.pvc / 180) * ch;
    const y = padT + ch - bh;
    const isSurge = (d.h === "06:00" || d.h === "07:00" || d.h === "08:00");
    const isST = (d.h === "13:00");
    const fill = isSurge ? "#dc2626" : (isST ? "#d97706" : "#0284c7");
    bars += `<rect x="${(x + 2).toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(4, barW - 4).toFixed(1)}" height="${bh.toFixed(1)}" rx="2.5" fill="${fill}" opacity="0.88"><title>${d.h}: ${d.pvc} PVCs | HR ${d.hr} | ST ${d.st}</title></rect>`;
    if (d.pvc >= 45 || isSurge) {
      bars += `<text x="${(x + barW / 2).toFixed(1)}" y="${(y - 4).toFixed(1)}" text-anchor="middle" font-size="9.5" font-weight="700" fill="${fill}">${d.pvc}</text>`;
    }
    const hy = padT + ch - ((d.hr - 40) / 80) * ch;
    hrPts += `${(x + barW / 2).toFixed(1)},${hy.toFixed(1)} `;
    if (i % 2 === 0 || isSurge || isST) {
      xTicks += `<text x="${(x + barW / 2).toFixed(1)}" y="${Ht - 24}" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.78">${d.h.slice(0, 2)}</text>`;
    }
  });

  const surgeX = padL + 21 * barW;
  const stX = padL + 4 * barW + barW / 2;

  document.getElementById("holterSvgWrap").innerHTML = `
  <svg viewBox="0 0 ${W} ${Ht}" dir="ltr" role="img" aria-label="${t3.holterChartTitle}">
    <rect x="${surgeX.toFixed(1)}" y="${padT}" width="${(3 * barW).toFixed(1)}" height="${ch}" fill="#dc2626" opacity="0.10" rx="4"/>
    <text x="${(surgeX + 1.5 * barW).toFixed(1)}" y="${padT + 12}" text-anchor="middle" font-size="9.5" font-weight="700" fill="#dc2626">${t3.holterSurgeLbl}</text>
    <line x1="${stX.toFixed(1)}" y1="${padT + 18}" x2="${stX.toFixed(1)}" y2="${padT + ch}" stroke="#d97706" stroke-dasharray="3,3" stroke-width="1.5"/>
    <text x="${stX.toFixed(1)}" y="${padT + 14}" text-anchor="middle" font-size="9" font-weight="700" fill="#b45309">${t3.holterStLbl}</text>
    <line x1="${padL}" y1="${padT + ch}" x2="${W - padR}" y2="${padT + ch}" stroke="currentColor" opacity="0.3"/>
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + ch}" stroke="currentColor" opacity="0.3"/>
    ${[0, 60, 120, 180].map(v => {
      const y = padT + ch - (v / 180) * ch;
      return `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="currentColor" opacity="0.08"/>
              <text x="${padL - 5}" y="${y + 3}" text-anchor="end" font-size="9.5" fill="currentColor" opacity="0.72">${v}</text>`;
    }).join("")}
    ${bars}
    <polyline fill="none" stroke="#16a34a" stroke-width="2.3" points="${hrPts}"/>
    ${xTicks}
    <rect x="${padL}" y="${Ht - 14}" width="10" height="10" rx="2" fill="#0284c7"/>
    <text x="${padL + 14}" y="${Ht - 5}" font-size="10" fill="currentColor">${t3.holterLegend1}</text>
    <line x1="${W - 295}" y1="${Ht - 9}" x2="${W - 275}" y2="${Ht - 9}" stroke="#16a34a" stroke-width="2.5"/>
    <text x="${W - 270}" y="${Ht - 5}" font-size="10" fill="currentColor">${t3.holterLegend2}</text>
  </svg>
  <div style="font-size:11.5px;color:var(--text-muted);margin-top:4px;text-align:center;">${t3.holterFoot}</div>`;
}

function renderAudioSVG() {
  const t3 = getT().t3;
  const freqs = [250, 500, 1000, 2000, 4000, 8000];
  const rt = [20, 25, 45, 50, 50, 70];
  const lt = [50, 50, 55, 57.5, 70, 80];
  const W = 740, Ht = 278, padL = 48, ptaW = 395, padT = 32, padB = 46;
  const ch = Ht - padT - padB;
  const xPos = i => padL + 30 + (i / (freqs.length - 1)) * (ptaW - 50);
  const yPos = db => padT + ((db + 10) / 110) * ch;

  let rtPoly = "", ltPoly = "", markers = "", grid = "";
  [0, 20, 40, 60, 80, 100].forEach(db => {
    const y = yPos(db);
    grid += `<line x1="${padL}" y1="${y}" x2="${padL + ptaW}" y2="${y}" stroke="currentColor" opacity="0.1"/>
             <text x="${padL - 6}" y="${y + 3}" text-anchor="end" font-size="9.5" fill="currentColor" opacity="0.72">${db}</text>`;
  });
  freqs.forEach((f, i) => {
    const x = xPos(i);
    grid += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + ch}" stroke="currentColor" opacity="0.08"/>
             <text x="${x}" y="${padT + ch + 14}" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.8">${f >= 1000 ? (f / 1000) + "k" : f}</text>`;
    rtPoly += `${x},${yPos(rt[i])} `;
    ltPoly += `${x},${yPos(lt[i])} `;
    markers += `<circle cx="${x}" cy="${yPos(rt[i])}" r="4.5" fill="none" stroke="#dc2626" stroke-width="2.2"/>
                <rect x="${x - 3.5}" y="${yPos(rt[i]) - 3.5}" width="7" height="7" fill="none" stroke="#dc2626" stroke-width="1.3"/>
                <text x="${x}" y="${yPos(rt[i]) - 7}" text-anchor="middle" font-size="9" fill="#dc2626" font-weight="700">${rt[i]}</text>`;
    markers += `<line x1="${x - 4}" y1="${yPos(lt[i]) - 4}" x2="${x + 4}" y2="${yPos(lt[i]) + 4}" stroke="#2563eb" stroke-width="2.2"/>
                <line x1="${x - 4}" y1="${yPos(lt[i]) + 4}" x2="${x + 4}" y2="${yPos(lt[i]) - 4}" stroke="#2563eb" stroke-width="2.2"/>
                <text x="${x}" y="${yPos(lt[i]) + 13}" text-anchor="middle" font-size="9" fill="#2563eb" font-weight="700">${lt[i]}</text>`;
  });

  const bx = 482, bw = 238;
  const maxC = 3.5;
  const tympBars = [
    { lbl: "24 Rt (As)", val: 0.3, col: "#f87171" },
    { lbl: "26 Rt (A)", val: 1.5, col: "#dc2626" },
    { lbl: "24 Lt (Ad)", val: 1.7, col: "#60a5fa" },
    { lbl: "26 Lt (Ad!)", val: 3.2, col: "#2563eb" },
  ];
  const barStep = bw / 4;
  const ceilY = padT + ch - (3.0 / maxC) * ch;

  document.getElementById("audioSvgWrap").innerHTML = `
  <svg viewBox="0 0 ${W} ${Ht}" dir="ltr" role="img" aria-label="${t3.audioChartTitle}">
    ${grid}
    <rect x="${padL}" y="${padT}" width="${ptaW}" height="${ch}" fill="none" stroke="currentColor" opacity="0.24"/>
    <polyline fill="none" stroke="#dc2626" stroke-width="2.3" points="${rtPoly}"/>
    <polyline fill="none" stroke="#2563eb" stroke-width="2.3" points="${ltPoly}"/>
    ${markers}
    <text x="${padL + 10}" y="${padT + 14}" font-size="10" font-weight="700" fill="#dc2626">&#9679; ${t3.audioRtLbl}</text>
    <text x="${padL + 10}" y="${padT + 27}" font-size="10" font-weight="700" fill="#2563eb">&#10005; ${t3.audioLtLbl}</text>

    <rect x="${bx}" y="${padT}" width="${bw}" height="${ch}" fill="currentColor" fill-opacity="0.03" stroke="currentColor" stroke-opacity="0.22" rx="6"/>
    <text x="${bx + bw / 2}" y="${padT - 14}" text-anchor="middle" font-size="11" font-weight="700" fill="currentColor">${t3.audioTympHead}</text>
    <text x="${bx + bw / 2}" y="${padT - 3}" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.72">${t3.audioTympSub}</text>
    <line x1="${bx}" y1="${ceilY}" x2="${bx + bw}" y2="${ceilY}" stroke="#d97706" stroke-dasharray="4,3" stroke-width="1.3"/>
    <text x="${bx + 6}" y="${ceilY - 3}" font-size="8.8" font-weight="700" fill="#d97706">${t3.audioLimitLbl}</text>
    ${tympBars.map((tb, idx) => {
      const h = (tb.val / maxC) * ch;
      const x = bx + idx * barStep + 10;
      const y = padT + ch - h;
      return `<rect x="${x}" y="${y}" width="${barStep - 20}" height="${h}" rx="3" fill="${tb.col}"/>
              <text x="${x + (barStep - 20) / 2}" y="${y - 4}" text-anchor="middle" font-size="10" font-weight="800" fill="${tb.col}">${tb.val}</text>
              <text x="${x + (barStep - 20) / 2}" y="${padT + ch + 13}" text-anchor="middle" font-size="8.8" fill="currentColor">${tb.lbl}</text>`;
    }).join("")}
    <text x="${bx + bw / 2}" y="${Ht - 6}" text-anchor="middle" font-size="9.8" font-weight="700" fill="#16a34a">${t3.audioRefute}</text>
  </svg>`;
}

function setGalleryFilter(grp) {
  activeGalleryFilter = grp;
  renderGallerySection();
}

function renderGallerySection() {
  const t = getT();
  const t3 = t.t3;
  const all = MDT_DATA.roi_cards[currentLang].map((r, i) => ({ ...r, _idx: i }));
  const filtered = (activeGalleryFilter === "ALL") ? all : all.filter(r => r.grp === activeGalleryFilter);

  document.getElementById("galleryFilterHost").innerHTML = t3.galFilters.map(f =>
    `<button type="button" class="btn-filter ${activeGalleryFilter === f.grp ? "active" : ""}" onclick="setGalleryFilter('${f.grp}')">${f.lbl}</button>`
  ).join("");

  document.getElementById("roiGallery").innerHTML = filtered.map(r => `
    <div class="roi-card ${r.historical ? "hist-warn-card" : ""}">
      <div class="roi-img-box" onclick="openModal(${r._idx})">
        <img loading="lazy" src="${r.img}" alt="${r.title}">
      </div>
      <div class="roi-body">
        ${r.historical ? `<span class="hist-banner">&#9888;&#65039; ${t.histShortBadge}</span>` : `<span class="curr-banner">&#9989; ${t.currBadge}</span>`}
        <h4>${r.title}</h4>
        <p>${r.desc}</p>
        <button type="button" class="btn-filter" style="width:100%;" onclick="openModal(${r._idx})">&#128269; ${t3.roiBtn}</button>
      </div>
    </div>
  `).join("");
}

function setFileFilter(grp) {
  activeFileFilter = grp;
  renderFileFiltersHost();
  renderFileTable();
}

function renderFileFiltersHost() {
  const t3 = getT().t3;
  document.getElementById("fileFiltersHost").innerHTML = t3.fileFilters.map(f =>
    `<button type="button" class="btn-filter ${activeFileFilter === f.grp ? "active" : ""}" onclick="setFileFilter('${f.grp}')">${f.lbl}</button>`
  ).join("");
}

function renderFileTable() {
  const t = getT();
  const q = (document.getElementById("fileSearchInput")?.value || "").toLowerCase().trim();
  const rows = MDT_DATA.files_index.map((f, idx) => ({ ...f, num: idx + 1 })).filter(f => {
    if (activeFileFilter !== "ALL" && f.grp !== activeFileFilter) return false;
    if (!q) return true;
    return [f.cat, f.date, f.raw_en, f.raw_ar, f.newName, f.summary_en, f.summary_ar].join(" ").toLowerCase().includes(q);
  });
  document.getElementById("fileTableBody").innerHTML = rows.map(f => {
    const raw = (currentLang === "ar") ? f.raw_ar : f.raw_en;
    const sum = (currentLang === "ar") ? f.summary_ar : f.summary_en;
    const sz = (currentLang === "ar") ? f.size_ar : f.size_en;
    return `
      <tr class="${f.is_historical ? "tr-hist" : ""}">
        <td>${f.num}</td>
        <td><span class="badge ${f.is_historical ? "b-amber" : "b-blue"}">${f.cat}</span></td>
        <td style="white-space:nowrap;">
          <div>${f.date}</div>
          ${f.is_historical ? `<span class="hist-banner" style="margin-top:2px;">&#9888;&#65039; ${t.histShortBadge}</span>` : ""}
        </td>
        <td style="font-size:12px;color:var(--text-muted);">${raw}</td>
        <td><a href="${f.href}" target="_blank" rel="noopener" style="font-weight:700;color:var(--accent);text-decoration:none;word-break:break-all;">${f.newName}</a></td>
        <td style="white-space:nowrap;">${sz}</td>
        <td style="font-size:12.4px;">${sum}</td>
      </tr>
    `;
  }).join("");
}

function renderTab3Diagnostics() {
  const t = getT();
  const t3 = t.t3;
  document.getElementById("tab-diagnostics").innerHTML = `
    <div id="pendingWorkup" class="pending-box">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <strong style="font-size:14.5px;color:var(--accent-dark);">&#9203; ${t3.pendingHead}</strong>
      </div>
      <div class="pending-grid">
        ${t3.pendingItems.map(p => `
          <div class="pending-item">
            <strong style="display:block;margin-bottom:4px;font-size:13px;">${p.title}</strong>
            <div style="color:var(--text-muted);line-height:1.5;">${p.txt}</div>
          </div>
        `).join("")}
      </div>
    </div>

    <h2 class="section-head">${t3.diagHead}</h2>
    <div class="grid-2" style="margin-bottom:18px;">
      ${t3.investigations.map(inv => `
        <div class="glass-card ${inv.hist ? "hist-warn-card" : ""}">
          ${inv.hist ? `<div class="hist-banner">&#9888;&#65039; ${t.histShortBadge}</div>` : `<div class="curr-banner">&#9989; ${t.currBadge}</div>`}
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;">
            <strong style="font-size:14.2px;">${inv.title}</strong>
            <span class="badge ${inv.badge}">${inv.status}</span>
          </div>
          <ul style="margin:0;padding-inline-start:18px;font-size:12.8px;color:var(--text-muted);line-height:1.55;">
            ${inv.metrics.map(m => `<li style="margin-bottom:4px;">${m}</li>`).join("")}
          </ul>
        </div>
      `).join("")}
    </div>

    <h3 class="section-head">${t3.chartsHead}</h3>
    <div class="grid-2" style="margin-bottom:18px;">
      <div class="glass-card">
        <strong style="display:block;margin-bottom:8px;font-size:13.8px;">${t3.holterChartTitle}</strong>
        <div id="holterSvgWrap" class="svg-wrap"></div>
      </div>
      <div class="glass-card">
        <strong style="display:block;margin-bottom:8px;font-size:13.8px;">${t3.audioChartTitle}</strong>
        <div id="audioSvgWrap" class="svg-wrap"></div>
      </div>
    </div>

    <h3 class="section-head">${t3.roiHead}</h3>
    <div id="galleryFilterHost" class="filter-bar" style="margin-bottom:12px;"></div>
    <div id="roiGallery" class="gallery-grid" style="margin-bottom:20px;"></div>

    <h3 class="section-head">${t3.filesHead}</h3>
    <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px;">
      <input type="text" id="fileSearchInput" class="search-input" style="flex:1;min-width:250px;" placeholder="${t3.searchPh}" oninput="renderFileTable()">
    </div>
    <div id="fileFiltersHost" class="filter-bar" style="margin-bottom:10px;"></div>
    <div class="table-wrap glass-card">
      <table>
        <thead><tr>${t3.fileCols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
        <tbody id="fileTableBody"></tbody>
      </table>
    </div>
    <div style="font-size:12.2px;color:var(--text-muted);margin-top:8px;">${t3.fileFoot}</div>
  `;
  renderHolterSVG();
  renderAudioSVG();
  renderGallerySection();
  renderFileFiltersHost();
  renderFileTable();
}

function renderTab4Prior() {
  const t4 = getT().t4;
  document.getElementById("tab-prior").innerHTML = `
    <h2 class="section-head">${t4.head}</h2>
    <p class="section-sub">${t4.sub}</p>
    <div class="grid-2" style="margin-bottom:18px;">
      ${t4.priorCards.map(p => `
        <div class="glass-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px;">
            <div>
              <strong style="font-size:14.6px;display:block;">${p.role}</strong>
              <span style="font-size:12px;color:var(--text-muted);">${p.doctor}</span>
            </div>
            <span class="badge ${p.badge}">${p.tag}</span>
          </div>
          <div class="verdict-box" style="margin:8px 0;">${p.stance}</div>
          <ul style="margin:0;padding-inline-start:18px;font-size:12.8px;color:var(--text-muted);line-height:1.55;">
            ${p.points.map(pt => `<li style="margin-bottom:5px;">${pt}</li>`).join("")}
          </ul>
        </div>
      `).join("")}
    </div>

    <h3 class="section-head">${t4.bridgeTitle}</h3>
    <div class="grid-2">
      ${t4.bridgeCards.map(b => `
        <div class="alert ${b.cls}">
          <strong style="display:block;font-size:13.8px;margin-bottom:5px;">${b.title}</strong>
          <div style="font-size:12.8px;line-height:1.55;">${b.txt}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function setBoardFilter(id) {
  activeBoardFilter = id;
  renderTab5Board();
}

function renderTab5Board() {
  const t5 = getT().t5;
  const list = MDT_DATA.consultants[currentLang];
  const dilemmas = MDT_DATA.dilemmas[currentLang];
  const shown = (activeBoardFilter === 0) ? list : list.filter(c => c.id === activeBoardFilter);

  document.getElementById("tab-board").innerHTML = `
    <h2 class="section-head">${t5.head}</h2>
    <p class="section-sub">${t5.sub}</p>

    <h3 class="section-head">${t5.qaHead}</h3>
    <div class="qa-grid">
      ${t5.qaCards.map(q => `
        <div class="qa-card ${q.cls}">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:6px;">
            <strong style="font-size:14px;">${q.title}</strong>
            <span class="badge ${q.badge}">${q.tag}</span>
          </div>
          <ul>${q.bullets.map(b => `<li>${b}</li>`).join("")}</ul>
        </div>
      `).join("")}
    </div>

    <div class="filter-bar" style="margin-bottom:14px;">
      <button type="button" class="btn-filter ${activeBoardFilter === 0 ? "active" : ""}" onclick="setBoardFilter(0)">${t5.allBtn}</button>
      ${list.map(c => `<button type="button" class="btn-filter ${activeBoardFilter === c.id ? "active" : ""}" onclick="setBoardFilter(${c.id})">#${c.id} ${c.role.split("—")[0].trim()}</button>`).join("")}
    </div>

    <div style="display:flex;flex-direction:column;gap:14px;margin-bottom:20px;">
      ${shown.map(c => `
        <article class="consultant-card">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
            <strong style="font-size:15.2px;">${c.role}</strong>
            <span class="badge ${c.badge}">#${c.id} / 7</span>
          </div>
          <div class="c-block c-block-conc">
            <span class="c-block-lbl">${t5.blkConcLbl}</span>
            <div>${c.conclusion}</div>
          </div>
          <div class="c-block c-block-imm">
            <span class="c-block-lbl">${t5.blkImmLbl}</span>
            <div>${c.immediate_action}</div>
          </div>
          <div class="c-block c-block-plan">
            <span class="c-block-lbl">${t5.blkPlanLbl}</span>
            <ul>${c.detailed_plan.map(p => `<li>${p}</li>`).join("")}</ul>
          </div>
          <div class="verdict-box" style="margin-top:8px;font-size:12.6px;">
            <strong>${t5.tieLbl}</strong> ${c.tie}
          </div>
        </article>
      `).join("")}
    </div>

    <h3 class="section-head">${t5.rankHead}</h3>
    <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:20px;">
      ${t5.rankedDx.map(d => `
        <div class="rank-card">
          <div class="rank-num">${d.rank}</div>
          <div style="flex:1;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px;">
              <strong style="font-size:14px;">${d.title}</strong>
              <span class="badge ${d.cls}">${d.cat}</span>
            </div>
            <div style="font-size:12.8px;color:var(--text-muted);line-height:1.5;">${d.evidence}</div>
          </div>
        </div>
      `).join("")}
    </div>

    <h3 class="section-head">${t5.dilHead}</h3>
    <div style="display:flex;flex-direction:column;gap:12px;">
      ${dilemmas.map(dl => `
        <div class="glass-card">
          <strong style="display:block;font-size:14.6px;margin-bottom:10px;">${dl.title}</strong>
          <div class="grid-2">
            <div class="alert alert-amber">
              <strong style="display:block;font-size:12.5px;margin-bottom:4px;">${dl.left_lbl}</strong>
              <div style="font-size:12.8px;line-height:1.52;">${dl.left}</div>
            </div>
            <div class="alert alert-green">
              <strong style="display:block;font-size:12.5px;margin-bottom:4px;">${dl.right_lbl}</strong>
              <div style="font-size:12.8px;line-height:1.52;">${dl.right}</div>
            </div>
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

function getSavedChecks() {
  try { return JSON.parse(localStorage.getItem("mdt_checklist_v4") || "{}"); }
  catch (e) { return {}; }
}

function toggleCheck(id, val) {
  const s = getSavedChecks();
  s[id] = !!val;
  localStorage.setItem("mdt_checklist_v4", JSON.stringify(s));
  renderTab6Plan();
}

function resetChecklist() {
  localStorage.removeItem("mdt_checklist_v4");
  renderTab6Plan();
}

function setMedFilter(tag) {
  activeMedFilter = tag;
  renderTab6Plan();
}

function renderTab6Plan() {
  const t = getT();
  const t6 = t.t6;
  const saved = getSavedChecks();
  const checks = MDT_DATA.checklist[currentLang];
  const doneCount = checks.filter(c => saved[c.id]).length;
  const auditAll = MDT_DATA.med_audit[currentLang];
  const auditShown = (activeMedFilter === "ALL") ? auditAll : auditAll.filter(m => m.tag === activeMedFilter);
  const rxList = MDT_DATA.rx_table[currentLang];

  document.getElementById("tab-plan").innerHTML = `
    <h2 class="section-head">${t6.head}</h2>

    <div class="glass-card" style="margin-bottom:18px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
        <strong style="font-size:14.4px;">${t6.checkHead}</strong>
        <div style="display:flex;gap:8px;align-items:center;">
          <span class="badge b-blue">${doneCount}/${checks.length} ${t6.doneWord}</span>
          <button type="button" class="btn-filter" onclick="resetChecklist()">${t6.resetBtn}</button>
        </div>
      </div>
      ${checks.map(c => `
        <label class="check-row ${saved[c.id] ? "check-done" : ""}">
          <input type="checkbox" ${saved[c.id] ? "checked" : ""} onchange="toggleCheck('${c.id}', this.checked)">
          <span>${c.txt}</span>
        </label>
      `).join("")}
    </div>

    <div class="alert alert-blue" style="margin-bottom:18px;">
      <strong style="font-size:14.4px;">&#128737;&#65039; ${t6.haHead}</strong>
      <ul style="margin:8px 0 0;padding-inline-start:20px;font-size:13px;line-height:1.58;">
        ${t6.haPoints.map(p => `<li style="margin-bottom:5px;">${p}</li>`).join("")}
      </ul>
    </div>

    <h3 class="section-head">${t6.auditHead}</h3>
    <div class="filter-bar" style="margin-bottom:10px;">
      ${t6.auditFilters.map(f => `<button type="button" class="btn-filter ${activeMedFilter === f.tag ? "active" : ""}" onclick="setMedFilter('${f.tag}')">${f.lbl}</button>`).join("")}
    </div>
    <div class="table-wrap glass-card" style="margin-bottom:18px;">
      <table>
        <thead><tr>${t6.auditCols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
        <tbody>
          ${auditShown.map(m => `
            <tr>
              <td><strong>${m.med}</strong></td>
              <td><span class="badge ${m.badge}">${m.act}</span></td>
              <td style="font-size:12.4px;">${m.mech}</td>
              <td style="font-size:12.4px;color:var(--red);">${m.hazard}</td>
              <td style="font-size:12.4px;">${m.res}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>

    <h3 class="section-head">${t6.rxHead}</h3>
    <div class="table-wrap glass-card" style="margin-bottom:18px;">
      <table>
        <thead><tr>${t6.rxCols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
        <tbody>
          ${rxList.map(r => `
            <tr>
              <td><strong>${r.domain}</strong></td>
              <td style="font-size:12.6px;">${r.drug}</td>
              <td style="font-size:12.4px;">${r.timing}</td>
              <td style="font-size:12.4px;color:var(--text-muted);">${r.rationale}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>

    <h3 class="section-head">${t6.pdfSectionHead}</h3>
    <p class="section-sub">${t6.pdfSectionSub}</p>
    <div class="pdf-dl-grid">
      ${t.pdfLinks.map(p => `
        <a href="${p.href}" target="_blank" rel="noopener" class="pdf-dl-card">
          <strong style="font-size:13.8px;color:var(--accent);">&#128196; ${p.txt}</strong>
          <span style="font-size:11.8px;color:var(--text-muted);word-break:break-all;">${p.href.split("/").pop()}</span>
        </a>
      `).join("")}
    </div>
  `;
}

function printSection() {
  document.body.classList.remove("print-all");
  window.print();
}

function printFullDossier() {
  document.body.classList.add("print-all");
  window.print();
  setTimeout(() => document.body.classList.remove("print-all"), 600);
}

function openModal(idx) {
  const t = getT();
  const r = MDT_DATA.roi_cards[currentLang][idx];
  const btns = t.t3.modalBtns;
  document.getElementById("modalTitle").textContent = r.title;
  document.getElementById("modalHistWrap").innerHTML = r.historical
    ? `<div class="hist-banner">&#9888;&#65039; ${t.histBadge}</div>`
    : `<div class="curr-banner">&#9989; ${t.currBadge}</div>`;
  document.getElementById("modalImg").src = r.img;
  document.getElementById("modalImg").style.transform = "scale(1)";
  document.getElementById("modalRawLink").href = r.img;
  document.getElementById("modalRawLink").textContent = btns.open;
  document.getElementById("mZ1").textContent = btns.z1;
  document.getElementById("mZ2").textContent = btns.z2;
  document.getElementById("mZ3").textContent = btns.z3;
  document.getElementById("mClose").textContent = btns.close;
  document.getElementById("modalCaption").textContent = r.desc;
  document.getElementById("imgModal").classList.add("open");
}

function closeModal() { document.getElementById("imgModal").classList.remove("open"); }
function setZoom(scale) { document.getElementById("modalImg").style.transform = `scale(${scale})`; }

function renderAll() {
  renderHeaderLabels();
  renderSidebarNav();
  renderTab1();
  renderTab2Videos();
  renderTab3Diagnostics();
  renderTab4Prior();
  renderTab5Board();
  renderTab6Plan();
}

(function init() {
  const savedTheme = localStorage.getItem("mdt_theme_v4") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  const savedLang = localStorage.getItem("mdt_lang_strict_v4");
  currentLang = (savedLang === "ar") ? "ar" : "en";
  document.documentElement.setAttribute("lang", currentLang);
  document.documentElement.setAttribute("dir", currentLang === "ar" ? "rtl" : "ltr");
  document.querySelectorAll(".lang-opt").forEach(b => {
    b.classList.toggle("active", b.getAttribute("data-lang") === currentLang);
  });
  renderAll();
})();
</script>
</body>
</html>
"""


def render_dashboard_html(data_json: str, file_count: int) -> str:
    """Inject bilingual payload cleanly into standalone HTML template."""
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json).replace("__FILE_COUNT__", str(file_count))


def write_root_dashboard(html_out: str) -> None:
    """Write standalone root index.html rewriting ../ folder refs to ./ for root GitHub Pages serving."""
    root_html = (
        html_out.replace("../01_Structured_and_Renamed_Inputs/", "./01_Structured_and_Renamed_Inputs/")
        .replace("../05_Shareable_PDF_Reports/", "./05_Shareable_PDF_Reports/")
        .replace("../04_Final_MDT_Deliverables/", "./04_Final_MDT_Deliverables/")
    )
    assert "../01_" not in root_html and "../04_" not in root_html and "../05_" not in root_html
    ROOT_REDIRECT_HTML.write_text(root_html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bilingual MDT clinical dashboard & neuroimaging videos.")
    parser.add_argument("--rebuild-videos", action="store_true", help="Force re-encoding all 3 neuroimaging MP4 files.")
    args = parser.parse_args()

    ffmpeg_bin = find_ffmpeg()
    print(f"Using ffmpeg: {ffmpeg_bin}")
    video_info = build_neuroimaging_videos(ffmpeg_bin, force=args.rebuild_videos)
    for k, info in video_info.items():
        tag = "REBUILT" if info.get("rebuilt") else "VERIFIED"
        print(f"[VIDEO {tag}] {k}: {info['path'].name} ({info['size']:,} bytes)")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    quant = json.loads(QUANT_PATH.read_text(encoding="utf-8"))
    file_count = len(manifest.get("files", []))
    data_json = build_js_data_payload(manifest, quant, file_count)

    DASH_DIR.mkdir(parents=True, exist_ok=True)
    html_out = render_dashboard_html(data_json, file_count)
    DASH_HTML.write_text(html_out, encoding="utf-8")
    write_root_dashboard(html_out)
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"[DASHBOARD OK] {DASH_HTML} ({DASH_HTML.stat().st_size:,} bytes) + Root {ROOT_REDIRECT_HTML} ({ROOT_REDIRECT_HTML.stat().st_size:,} bytes) [{file_count} files]")


if __name__ == "__main__":
    main()
