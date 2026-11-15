#!/usr/bin/env python3
"""
Skill 3: Radiology Enhancer (CLAHE & Micro-Zoom ROI Extractor)
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and unsharp masking
to upright neuroimaging films, deduplicated MRA 3D TOF axial source slices, clinical
handwriting, and Holter ECG strips to surface subtle microvascular, dural/venous,
neuro-otological (IAC/CPA), and deep parenchymal features.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import exposure

WORKSPACE_ROOT = Path(r"d:\mum")
INPUTS_DIR = WORKSPACE_ROOT / "01_Structured_and_Renamed_Inputs"
FRAMES_ROOT = INPUTS_DIR / "02_Extracted_and_Enhanced_MRI_Frames"
AXIAL_DIR = FRAMES_ROOT / "01_Axial_T1_T2_FLAIR_Brain_and_BasalGanglia"
MRA_DIR = FRAMES_ROOT / "02_MRA_Intracranial_Arteries"
MRV_DIR = FRAMES_ROOT / "03_MRV_Venous_Sinuses_and_Left_IJV"
ROI_DIR = FRAMES_ROOT / "04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA"
FULL_CLAHE_DIR = ROI_DIR / "Full_Sheets_CLAHE"
CLIN_DIR = INPUTS_DIR / "08_Clinical_Notes_and_Prescriptions"
AUD_2026_DIR = INPUTS_DIR / "04_Audiology_PTA_and_Tympanometry" / "2026_11_03_FollowUp"
CT_DIR = INPUTS_DIR / "03_CT_Brain_Scans"
EEG_DIR = INPUTS_DIR / "07_Neurophysiology_Prolonged_EEG"
HOLTER_PAGES_DIR = INPUTS_DIR / "06_Cardiology_24h_Holter_ECG" / "Extracted_Pages"


def apply_clahe_unsharp(bgr_img: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8, sharpen: float = 0.45) -> np.ndarray:
    """Apply luminance-preserving CLAHE (in LAB space) + subtle unsharp masking."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l_chan)
    enhanced = cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    if sharpen > 0:
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
        enhanced = cv2.addWeighted(enhanced, 1.0 + sharpen, blur, -sharpen, 0)
    return enhanced


def apply_gray_clahe_and_gamma(bgr_img: np.ndarray, clip_limit: float = 3.2, gamma: float = 1.0) -> np.ndarray:
    """Grayscale CLAHE + skimage gamma adjustment for crisp anatomical contrast."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    if abs(gamma - 1.0) > 1e-3:
        eq = exposure.adjust_gamma(eq, gamma=gamma)
    blur = cv2.GaussianBlur(eq, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(eq, 1.4, blur, -0.4, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def label_tile(bgr_img: np.ndarray, text: str, target_w: int = None, target_h: int = None) -> Image.Image:
    """Resize if requested and add a clean dark banner caption above the tile."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    if target_w and target_h:
        im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    elif target_w:
        scale = target_w / im.width
        im = im.resize((target_w, int(im.height * scale)), Image.Resampling.LANCZOS)

    banner_h = 30
    canvas = Image.new("RGB", (im.width, im.height + banner_h), (18, 22, 28))
    canvas.paste(im, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 7), text, fill=(235, 242, 250), font=font)
    return canvas


def make_grid(tiles: list[Image.Image], cols: int, pad: int = 10, bg=(10, 12, 16)) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles provided")
    max_w = max(t.width for t in tiles)
    max_h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * max_w + (cols + 1) * pad, rows * max_h + (rows + 1) * pad), bg)
    for idx, t in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = pad + c * (max_w + pad)
        y = pad + r * (max_h + pad)
        grid.paste(t, (x, y))
    return grid


def crop_norm(img: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)].copy()


def generate_full_clahe_sheets() -> list[str]:
    FULL_CLAHE_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        (AXIAL_DIR / "MRI_2026-11-05_Film07_Axial_eFLAIR_CSVD_LacunarInfarcts_EthmoidSinusitis_Upright.png", "CLAHE_MRI_Film07_Axial_eFLAIR.png"),
        (AXIAL_DIR / "MRI_2026-11-05_Film09_Axial_eT2W_TSE_Lacunes_IAC_EthmoidSinusitis_Upright.png", "CLAHE_MRI_Film09_Axial_eT2W.png"),
        (AXIAL_DIR / "MRI_2026-11-05_Film08_Axial_T1W_SE_SulcalProminence_LtEyeSiliconeOil_Upright.png", "CLAHE_MRI_Film08_Axial_T1W.png"),
        (AXIAL_DIR / "MRI_2026-11-05_Film05_Axial_DWI_ADC_NoAcuteRestrictedDiffusion_Upright.png", "CLAHE_MRI_Film05_Axial_DWI_ADC.png"),
        (AXIAL_DIR / "MRI_2026-11-05_Film10_Coronal_T2W_and_T1W_TSE_BasalGanglia_IAC_EthmoidSinusitis_Upright.png", "CLAHE_MRI_Film10_Coronal_T2W_T1W.png"),
        (MRV_DIR / "MRV_2026-11-05_Film06_3D_PCA_Venography_Attenuated_Lt_Transverse_Sigmoid_IJV_Upright.png", "CLAHE_MRV_Film06_3D_PCA_Venography.png"),
        (MRA_DIR / "MRA_2026-11-05_Film02_3D_TOF_Coronal_MIP_Intracranial_Arteries_Unremarkable_Upright.png", "CLAHE_MRA_Film02_Coronal_MIP.png"),
        (MRA_DIR / "MRA_2026-11-05_Film03_3D_TOF_Axial_MIP_CircleOfWillis_Unremarkable_Upright.png", "CLAHE_MRA_Film03_Axial_MIP.png"),
    ]
    out_paths = []
    for src, out_name in targets:
        img = cv2.imread(str(src))
        enh = apply_clahe_unsharp(img, clip_limit=3.0, tile_size=8, sharpen=0.4)
        dst = FULL_CLAHE_DIR / out_name
        cv2.imwrite(str(dst), enh)
        out_paths.append(str(dst.relative_to(WORKSPACE_ROOT)))
    return out_paths


def build_roi1_mrv_and_mra_davf_panels() -> list[str]:
    outputs = []
    # ROI 1A: MRV 3D PCA zoomed AP & Oblique views comparing Right vs Left Transverse/Sigmoid/IJV
    mrv_path = MRV_DIR / "MRV_2026-11-05_Film06_3D_PCA_Venography_Attenuated_Lt_Transverse_Sigmoid_IJV_Upright.png"
    mrv = cv2.imread(str(mrv_path))

    # On Film06 upright: top row (y: 0.23..0.34) has AP 18 post projections; bottom row (y: 0.67..0.78) also AP/oblique
    # Left half (x: 0.08..0.52) is dark background; right half (x: 0.52..0.96) is inverted light background
    crops_1a = [
        ("MRV AP Dark MIP (Raw Crop) - Row1", crop_norm(mrv, 0.09, 0.23, 0.52, 0.35), False),
        ("MRV AP Light Inverted (CLAHE) - Row1", crop_norm(mrv, 0.52, 0.23, 0.96, 0.35), True),
        ("MRV Oblique Dark (CLAHE) - Row2", crop_norm(mrv, 0.09, 0.33, 0.52, 0.45), True),
        ("MRV Oblique Light (CLAHE) - Row2", crop_norm(mrv, 0.52, 0.33, 0.96, 0.45), True),
        ("MRV Posterior-Oblique Dark (CLAHE) - Row6", crop_norm(mrv, 0.09, 0.67, 0.52, 0.785), True),
        ("MRV Posterior-Oblique Light (CLAHE) - Row6", crop_norm(mrv, 0.52, 0.67, 0.96, 0.785), True),
    ]
    tiles_1a = []
    for title, c, do_clahe in crops_1a:
        proc = apply_clahe_unsharp(c, clip_limit=3.5, sharpen=0.55) if do_clahe else c
        tiles_1a.append(label_tile(proc, title, target_w=760))
    grid_1a = make_grid(tiles_1a, cols=2)
    out_1a = ROI_DIR / "ROI1a_MRV_Left_Transverse_Sigmoid_IJV_CLAHE_Zoom.png"
    grid_1a.save(out_1a)
    outputs.append(str(out_1a.relative_to(WORKSPACE_ROOT)))

    # ROI 1B: Scan Video01 unique slices across skull base / mastoid / posterior fossa (t = 4.5s to 15.5s)
    meta = json.loads((FRAMES_ROOT / "extraction_metadata.json").read_text(encoding="utf-8"))
    slices = meta["video_extraction"]["slices"]
    skull_base_slices = [s for s in slices if 4.5 <= s["timestamp_sec"] <= 15.5]
    if len(skull_base_slices) >= 12:
        pick_idx = np.linspace(0, len(skull_base_slices) - 1, 12, dtype=int)
        picked = [skull_base_slices[i] for i in pick_idx]
    else:
        picked = skull_base_slices

    tiles_1b = []
    for s in picked:
        fpath = MRA_DIR / "Video01_Source_Slices" / s["filename"]
        im = cv2.imread(str(fpath))
        # Crop strictly the brain/skull-base area (y: 0.18..0.72, x: 0.02..0.98) and enhance with CLAHE
        brain_crop = crop_norm(im, 0.02, 0.18, 0.98, 0.72)
        enh = apply_gray_clahe_and_gamma(brain_crop, clip_limit=3.5, gamma=0.95)
        # Also zoom tightly on posterior fossa + left mastoid/occipital (right side of axial image = patient Left!)
        tiles_1b.append(label_tile(enh, f"TOF 1mm Source t={s['timestamp_sec']:.1f}s (Idx {s['unique_index']})", target_w=420))

    grid_1b = make_grid(tiles_1b, cols=4)
    out_1b = ROI_DIR / "ROI1b_MRA_3D_TOF_SourceSlices_SkullBase_Mastoid_Occipital_CLAHE.png"
    grid_1b.save(out_1b)
    outputs.append(str(out_1b.relative_to(WORKSPACE_ROOT)))

    # ROI 1C: Extra-high-magnification zoom on the LEFT posterior fossa / mastoid / sigmoid / occipital scalp
    # Select slices at jugular bulb / mastoid / torcular / IAC levels (~6s, 8.5s, 10s, 11.5s)
    zoom_targets = [6.0, 8.5, 10.0, 11.5]
    tiles_1c = []
    for t_target in zoom_targets:
        best = min(slices, key=lambda x: abs(x["timestamp_sec"] - t_target))
        im = cv2.imread(str(MRA_DIR / "Video01_Source_Slices" / best["filename"]))
        # Patient's LEFT posterior fossa / mastoid / occipital region is bottom-right of axial MRI (x: 0.42..0.96, y: 0.42..0.72)
        post_lt_raw = crop_norm(im, 0.35, 0.40, 0.96, 0.71)
        post_lt_clahe = apply_gray_clahe_and_gamma(post_lt_raw, clip_limit=4.0, gamma=0.9)
        tiles_1c.append(label_tile(post_lt_raw, f"Raw Lt PostFossa/Mastoid t={best['timestamp_sec']:.1f}s", target_w=520))
        tiles_1c.append(label_tile(post_lt_clahe, f"CLAHE Lt PostFossa/Mastoid t={best['timestamp_sec']:.1f}s", target_w=520))

    grid_1c = make_grid(tiles_1c, cols=2)
    out_1c = ROI_DIR / "ROI1c_Left_Mastoid_Sigmoid_Occipital_MicroZoom_dAVF_Audit.png"
    grid_1c.save(out_1c)
    outputs.append(str(out_1c.relative_to(WORKSPACE_ROOT)))
    return outputs


def build_roi2_neuro_otological_and_orbits() -> str:
    t2_ax = cv2.imread(str(AXIAL_DIR / "MRI_2026-11-05_Film09_Axial_eT2W_TSE_Lacunes_IAC_EthmoidSinusitis_Upright.png"))
    t1_ax = cv2.imread(str(AXIAL_DIR / "MRI_2026-11-05_Film08_Axial_T1W_SE_SulcalProminence_LtEyeSiliconeOil_Upright.png"))
    flair_ax = cv2.imread(str(AXIAL_DIR / "MRI_2026-11-05_Film07_Axial_eFLAIR_CSVD_LacunarInfarcts_EthmoidSinusitis_Upright.png"))
    cor = cv2.imread(str(AXIAL_DIR / "MRI_2026-11-05_Film10_Coronal_T2W_and_T1W_TSE_BasalGanglia_IAC_EthmoidSinusitis_Upright.png"))

    # On upright Axial films: Row 1 (y: 0.22..0.335) has skull base / ethmoids / cerebellum; Row 2 (y: 0.335..0.46) has IAC/CPA & orbits
    tiles = [
        label_tile(apply_clahe_unsharp(crop_norm(t2_ax, 0.05, 0.225, 0.94, 0.345), 3.2), "Axial T2 Row1 (Ethmoids, Mastoids, Lower Cerebellum - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(t2_ax, 0.05, 0.345, 0.94, 0.465), 3.2), "Axial T2 Row2 (Bilateral IAC/CPA, Orbits, Pons - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(t1_ax, 0.05, 0.225, 0.94, 0.345), 3.2), "Axial T1 Row1 (Ethmoids, Left Globe Silicone Oil Hyperintensity - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(flair_ax, 0.05, 0.225, 0.94, 0.345), 3.2), "Axial FLAIR Row1 (Bilateral Ethmoidal Sinusitis & Posterior Fossa - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(cor, 0.05, 0.23, 0.50, 0.35), 3.2), "Coronal T2 Anterior (Ethmoid Sinusitis & Orbits - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(cor, 0.05, 0.46, 0.50, 0.58), 3.2), "Coronal T2 Posterior (IAC/CPA, Hippocampi, Basal Ganglia - CLAHE)", target_w=860),
    ]
    grid = make_grid(tiles, cols=1)
    out_path = ROI_DIR / "ROI2_NeuroOtological_IAC_CPA_Orbits_Ethmoid_CLAHE.png"
    grid.save(out_path)
    return str(out_path.relative_to(WORKSPACE_ROOT))


def build_roi3_basal_ganglia_and_white_matter() -> str:
    flair_ax = cv2.imread(str(AXIAL_DIR / "MRI_2026-11-05_Film07_Axial_eFLAIR_CSVD_LacunarInfarcts_EthmoidSinusitis_Upright.png"))
    t2_ax = cv2.imread(str(AXIAL_DIR / "MRI_2026-11-05_Film09_Axial_eT2W_TSE_Lacunes_IAC_EthmoidSinusitis_Upright.png"))
    t1_ax = cv2.imread(str(AXIAL_DIR / "MRI_2026-11-05_Film08_Axial_T1W_SE_SulcalProminence_LtEyeSiliconeOil_Upright.png"))
    dwi_adc = cv2.imread(str(AXIAL_DIR / "MRI_2026-11-05_Film05_Axial_DWI_ADC_NoAcuteRestrictedDiffusion_Upright.png"))

    # Basal ganglia, periventricular & subcortical centrum semiovale levels are Row 2 & Row 3 & Row 4 on axial films
    tiles = [
        label_tile(apply_clahe_unsharp(crop_norm(flair_ax, 0.05, 0.34, 0.94, 0.465), 3.4), "Axial FLAIR Row2 (Basal Ganglia, Internal Capsule, Periventricular - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(flair_ax, 0.05, 0.46, 0.94, 0.59), 3.4), "Axial FLAIR Row3 (Corona Radiata, Periventricular Sheet, Subcortical - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(t2_ax, 0.05, 0.46, 0.94, 0.59), 3.4), "Axial T2 Row3 (Basal Ganglia / Corona Radiata Lacunes - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(t1_ax, 0.05, 0.46, 0.94, 0.59), 3.4), "Axial T1 Row3 (T1 Hypointense Chronic Lacunes & Sulcal Widening - CLAHE)", target_w=860),
        label_tile(apply_clahe_unsharp(crop_norm(dwi_adc, 0.04, 0.43, 0.96, 0.565), 3.0), "Axial DWI (left) & ADC (right) Mid-Ventricular / Basal Ganglia (No Acute Restriction)", target_w=860),
    ]
    grid = make_grid(tiles, cols=1)
    out_path = ROI_DIR / "ROI3_BasalGanglia_WhiteMatter_LacunarInfarcts_CLAHE.png"
    grid.save(out_path)
    return str(out_path.relative_to(WORKSPACE_ROOT))


def build_roi4_clinical_and_holter_zooms() -> str:
    rx = cv2.imread(str(CLIN_DIR / "CLIN_2026-11-03_Neurosurgery_Rx_DrAymanGalhom_Inderal10_Ciprapra10_CobalF_Brainox.jpg"))
    note = cv2.imread(str(CLIN_DIR / "CLIN_2026-11-03_Neurosurgery_ClinicalNote_DrAymanGalhom_EssentialTremor_Titubation.jpg"))
    holter_p2 = cv2.imread(str(HOLTER_PAGES_DIR / "CARD_2026-11-05_Holter24h_Page02_24HourTrends.png"))
    holter_p3 = cv2.imread(str(HOLTER_PAGES_DIR / "CARD_2026-11-05_Holter24h_Page03_AbsoluteSTAnalysis.png"))

    tiles = [
        label_tile(apply_clahe_unsharp(crop_norm(rx, 0.12, 0.29, 0.92, 0.72), 2.5), "Dr. Ayman Galhom Rx (Diagnosis & Inderal/Ciprapra/CobalF/Brainox)", target_w=840),
        label_tile(apply_clahe_unsharp(crop_norm(note, 0.12, 0.30, 0.85, 0.80), 2.5), "Dr. Ayman Galhom Note (Essential Tremor / Titubation)", target_w=840),
        label_tile(crop_norm(holter_p2, 0.01, 0.50, 0.99, 0.92), "Holter Page 2: 24-Hour Hourly Table (Full Resolution Crop)", target_w=1100),
        label_tile(crop_norm(holter_p3, 0.01, 0.04, 0.99, 0.39), "Holter Page 3: Max ST Depression (-2.1 mm @ 13:11) Strip & Table", target_w=1100),
    ]
    grid = make_grid(tiles, cols=1)
    out_path = ROI_DIR / "ROI4_Clinical_Handwriting_and_Holter_KeyStrips_Zoom.png"
    grid.save(out_path)
    return str(out_path.relative_to(WORKSPACE_ROOT))


def build_roi5_new_baseline_audiology_and_neurophys() -> list[str]:
    outputs = []
    # ROI 5A: Definitive Follow-Up Audiology (2026-11-03 Masked PTA + GSI-39 Tymp)
    pta_im = cv2.imread(str(AUD_2026_DIR / "AUD_2026-11-03_PTA_Masked_AC_BC_FollowUp_RtWRS84_LtWRS76_PureSNHL.jpg"))
    tymp_im = cv2.imread(str(AUD_2026_DIR / "AUD_2026-11-03_Tympanometry_GSI39_FollowUp_Rt1.5cm3_Lt3.2cm3_ExtremeTypeAd.jpg"))

    tiles_5a = [
        label_tile(
            apply_clahe_unsharp(crop_norm(pta_im, 0.02, 0.03, 0.98, 0.62), clip_limit=2.2, sharpen=0.35),
            "2026-11-03 Definitive Masked PTA Curves (Rt 20-70 dB; Lt 50-80 dB; 0 dB ABG Pure SNHL)",
            target_w=1080,
        ),
        label_tile(
            apply_clahe_unsharp(crop_norm(pta_im, 0.02, 0.60, 0.98, 0.96), clip_limit=2.4, sharpen=0.4),
            "Speech Table (Rt WRS 84% @ 65 dB; Lt WRS 76% @ 90 dB) + Handwritten Note Claiming 'Rt As'",
            target_w=1080,
        ),
        label_tile(
            apply_clahe_unsharp(crop_norm(tymp_im, 0.04, 0.02, 0.96, 0.97), clip_limit=2.4, sharpen=0.45),
            "GSI-39 Thermal Printout: Rt 1.5 cm3 @ -5 daPa (Normal Type A, NOT As) | Lt 3.2 cm3 @ 0 daPa (Extreme Type Ad) | Ipsi NT",
            target_w=1080,
        ),
    ]
    grid_5a = make_grid(tiles_5a, cols=1)
    out_5a = ROI_DIR / "ROI5a_Audiology_2026-11-03_MaskedPTA_and_GSI39_Tymp_Zoom.png"
    grid_5a.save(out_5a)
    outputs.append(str(out_5a.relative_to(WORKSPACE_ROOT)))

    # ROI 5B: Jul 2024 CT Brain (Silicone Oil Forensic Catch) & Prolonged EEG Tremor/EMG Epochs
    ct_req = cv2.imread(str(CT_DIR / "RAD_2024-07-17_CT_Brain_Requisition_IMC_Reason_Tremors.jpg"))
    ct_rep = cv2.imread(str(CT_DIR / "RAD_2024-07-27_CT_Brain_NonContrast_Report_Involutional_LtEyeSiliconeOil.jpg"))
    eeg_rep = cv2.imread(str(EEG_DIR / "NEURO_2024-07-17_Prolonged_2h_EEG_Report_IMC_Normal9HzAlpha.jpg"))
    eeg_ep3 = cv2.rotate(
        cv2.imread(str(EEG_DIR / "NEURO_2024-07-17_EEG_Tracing_Epoch03_12-01-10_PM_AlphaBackground_MuscleArtifact.jpg")),
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    )
    eeg_ep7 = cv2.rotate(
        cv2.imread(str(EEG_DIR / "NEURO_2024-07-17_EEG_Tracing_Epoch07_13-21-46_PM_AlphaBackground_MuscleArtifact.jpg")),
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    )

    doc_tiles = make_grid(
        [
            label_tile(apply_clahe_unsharp(crop_norm(ct_req, 0.02, 0.14, 0.98, 0.86), 2.2), "2024-07-17 IMC CT Req (#474510 Tremors)", target_w=360),
            label_tile(apply_clahe_unsharp(crop_norm(ct_rep, 0.02, 0.14, 0.98, 0.86), 2.2), "2024-07-27 CT Report (Lt Vitreous Silicone Oil)", target_w=360),
            label_tile(apply_clahe_unsharp(crop_norm(eeg_rep, 0.02, 0.14, 0.98, 0.86), 2.2), "2024-07-17 IMC 2h EEG Report (9 Hz Alpha)", target_w=360),
        ],
        cols=3,
        pad=8,
    )
    doc_bgr = cv2.cvtColor(np.array(doc_tiles), cv2.COLOR_RGB2BGR)

    tiles_5b = [
        label_tile(doc_bgr, "2024-07 Primary Records: CT Requisition, CT Report (Pre-Existing Silicone Oil) & Normal 2h EEG Report", target_w=1120),
        label_tile(apply_clahe_unsharp(crop_norm(eeg_ep3, 0.21, 0.02, 0.79, 0.98), 2.5), "2024-07-17 EEG Epoch 03 (12:01:10 PM): Normal Alpha + Fronto-Temporal EMG/Tremor Burst (Zero Spikes)", target_w=1120),
        label_tile(apply_clahe_unsharp(crop_norm(eeg_ep7, 0.21, 0.02, 0.79, 0.98), 2.5), "2024-07-17 EEG Epoch 07 (13:21:46 PM): Marked Somatic Tremor/EMG Artifact (No Epileptiform Activity)", target_w=1120),
    ]
    grid_5b = make_grid(tiles_5b, cols=1)
    out_5b = ROI_DIR / "ROI5b_2024_CT_Brain_and_ProlongedEEG_TremorArtifacts.png"
    grid_5b.save(out_5b)
    outputs.append(str(out_5b.relative_to(WORKSPACE_ROOT)))
    return outputs


def main() -> None:
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    full_clahe = generate_full_clahe_sheets()
    roi1 = build_roi1_mrv_and_mra_davf_panels()
    roi2 = build_roi2_neuro_otological_and_orbits()
    roi3 = build_roi3_basal_ganglia_and_white_matter()
    roi4 = build_roi4_clinical_and_holter_zooms()
    roi5 = build_roi5_new_baseline_audiology_and_neurophys()

    report = {
        "full_clahe_sheets": full_clahe,
        "roi_panels": roi1 + [roi2, roi3, roi4] + roi5,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
