#!/usr/bin/env python3
"""
Skill 7: Ophthalmology & Peripheral Vascular Auto-Cropper, Deduplicator & Manifest Updater
=========================================================================================
Discovers the 36 newly uploaded raw images (32 unique clinical captures + 4 exact byte-identical
duplicates spanning 2011 Lower-Limb Doppler, 2014 Optovue OCT/Macula/ONH, 2016 Optovue ONH,
2019 cataract/RD referral + Optopol OCT + OS B-Scans, 2021 OD/OS silicone-oil B-Scans, and
June–July 2026 Topcon Maestro2 OCT + Zeiss HFA Visual Field + CSO Flash VEP/ERG).

Pipeline:
1. Preserves all 36 untouched raw originals in `00_Raw_Backup_Archive/` with MD5/SHA-256 verification.
2. Applies explicit orientation normalization (e.g., -90 deg CCW on horizontal Doppler strip #06).
3. Strips any mobile gallery chrome / uniform letterbox border bars.
4. Detects the bright neutral clinical paper sheet + contiguous printed dark ultrasound/OCT boxes in
   CIE LAB color space (filtering warm/patterned bedsheets & tables) and crops tightly with a
   uniform 15-px safety margin.
5. Saves cropped PNGs under clean semantic filenames inside `01_Structured_and_Renamed_Inputs/`
   (`09_Ophthalmology_Fundus_and_OpticNerve/` & `10_Peripheral_Vascular_LowerLimb_Doppler/`).
6. Tags ALL pre-2026 records (2011, 2014, 2016, 2019, 2021, 2024) across the ENTIRE `case_manifest.json`
   and `FILE_RENAMING_INDEX.md` with `"temporal_status": "HISTORICAL_OLD"` and
   `"warning_badge": "⚠️ HISTORICAL RECORD — DO NOT CONFUSE WITH CURRENT STATUS"`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

WORKSPACE_ROOT = Path(r"d:\mum")
BACKUP_DIR = WORKSPACE_ROOT / "00_Raw_Backup_Archive"
INPUTS_DIR = WORKSPACE_ROOT / "01_Structured_and_Renamed_Inputs"
MANIFEST_PATH = INPUTS_DIR / "case_manifest.json"
INDEX_PATH = INPUTS_DIR / "FILE_RENAMING_INDEX.md"
HIST_BADGE = "⚠️ HISTORICAL RECORD — DO NOT CONFUSE WITH CURRENT STATUS"
SAFE_MARGIN_PX = 15

# Exact byte-identical duplicates in the uploaded batch (preserved in 00_Raw_Backup_Archive only)
EXACT_DUPLICATE_ALIASES: dict[str, str] = {
    "photo_2026-11-15_18-27-00 (4).jpg": "photo_2026-11-15_18-27-00 (2).jpg",
    "photo_2026-11-15_18-27-00.jpg": "photo_2026-11-15_18-27-00 (3).jpg",
    "photo_2026-11-15_18-27-01.jpg": "photo_2026-11-15_18-27-01 (2).jpg",
    "photo_2026-11-15_18-40-17.jpg": "photo_2026-11-15_18-40-17 (2).jpg",
}

NEW_CLINICAL_CATALOG: list[dict] = [
    # --- 2011-06-21 Lower Limb Color Doppler (HISTORICAL_OLD) ---
    {
        "raw_name": "photo_2026-11-15_18-26-53.jpg",
        "new_name": "VASC_2011-06-21_LowerLimb_ColorDoppler_Report_DrHelmyRashad_LtCalfPerforatorIncompetence.png",
        "folder": "10_Peripheral_Vascular_LowerLimb_Doppler",
        "category": "VASC",
        "test_date": "2011-06-21",
        "rotate_ccw": 0,
        "summary": "Typed Lower Limb Arterial & Venous Color Doppler Report (2011-06-21, Dr. Helmy Rashad): patent ATA/PTA arteries with normal triphasic/biphasic flow; patent SFV/popliteal/tibial deep veins with zero DVT; incompetent left mid/lower calf perforators feeding medial superficial varicosities.",
    },
    {
        "raw_name": "photo_2026-11-15_18-26-49 (2).jpg",
        "new_name": "VASC_2011-06-21_LowerLimb_ColorDoppler_Strip01_AltadamonHospital.png",
        "folder": "10_Peripheral_Vascular_LowerLimb_Doppler",
        "category": "VASC",
        "test_date": "2011-06-21",
        "rotate_ccw": 0,
        "summary": "Lower Limb Color Doppler thermal strip 1/4 (Altadamon Hospital, 2011-06-21 10:18-10:19 AM): spectral venous Doppler waveforms confirming patency.",
    },
    {
        "raw_name": "photo_2026-11-15_18-26-50.jpg",
        "new_name": "VASC_2011-06-21_LowerLimb_ColorDoppler_Strip02_NormalArteriesDeepVeins.png",
        "folder": "10_Peripheral_Vascular_LowerLimb_Doppler",
        "category": "VASC",
        "test_date": "2011-06-21",
        "rotate_ccw": 0,
        "summary": "Lower Limb Color Doppler thermal strip 2/4 (Altadamon Hospital, 2011-06-21 10:18 AM): multiphasic peripheral arterial and compressible deep venous tracings.",
    },
    {
        "raw_name": "photo_2026-11-15_18-26-51.jpg",
        "new_name": "VASC_2011-06-21_LowerLimb_ColorDoppler_Strip03_Waveforms_Upright.png",
        "folder": "10_Peripheral_Vascular_LowerLimb_Doppler",
        "category": "VASC",
        "test_date": "2011-06-21",
        "rotate_ccw": 90,
        "summary": "Lower Limb Color Doppler thermal strip 3/4 (Altadamon Hospital, 2011-06-21 10:16-10:17 AM, rotated upright): duplex B-mode + pulsed-wave spectral profiles.",
    },
    {
        "raw_name": "photo_2026-11-15_18-26-52.jpg",
        "new_name": "VASC_2011-06-21_LowerLimb_ColorDoppler_Strip04_DeepFemoralVenousFlow.png",
        "folder": "10_Peripheral_Vascular_LowerLimb_Doppler",
        "category": "VASC",
        "test_date": "2011-06-21",
        "rotate_ccw": 0,
        "summary": "Lower Limb Color Doppler thermal strip 4/4 (Altadamon Hospital, 2011-06-21 10:16 AM): proximal venous Doppler spectrum + Dr. Helmy Rashad stamp.",
    },
    # --- 2014-04-23 Optovue OCT Macula & Optic Nerve (HISTORICAL_OLD) ---
    {
        "raw_name": "photo_2026-11-15_18-26-47.jpg",
        "new_name": "OPHTH_2014-04-23_Optovue_Macula_OD_CystoidEdema_RPEDisruption_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT",
        "category": "OPHTH",
        "test_date": "2014-04-23",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Optovue OCT OD (2014-04-23, PortSaid Lasik Center): right eye macular cross-sections + en-face reference showing chronic intraretinal cystoid spaces and RPE irregularity/disruption.",
    },
    {
        "raw_name": "photo_2026-11-15_18-26-49.jpg",
        "new_name": "OPHTH_2014-04-23_Optovue_MaculaMap_EMM5_OU_RPE_Disruption_OD280_OS100_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT",
        "category": "OPHTH",
        "test_date": "2014-04-23",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Optovue Macula EMM5 Map OU (2014-04-23): OD SSI 35.8 Global RPE/Choroid Disruption 280 (central 208 um); OS SSI 37.1 Global Disruption 100 (central 207 um).",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-00.jpg",
        "new_name": "OPHTH_2014-04-23_Optovue_OpticNerve_OS_RNFL78.36um_CDVert0.83_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2014_04_23_Historical_Optovue_OCT",
        "category": "OPHTH",
        "test_date": "2014-04-23",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Optovue ONH/RNFL OS (2014-04-23): pre-existing glaucomatous optic neuropathy years before RD! SSI 46.5, Disc Area 1.61 mm2, Rim Area 0.38 mm2, Vertical C/D 0.83, RNFL Avg 78.36 um (NL 58, IT 77 borderline/thin).",
    },
    # --- 2016-02-16 Optovue ONH OS (HISTORICAL_OLD) ---
    {
        "raw_name": "photo_2026-11-15_18-26-48.jpg",
        "new_name": "OPHTH_2016-02-16_Optovue_OpticNerve_OS_RNFL71.41um_CDVert0.74_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2016_02_16_Historical_Optovue_ONH",
        "category": "OPHTH",
        "test_date": "2016-02-16",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Optovue ONH/RNFL OS (2016-02-16, Dr. Sanaa Hamam): progressive RNFL loss from 78.36 to 71.41 um (Sup 76.87, Inf 65.96; red ST 58, NL 48, IN 79), Disc Area 1.81 mm2, Vertical C/D 0.74.",
    },
    # --- 2019 Historical Referral, Optopol OCT & B-Scans (HISTORICAL_OLD) ---
    {
        "raw_name": "photo_2026-11-15_18-27-14.jpg",
        "new_name": "OPHTH_2019-08-19_MOH_HealthInsurance_Referral_LtCataract_RD_PriorRtRD_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan",
        "category": "OPHTH",
        "test_date": "2019-08-19",
        "rotate_ccw": 0,
        "summary": "HISTORICAL MOH Health Insurance Referral (2019-08-19, ID 26304090300188, BP 130/80): notes DM/HTN female with PRIOR Right cataract + retinal detachment surgery, now Lt cataract + RD needing surgery + urgent optic nerve OCT.",
    },
    {
        "raw_name": "photo_2026-11-15_18-27-01 (2).jpg",
        "new_name": "OPHTH_2019-08-20_Optopol_OCT_Report_OU_DrOmaimaMohamed_MarkedTSNIT_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan",
        "category": "OPHTH",
        "test_date": "2019-08-20",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Optopol OCT Written Report (2019-08-20, No 1107, Dr. Omaima Mohamed): marked OU TSNIT RNFL affection, recommends automated visual field ('Computerized field is a must').",
    },
    {
        "raw_name": "photo_2026-11-15_18-27-00 (2).jpg",
        "new_name": "OPHTH_2019-08-20_Optopol_OCT_Report_OU_DrOmaimaMohamed_AltTake_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan",
        "category": "OPHTH",
        "test_date": "2019-08-20",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Optopol OCT Written Report alternate capture (2019-08-20, Dr. Omaima Mohamed; dup raw photo_18-27-00 (4) archived).",
        "is_secondary_take": True,
    },
    {
        "raw_name": "photo_2026-11-15_18-27-15.jpg",
        "new_name": "OPHTH_2019-08-20_Optopol_OCT_OD_Spectral_MeanTSNIT260_CD0.24_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan",
        "category": "OPHTH",
        "test_date": "2019-08-20",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Optopol Spectral OCT OD (2019-08-20, No 1107): Disc Area 5.92 mm2, C/D 0.24, Mean TSNIT 260 um (artifactually elevated segmentation over Rt silicone/vitreoretinal interface).",
    },
    {
        "raw_name": "photo_2026-11-15_18-27-14 (2).jpg",
        "new_name": "OPHTH_2019-08-20_Optopol_OCT_OS_SevereRNFLThinning_MeanTSNIT64_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan",
        "category": "OPHTH",
        "test_date": "2019-08-20",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Optopol Spectral OCT OS (2019-08-20, No 1107): severe RNFL atrophy Mean TSNIT 64 um (red: N 57, T 71, I 53; 4 weeks prior to OS vitreous hemorrhage).",
    },
    {
        "raw_name": "photo_2026-11-15_18-27-16.jpg",
        "new_name": "OPHTH_2019-09-11_BScan_OS_VitreousHemorrhage_NoRetinalDetachment_DrOmaimaSharara_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan",
        "category": "OPHTH",
        "test_date": "2019-09-11",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Ocular B-Scan OS (2019-09-11, Dr. Omaima Sharara, Gain 90 dB): vitreous hemorrhage floaters with intact retina ('لا رسوم شبكي'), 3 months before OS silicone oil PPV.",
    },
    {
        "raw_name": "photo_2026-11-15_18-26-55.jpg",
        "new_name": "OPHTH_2019-12-15_BScan_OS_SiliconizedEye_RetinaInPlace_DrHaithamKhalil_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2019_Historical_OCT_Referral_BScan",
        "category": "OPHTH",
        "test_date": "2019-12-15",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Ocular B-Scan OS (2019-12-15, El Ramad Hospital, Dr. Haitham Khalil): 'Left siliconized eye w/ Retina in place' post-vitrectomy.",
    },
    # --- 2021 Historical Ocular B-Scans OD & OS (HISTORICAL_OLD) ---
    {
        "raw_name": "photo_2026-11-15_18-26-57.jpg",
        "new_name": "OPHTH_2021-04-07_BScan_OD_EmulsifiedSiliconeOil_RetinaInPlace_DrYasminSinan_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans",
        "category": "OPHTH",
        "test_date": "2021-04-07",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Ocular B-Scan OD (2021-04-07, Ramad Port Said Hospital, Dr. Yasmin Talaat Sinan, Gain 95 dB): proves BOTH eyes had silicone oil! Right eye 'emulsified silicon / Retina in place'.",
    },
    {
        "raw_name": "photo_2026-11-15_18-27-00 (3).jpg",
        "new_name": "OPHTH_2021-04-07_BScan_OS_SiliconFilledEye_RetinaInPlace_DrYasminSinan_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans",
        "category": "OPHTH",
        "test_date": "2021-04-07",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Ocular B-Scan OS (2021-04-07, Ramad Port Said Hospital, Dr. Yasmin Talaat Sinan, Gain 94 dB): 'silicon filled eye / Retina in place'.",
    },
    {
        "raw_name": "photo_2026-11-15_18-26-56.jpg",
        "new_name": "OPHTH_2021-05-29_BScan_OS_SiliconFilledEye_RetinaInPlace_DrManarElGayar_HISTORICAL.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2021_Historical_Ocular_BScans",
        "category": "OPHTH",
        "test_date": "2021-05-29",
        "rotate_ccw": 0,
        "summary": "HISTORICAL Ocular B-Scan OS (2021-05-29, Ramad Port Said Hospital, Dr. Manar Khaled El-Gayar): 'silicon oil filled eye / retina in place' 3 months prior to MOH legal blindness cert.",
    },
    # --- CURRENT 2026-06 Topcon Maestro2 OCT & Zeiss Visual Field (CURRENT_2026) ---
    {
        "raw_name": "photo_2026-11-15_18-40-03.jpg",
        "new_name": "OPHTH_2026-06-01_Topcon_OCT_Macula_Radial_OD_Avg221.9um_Qual10.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField",
        "category": "OPHTH",
        "test_date": "2026-06-01",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 Topcon Maestro2 Radial Macula OD (2026-06-01, ID A900000367): Image Quality 10 (clear optical media, silicone removed/clear OD); ETDRS central 259 um, Avg 221.9 um with inferior/temporal inner retinal thinning.",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-08.jpg",
        "new_name": "OPHTH_2026-06-01_Topcon_OCT_Macula_Radial_OS_LowQual2_SiliconeMediaOpacity.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField",
        "category": "OPHTH",
        "test_date": "2026-06-01",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 Topcon Maestro2 Radial Macula OS (2026-06-01): Image Quality 2 (severely degraded by left eye silicone oil interface/media opacity -> segmentation artifact 400.3 um).",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-05.jpg",
        "new_name": "OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OD_RNFL40um_SevereAtrophy_Qual9.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField",
        "category": "OPHTH",
        "test_date": "2026-06-01",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 Topcon Maestro2 3D Disc / ONH OD (2026-06-01): Image Quality 9 (reliable!). PROFOUND OPTIC ATROPHY: Total RNFL 40 um (Sup 57, Inf 34 um), Vertical C/D 0.70, Linear C/D 0.71.",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-06.jpg",
        "new_name": "OPHTH_2026-06-01_Topcon_OCT_OpticNerve_3DDisc_OS_Qual2_MediaArtifact.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField",
        "category": "OPHTH",
        "test_date": "2026-06-01",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 Topcon Maestro2 3D Disc / ONH OS (2026-06-01): Image Quality 2 (unreadable RNFL map due to left media/silicone oil scatter — 90 um printout is segmentation noise, do not misinterpret as healthy nerve!).",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-09.jpg",
        "new_name": "OPHTH_2026-06-27_Zeiss_VisualField_HFA_VFI18pct_UnreliableTest_DrAlaaKhaled.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_06_Topcon_OCT_and_VisualField",
        "category": "OPHTH",
        "test_date": "2026-06-27",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 Zeiss HFA 24-2 Visual Field (2026-06-26/27, ID A900000367): VFI 18%, MD threshold exceeded across superior/temporal field with small residual inferior-nasal island; signed 'Not reliable Test' by Dr. Alaa Khaled.",
    },
    # --- CURRENT 2026-07-08 CSO Electrophysiology: Flash VEP & Full-Field ERG (CURRENT_2026) ---
    {
        "raw_name": "photo_2026-11-15_18-40-07.jpg",
        "new_name": "OPHTH_2026-07-08_CSO_Flash_VEP_OD_Present_N1P1_10.29uV_P2_125.39ms.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG",
        "category": "OPHTH",
        "test_date": "2026-07-08",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 CSO Flash VEP OD (2026-07-08): MEASURABLE cortical flash transmission! N1-P1 10.29 uV, P2 peak 125.39 ms (delayed/attenuated), steady-state 23 Hz Amp 5.11 uV.",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-17 (2).jpg",
        "new_name": "OPHTH_2026-07-08_CSO_Flash_VEP_OS_Present_N1P1_7.99uV_P2_132.42ms.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG",
        "category": "OPHTH",
        "test_date": "2026-07-08",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 CSO Flash VEP OS (2026-07-08): CRITICAL LEFT-EYE ELECTROPHYSIOLOGY — Flash VEP is MEASURABLE (NOT extinguished!), N1-P1 7.99 uV, P1 86.72 ms, P2 132.42 ms, 13 Hz Amp 3.51 uV, proving partial residual retino-cortical conduction despite severe neuropathy & retinopathy.",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-09 (2).jpg",
        "new_name": "OPHTH_2026-07-08_CSO_ERG_DarkAdapted_0.01_OU_OD39.1uV_OS35.8uV.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG",
        "category": "OPHTH",
        "test_date": "2026-07-08",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 CSO Dark-Adapted 0.01 Rod ERG OU (2026-07-08): markedly reduced scotopic rod b-waves bilaterally (OD ISO-B1 39.10 uV, OS 35.76 uV), yet residual rod activity remains.",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-11.jpg",
        "new_name": "OPHTH_2026-07-08_CSO_ERG_DarkAdapted_3.0_OU_OD94.4uV_OS76.8uV.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG",
        "category": "OPHTH",
        "test_date": "2026-07-08",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 CSO Dark-Adapted 3.0 Mixed Rod-Cone ERG OU (2026-07-08): severely subnormal mixed responses bilaterally (OD A1-B1 94.42 uV / A1 -87.20 uV; OS A1-B1 76.78 uV / A1 -57.66 uV) with reduced b:a amplitude ratio.",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-12.jpg",
        "new_name": "OPHTH_2026-07-08_CSO_ERG_DarkAdapted_3.0_OU_AltTake.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG",
        "category": "OPHTH",
        "test_date": "2026-07-08",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 CSO Dark-Adapted 3.0 ERG OU alternate photo capture (2026-07-08).",
        "is_secondary_take": True,
    },
    {
        "raw_name": "photo_2026-11-15_18-40-17 (3).jpg",
        "new_name": "OPHTH_2026-07-08_CSO_ERG_OscillatoryPotentials_OD10.0uV_OS0.74uV_SevereOSIschemia.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG",
        "category": "OPHTH",
        "test_date": "2026-07-08",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 CSO Dark-Adapted 3.0 Oscillatory Potentials OU (2026-07-08): dramatic OS vs OD inner-retinal ischemic collapse — OD N1-P1 10.00 uV (148 Hz Amp 13.27 uV) vs OS N1-P1 0.74 uV (152 Hz Amp 6.38 uV), explaining poor functional prognosis in OS despite partial VEP.",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-10.jpg",
        "new_name": "OPHTH_2026-07-08_CSO_ERG_LightAdapted_3.0_OU_OD37.1uV_OS30.4uV.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG",
        "category": "OPHTH",
        "test_date": "2026-07-08",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 CSO Light-Adapted 3.0 Photopic Cone ERG OU (2026-07-08): reduced photopic cone responses (OD A1-B1 37.13 uV, OS 30.38 uV) with delayed peak times (OD 57.1 ms, OS 55.7 ms).",
    },
    {
        "raw_name": "photo_2026-11-15_18-40-18.jpg",
        "new_name": "OPHTH_2026-07-08_CSO_ERG_LightAdapted_3.0_Flicker_OD21.2uV_OS13.0uV.png",
        "folder": "09_Ophthalmology_Fundus_and_OpticNerve/2026_07_CSO_VEP_and_ERG",
        "category": "OPHTH",
        "test_date": "2026-07-08",
        "rotate_ccw": 0,
        "summary": "CURRENT 2026 CSO Light-Adapted 3.0 Flicker ERG OU (2026-07-08): severe bilateral macular/cone pathway impairment, significantly worse OS (OD A1-B1 21.19 uV @ 29.83 ms; OS 12.97 uV @ 31.45 ms).",
    },
]


def file_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def temporal_metadata(test_date: str) -> dict:
    year = int(str(test_date)[:4])
    if year < 2026:
        return {"temporal_status": "HISTORICAL_OLD", "warning_badge": HIST_BADGE}
    return {"temporal_status": "CURRENT_2026", "warning_badge": None}


def _contiguous_central_span(profile: np.ndarray, thresh: float = 0.28) -> tuple[int, int]:
    """Find the longest contiguous index interval where `profile >= thresh` (bridging holes <= 18px)."""
    active = (profile >= thresh).astype(np.uint8)
    # Bridge tiny dark internal print lines/folds up to 18 px, NEVER large cloth margins
    active = cv2.morphologyEx(active.reshape(-1, 1), cv2.MORPH_CLOSE, np.ones((19, 1), np.uint8)).ravel()
    n = len(active)
    best_s, best_e, best_len = 0, n, -1
    cur_s = None
    for idx, val in enumerate(active):
        if val and cur_s is None:
            cur_s = idx
        elif not val and cur_s is not None:
            if idx - cur_s > best_len:
                best_s, best_e, best_len = cur_s, idx, idx - cur_s
            cur_s = None
    if cur_s is not None and (n - cur_s) > best_len:
        best_s, best_e = cur_s, n
    return int(best_s), int(best_e)


def detect_document_crop_box(bgr: np.ndarray, margin: int = SAFE_MARGIN_PX) -> tuple[int, int, int, int]:
    """
    Precision document & ultrasound box detector:
    Uses CIE LAB neutral-white paper identification + small-gap connected-component
    filtering (without bridging across dark bedsheet gaps onto adjacent stacks)
    cross-checked against contiguous row/column paper occupancy spans, with a
    uniform 15-px safety margin.
    """
    h, w = bgr.shape[:2]

    # Step 1: Strip any flat uniform gallery chrome / letterbox bands on top/bottom
    row_std = np.std(bgr, axis=(1, 2))
    non_flat_y = np.where(row_std > 4.0)[0]
    y_min_chrome = int(non_flat_y[0]) if len(non_flat_y) else 0
    y_max_chrome = int(non_flat_y[-1]) + 1 if len(non_flat_y) else h

    # Step 2: Strict neutral white/gray paper mask in CIE LAB
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    chroma = np.hypot(a - 128.0, b - 128.0)

    # Neutral clinical paper has L >= 140 and low chroma (<= 11.5); dark printed B-scan/OCT areas inside
    # the paper still have white paper borders surrounding them above/below/sides.
    strict_paper = ((L >= 140.0) & (chroma <= 11.5)).astype(np.uint8) * 255
    if y_min_chrome > 0:
        strict_paper[:y_min_chrome, :] = 0
    if y_max_chrome < h:
        strict_paper[y_max_chrome:, :] = 0

    # Clean noise flecks on fabric WITHOUT wide bridging across cloth margins onto background pages
    strict_paper = cv2.morphologyEx(strict_paper, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    strict_closed = cv2.morphologyEx(strict_paper, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Largest connected paper sheet component (by pixel area)
    contours, _ = cv2.findContours(strict_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, y_min_chrome, w, y_max_chrome

    main_cnt = max(contours, key=cv2.contourArea)
    sheet_mask = np.zeros_like(strict_closed)
    cv2.drawContours(sheet_mask, [main_cnt], -1, 255, thickness=-1)

    # If thermal ultrasound strip has dark frames split into 2+ white halves by a horizontal fold or black box,
    # include ALL secondary neutral paper contours with area >= 4% of main_cnt (never clip split ultrasound strips!)
    mx, my, mw, mh = cv2.boundingRect(main_cnt)
    main_area = cv2.contourArea(main_cnt)
    for cnt in contours:
        if cnt is main_cnt:
            continue
        ar = cv2.contourArea(cnt)
        cx, cy, cw, ch_box = cv2.boundingRect(cnt)
        x_ov = max(0, min(mx + mw, cx + cw) - max(mx, cx))
        if ar >= 0.04 * main_area and (x_ov >= 0.30 * min(mw, cw) or cw >= 0.45 * w):
            cv2.drawContours(sheet_mask, [cnt], -1, 255, thickness=-1)

    ys, xs = np.where(sheet_mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    # Step 3: Trim any residual narrow bridge tail via 1D contiguous occupancy profile
    # Include both bright neutral paper AND dark neutral ultrasound boxes inside the span so black Doppler frames never break the span!
    dark_neutral = ((L < 95.0) & (chroma <= 18.0)).astype(np.float32)
    doc_mask = np.maximum((sheet_mask > 0).astype(np.float32), dark_neutral * 0.8)
    rx0, rx1 = _contiguous_central_span(doc_mask[y0:y1, :].mean(axis=0), thresh=0.22)
    ry0, ry1 = _contiguous_central_span(doc_mask[:, rx0:rx1].mean(axis=1), thresh=0.22)
    x0, x1 = max(x0, rx0), min(x1, rx1)
    y0, y1 = max(y0, ry0), min(y1, ry1)

    # Step 4: Explicitly union any dark printed ultrasound / OCT boxes overlapping [x0..x1] or within [x0..x1] vertically/horizontally
    dark_us = ((L < 95.0) & (chroma < 20.0)).astype(np.uint8) * 255
    dark_us = cv2.morphologyEx(dark_us, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    dark_us = cv2.morphologyEx(dark_us, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8))
    us_contours, _ = cv2.findContours(dark_us, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in us_contours:
        ux, uy, uw, uh = cv2.boundingRect(c)
        x_ov = max(0, min(x1, ux + uw) - max(x0, ux))
        if uh >= 40 and uw >= 0.25 * (x1 - x0) and x_ov >= 0.35 * min(uw, x1 - x0):
            x0 = min(x0, ux)
            x1 = max(x1, ux + uw)
            y0 = min(y0, uy)
            y1 = max(y1, uy + uh)

    # Apply safe 15-px margin
    x0 = max(0, x0 - margin)
    y0 = max(y_min_chrome, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(y_max_chrome, y1 + margin)

    # Safety fallback: never crop to implausibly tiny sliver (< 22% image area)
    if (x1 - x0) * (y1 - y0) < 0.22 * w * h:
        return 0, y_min_chrome, w, y_max_chrome

    return x0, y0, x1, y1


def load_upright_bgr(path: Path, rotate_ccw: int) -> np.ndarray:
    pil_im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    if rotate_ccw == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate_ccw == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif rotate_ccw == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    return bgr


def make_comparison_sheets(records: list[dict], out_dir: Path) -> list[Path]:
    """Build 2 visual inspection sheets showing Original (with red crop box) -> Cropped output."""
    out_paths: list[Path] = []
    chunk_size = 16
    cell_w, cell_h = 560, 220
    cols = 2  # 2 pairs per row => 4 images wide
    for s_idx in range(0, len(records), chunk_size):
        batch = records[s_idx : s_idx + chunk_size]
        rows = (len(batch) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + 40), (24, 28, 36))
        draw = ImageDraw.Draw(canvas)
        sheet_num = s_idx // chunk_size + 1
        draw.text((16, 10), f"AUTO-CROP VISUAL AUDIT SHEET {sheet_num} ({len(batch)} items) — Red Box = Detected Paper Crop (+15px margin)", fill=(248, 250, 252))
        for i, rec in enumerate(batch):
            r, c = divmod(i, cols)
            bx, by = c * cell_w + 8, 38 + r * cell_h
            # Original with red rect
            orig = cv2.imread(rec["backup_abs"])
            if rec["rotate_ccw"] == 90:
                orig = cv2.rotate(orig, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x0, y0, x1, y1 = rec["crop_box"]
            cv2.rectangle(orig, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 8)
            orig_rgb = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
            orig_rgb.thumbnail((250, 175))
            cropped_rgb = Image.open(rec["out_abs"]).convert("RGB")
            cropped_rgb.thumbnail((270, 175))
            canvas.paste(orig_rgb, (bx, by + 24))
            canvas.paste(cropped_rgb, (bx + 260, by + 24))
            tag = "HIST" if rec["temporal_status"] == "HISTORICAL_OLD" else "2026"
            pct = rec["area_kept_pct"]
            draw.text((bx, by + 4), f"#{s_idx+i+1:02d} [{tag}] {rec['test_date']} ({pct:.0f}% area) {rec['new_name'][:42]}", fill=(250, 204, 21) if tag == "HIST" else (74, 222, 128))
        out_p = out_dir / f"crop_verify_sheet_{sheet_num}.png"
        canvas.save(out_p, quality=92)
        out_paths.append(out_p)
    return out_paths


def run_crop_and_organize(unlink_root_raws: bool = False, verify_dir: Path | None = None) -> dict:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Backup ALL 36 raw files (including the 4 exact MD5 duplicates) first
    for dup_name in EXACT_DUPLICATE_ALIASES:
        src = WORKSPACE_ROOT / dup_name
        dst = BACKUP_DIR / dup_name
        if src.exists():
            src_md5, _ = file_hashes(src)
            if not dst.exists():
                shutil.copy2(src, dst)
            assert file_hashes(dst)[0] == src_md5

    crop_records: list[dict] = []
    new_manifest_entries: list[dict] = []

    for item in NEW_CLINICAL_CATALOG:
        raw_name = item["raw_name"]
        raw_src = WORKSPACE_ROOT / raw_name
        backup_dst = BACKUP_DIR / raw_name
        source_to_read = raw_src if raw_src.exists() else backup_dst
        if not source_to_read.exists():
            raise FileNotFoundError(f"Missing new raw image: {raw_name}")

        src_md5, src_sha256 = file_hashes(source_to_read)
        if raw_src.exists():
            if not backup_dst.exists():
                shutil.copy2(raw_src, backup_dst)
            assert file_hashes(backup_dst)[0] == src_md5

        # Auto-crop & upright transform
        bgr = load_upright_bgr(backup_dst, item.get("rotate_ccw", 0))
        H, W = bgr.shape[:2]
        x0, y0, x1, y1 = detect_document_crop_box(bgr, SAFE_MARGIN_PX)
        cropped = bgr[y0:y1, x0:x1]

        out_folder = INPUTS_DIR / item["folder"]
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / item["new_name"]
        cv2.imwrite(str(out_path), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 6])

        tmeta = temporal_metadata(item["test_date"])
        out_size = out_path.stat().st_size
        out_md5, out_sha256 = file_hashes(out_path)
        area_pct = 100.0 * ((x1 - x0) * (y1 - y0)) / (W * H)

        crop_records.append(
            {
                "raw_name": raw_name,
                "new_name": item["new_name"],
                "test_date": item["test_date"],
                "rotate_ccw": item.get("rotate_ccw", 0),
                "crop_box": [x0, y0, x1, y1],
                "orig_dim": [W, H],
                "cropped_dim": [x1 - x0, y1 - y0],
                "area_kept_pct": area_pct,
                "temporal_status": tmeta["temporal_status"],
                "backup_abs": str(backup_dst),
                "out_abs": str(out_path),
            }
        )

        dups = [d for d, target in EXACT_DUPLICATE_ALIASES.items() if target == raw_name]
        entry = {
            "original_raw_filename": raw_name,
            "new_semantic_filename": item["new_name"],
            "folder_path": f"01_Structured_and_Renamed_Inputs/{item['folder']}/{item['new_name']}",
            "backup_path": f"00_Raw_Backup_Archive/{raw_name}",
            "category_code": item["category"],
            "date_of_test": item["test_date"],
            "temporal_status": tmeta["temporal_status"],
            "warning_badge": tmeta["warning_badge"],
            "size_bytes": out_size,
            "md5": out_md5,
            "sha256": out_sha256,
            "raw_source_md5": src_md5,
            "auto_crop_box_xyxy": [x0, y0, x1, y1],
            "cropped_dimensions": f"{x1 - x0}x{y1 - y0}",
            "one_line_clinical_summary": item["summary"],
        }
        if dups:
            entry["exact_duplicate_raw_backups"] = dups
        if item.get("is_secondary_take"):
            entry["is_secondary_take"] = True
        new_manifest_entries.append(entry)

    # Step 2: Update case_manifest.json cleanly preserving all 37 existing files + adding temporal tags to ALL entries
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing_files = [
        f for f in manifest["files"] if not str(f.get("folder_path", "")).startswith(("01_Structured_and_Renamed_Inputs/09_", "01_Structured_and_Renamed_Inputs/10_"))
    ]
    for ef in existing_files:
        tmeta = temporal_metadata(ef["date_of_test"])
        ef["temporal_status"] = tmeta["temporal_status"]
        ef["warning_badge"] = tmeta["warning_badge"]

    for hist in manifest.get("clinical_history_records_without_raw_local_files", []):
        tmeta = temporal_metadata(hist["test_date"])
        hist["temporal_status"] = tmeta["temporal_status"]
        hist["warning_badge"] = tmeta["warning_badge"]

    all_files = existing_files + new_manifest_entries
    manifest["generated_at"] = datetime.now().isoformat()
    manifest["total_raw_files_organized"] = len(all_files)
    manifest["total_raw_files_in_backup_archive"] = len(list(BACKUP_DIR.iterdir()))
    manifest["definitive_ophthalmology_dates"] = ["2026-06-01", "2026-06-27", "2026-07-08"]
    manifest["historical_cutoff_year"] = 2026
    manifest["files"] = all_files

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Step 3: Regenerate FILE_RENAMING_INDEX.md
    index_lines = [
        "# Workspace File Renaming Index & Clinical Inventory (Chronologically Tagged)",
        "",
        f"**Patient:** {manifest['patient_name']} | **Verified DOB:** 1963-04-09 (National ID `26304090300188`; MRI DICOM header lists `27-Mar-1963`)",
        f"**Generated:** {manifest['generated_at']} | **Total Unique Organized Files:** {len(all_files)} (`37` baseline neuro/cardiac/ENT/EEG/CT + `32` auto-cropped ophthalmology/lower-limb Doppler; `73` total raw files backed up in `00_Raw_Backup_Archive/` including `4` byte-identical duplicate photos)",
        "",
        "> ⚠️ **TEMPORAL AUDIT GUARDRAIL:** All pre-2026 records (`2011`, `2014`, `2016`, `2019`, `2021`, `2024`) carry `HISTORICAL_OLD` status and `⚠️ HISTORICAL RECORD — DO NOT CONFUSE WITH CURRENT STATUS` so longitudinal baseline trends are never conflated with active 2026 status.",
        "",
        "## 1. Local Workspace Files Mapping Table (69 Unique Organized Records)",
        "",
        "| # | Status | Original Raw Filename | New Semantic Filename | Date of Test | Size (Bytes) | One-Line Clinical Summary |",
        "|---|---|---|---|---|---:|---|",
    ]
    for idx, entry in enumerate(all_files, start=1):
        status_label = "⚠️ `HISTORICAL_OLD`" if entry.get("temporal_status") == "HISTORICAL_OLD" else "✅ `CURRENT_2026`"
        index_lines.append(
            f"| {idx} | {status_label} | `{entry['original_raw_filename']}` | `{entry['new_semantic_filename']}` | {entry['date_of_test']} | {entry['size_bytes']:,} | {entry['one_line_clinical_summary']} |"
        )

    index_lines.extend(
        [
            "",
            "## 2. Historical Narrative Records Without Raw Local Scans (Retained for Timeline Continuity)",
            "",
            "| Status | Category | Date of Test | Modality / Source | Workspace Status | Clinical Summary |",
            "|---|---|---|---|---|---|",
        ]
    )
    for rec in manifest.get("clinical_history_records_without_raw_local_files", []):
        st = "⚠️ `HISTORICAL_OLD`" if rec.get("temporal_status") == "HISTORICAL_OLD" else "✅ `CURRENT_2026`"
        index_lines.append(
            f"| {st} | {rec['category']} | {rec['test_date']} | {rec['modality']} | {rec['local_file_status']} | {rec['summary']} |"
        )
    INDEX_PATH.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    # Step 4: Optional verification sheets
    sheets: list[Path] = []
    if verify_dir:
        verify_dir.mkdir(parents=True, exist_ok=True)
        sheets = make_comparison_sheets(crop_records, verify_dir)

    # Step 5: Remove root raw photos ONLY when requested AND backup MD5s are verified
    removed_count = 0
    if unlink_root_raws:
        all_new_raws = [c["raw_name"] for c in NEW_CLINICAL_CATALOG] + list(EXACT_DUPLICATE_ALIASES.keys())
        for rn in all_new_raws:
            rp = WORKSPACE_ROOT / rn
            bp = BACKUP_DIR / rn
            if rp.exists() and bp.exists() and file_hashes(rp)[0] == file_hashes(bp)[0]:
                rp.unlink()
                removed_count += 1

    return {
        "unique_cropped": len(crop_records),
        "total_manifest_files": len(all_files),
        "backup_count": manifest["total_raw_files_in_backup_archive"],
        "removed_root_raws": removed_count,
        "verify_sheets": [str(s) for s in sheets],
        "crops": crop_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlink-root", action="store_true", help="Remove raw root photo_*.jpg files after MD5-verified backup")
    parser.add_argument(
        "--verify-dir",
        type=Path,
        default=None,
        help="Optional directory to write before/after visual verification contact sheets",
    )
    args = parser.parse_args()
    res = run_crop_and_organize(unlink_root_raws=args.unlink_root, verify_dir=args.verify_dir)
    print(f"Processed {res['unique_cropped']} unique cropped plates -> {res['total_manifest_files']} total manifest files.")
    print(f"Backup archive count: {res['backup_count']} files. Unlinked root duplicates: {res['removed_root_raws']}.")
    for c in res["crops"]:
        print(f"  {c['test_date']} | {c['area_kept_pct']:5.1f}% | {c['cropped_dim']} | {c['new_name']}")
    for s in res["verify_sheets"]:
        print(f"Verify sheet: {s}")


if __name__ == "__main__":
    main()
