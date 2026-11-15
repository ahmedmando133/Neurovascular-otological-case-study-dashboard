#!/usr/bin/env python3
"""
Skill 1: Semantic File Organizer & Auto-Renamer
Discovers all raw files in workspace_root, preserves untouched backups in
00_Raw_Backup_Archive/, renames files using strict clinical metadata convention:
[CategoryCode]_[YYYY-MM-DD]_[Modality_or_TestType]_[Specific_Sequence_or_Key_Finding].[ext]
organizes them into 01_Structured_and_Renamed_Inputs/, verifies MD5 checksums,
and generates case_manifest.json and FILE_RENAMING_INDEX.md.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(r"d:\mum")
BACKUP_DIR = WORKSPACE_ROOT / "00_Raw_Backup_Archive"
INPUTS_DIR = WORKSPACE_ROOT / "01_Structured_and_Renamed_Inputs"

SUBFOLDERS = [
    "01_Neuroimaging_MRI_MRA_MRV_Videos",
    "02_Extracted_and_Enhanced_MRI_Frames/01_Axial_T1_T2_FLAIR_Brain_and_BasalGanglia",
    "02_Extracted_and_Enhanced_MRI_Frames/02_MRA_Intracranial_Arteries",
    "02_Extracted_and_Enhanced_MRI_Frames/03_MRV_Venous_Sinuses_and_Left_IJV",
    "02_Extracted_and_Enhanced_MRI_Frames/04_CLAHE_ROI_Zoom_Left_Mastoid_Sigmoid_and_CPA",
    "03_CT_Brain_Scans",
    "04_Audiology_PTA_and_Tympanometry/2024_11_02_Baseline",
    "04_Audiology_PTA_and_Tympanometry/2026_11_03_FollowUp",
    "05_Vascular_Carotid_Duplex",
    "06_Cardiology_24h_Holter_ECG",
    "07_Neurophysiology_Prolonged_EEG",
    "08_Clinical_Notes_and_Prescriptions",
]

OTHER_TOP_DIRS = [
    "02_Custom_Skills_and_Scripts",
    "03_Individual_Consultant_Audits",
    "04_Final_MDT_Deliverables",
]

# Deterministic clinical mapping verified by direct multimodal visual + PDF inspection
RAW_FILE_CATALOG = [
    {
        "raw_name": "6aecc202e4017dfe6fe8b1bd-1793924756.pdf",
        "new_name": "RAD_2026-11-05_MRI_MRA_MRV_Brain_Report_PortSaeedMilitary_DrGhadaAwad.pdf",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "RAD",
        "test_date": "2026-11-05",
        "summary": "Official 3-page MRI/MRA/MRV Brain report (reported 2026-11-06 via Rology, Dr. Ghada Awad): involutional changes, CSVD/lacunar infarcts (periventricular/deep WM, Rt basal ganglia, Lt frontal, Rt parietal), bilateral ethmoidal sinusitis, unremarkable MRA, attenuated Lt transverse/sigmoid sinuses & Lt IJV on MRV.",
    },
    {
        "raw_name": "IMG_8048.MOV",
        "new_name": "MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil.mov",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRA",
        "test_date": "2026-11-05",
        "summary": "28.1-sec DICOM viewer cine video scrolling through ~237 1.0-mm axial source slices of Sc 11 MRA/3D/TOF CLEAR (skull base to vertex), showing arterial flow, IAC/posterior fossa anatomy, and hyperintense left globe vitreous (intraocular silicone oil).",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-34.jpg",
        "new_name": "MRI_2026-11-05_Film01_Sagittal_T1W_SE_Brain_Midline_NormalBrainstem.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRI",
        "test_date": "2026-11-05",
        "summary": "MRI Brain Sagittal T1W_SE CLEAR (20 slices + scouts, 18:58): preserved midline corpus callosum, pituitary, brainstem, cerebellar vermis; age-related sulcal prominence.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-34 (2).jpg",
        "new_name": "MRA_2026-11-05_Film02_3D_TOF_Coronal_MIP_Intracranial_Arteries_Unremarkable.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRA",
        "test_date": "2026-11-05",
        "summary": "MRA/3D/TOF CLEAR Coronal MIP projections (Sc 11, 12 dark & 12 inverted views, 19:00): patent bilateral ICAs, MCAs, ACAs, PCAs, vertebral and basilar arteries without aneurysm or large-vessel stenosis.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-35.jpg",
        "new_name": "MRA_2026-11-05_Film03_3D_TOF_Axial_MIP_CircleOfWillis_Unremarkable.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRA",
        "test_date": "2026-11-05",
        "summary": "MRA/3D/TOF CLEAR Transverse/Axial MIP projections (Sc 11, 12 dark & 12 inverted views, 19:00, raw capture rotated 90 deg): patent Circle of Willis and vertebrobasilar junction.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-36.jpg",
        "new_name": "MRA_2026-11-05_Film04a_3D_TOF_SagittalOblique_MIP_Intracranial_Arteries_Take1.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRA",
        "test_date": "2026-11-05",
        "summary": "MRA/3D/TOF CLEAR Coronal/Sagittal-Oblique (RL -10 right / AP 1 post) MIP projections (Sc 11, 12 views, 19:00, camera take 1).",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-36 (2).jpg",
        "new_name": "MRA_2026-11-05_Film04b_3D_TOF_SagittalOblique_MIP_Intracranial_Arteries_Take2.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRA",
        "test_date": "2026-11-05",
        "summary": "MRA/3D/TOF CLEAR Coronal/Sagittal-Oblique (RL -10 right / AP 1 post) MIP projections (Sc 11, 12 views, 19:00, camera take 2).",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-37.jpg",
        "new_name": "MRI_2026-11-05_Film05_Axial_DWI_ADC_NoAcuteRestrictedDiffusion.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRI",
        "test_date": "2026-11-05",
        "summary": "MRI Brain Axial isoDWI SENSE (b=1000) & ADC map (Sc 4, 18 slices each, 18:47): no acute restricted diffusion, excluding acute infarction.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-39.jpg",
        "new_name": "MRV_2026-11-05_Film06_3D_PCA_Venography_Attenuated_Lt_Transverse_Sigmoid_IJV.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRV",
        "test_date": "2026-11-05",
        "summary": "MRV 3D PCA SAG SENSE (Sc 12, T1FFE/PCA/M, 20 dark & 20 inverted views, 19:07, raw capture rotated 90 deg): right-dominant venous drainage with attenuated Lt transverse sinus, Lt sigmoid sinus, and Lt IJV; no dural thrombosis.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-39 (2).jpg",
        "new_name": "MRI_2026-11-05_Film07_Axial_eFLAIR_CSVD_LacunarInfarcts_EthmoidSinusitis.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRI",
        "test_date": "2026-11-05",
        "summary": "MRI Brain Axial eFLAIR_longTR SENSE (Sc 5, 20 slices + scouts, 18:48): periventricular thin sheet and multifocal deep WM/subcortical/basal ganglia FLAIR hyperintensities (CSVD + chronic lacunes), sulcal widening, bilateral ethmoidal sinusitis.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-40.jpg",
        "new_name": "MRI_2026-11-05_Film08_Axial_T1W_SE_SulcalProminence_LtEyeSiliconeOil.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRI",
        "test_date": "2026-11-05",
        "summary": "MRI Brain Axial T1W_SE CLEAR (Sc 7, 20 slices + scouts, 18:54): cortical sulcal prominence, T1-hypointense chronic lacunar foci, hyperintense left globe vitreous consistent with intraocular silicone oil.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-41.jpg",
        "new_name": "MRI_2026-11-05_Film09_Axial_eT2W_TSE_Lacunes_IAC_EthmoidSinusitis.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRI",
        "test_date": "2026-11-05",
        "summary": "MRI Brain Axial eT2W_TSE SENSE (Sc 6, 20 slices + scouts, 18:52, raw capture inverted 180 deg): deep WM and right basal ganglia T2 hyperintense lacunes, symmetric IACs/CPAs, bilateral ethmoidal sinusitis.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-41 (2).jpg",
        "new_name": "MRI_2026-11-05_Film10_Coronal_T2W_and_T1W_TSE_BasalGanglia_IAC_EthmoidSinusitis.jpg",
        "folder": "01_Neuroimaging_MRI_MRA_MRV_Videos",
        "category": "MRI",
        "test_date": "2026-11-05",
        "summary": "MRI Brain Coronal T2W_TSE SENSE (Sc 8, 18:55) & Coronal T1W_TSE CLEAR (Sc 9, 18:56): bilateral ethmoidal sinus mucosal thickening, basal ganglia/deep WM lacunar disease, preserved hippocampi and IAC/CPA anatomy.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-42.jpg",
        "new_name": "VASC_2026-11-07_Carotid_Vertebral_ColorDuplex_IMT0.13cm_LtBulbPlaque5mm.jpg",
        "folder": "05_Vascular_Carotid_Duplex",
        "category": "VASC",
        "test_date": "2026-11-07",
        "summary": "Bilateral Carotid & Vertebral Color Duplex Ultrasonography (Dr. Ibrahim Hashim, 2026-11-07): IMT 0.13 cm, bilateral atheromatous plaques most prominent in Lt carotid bulb measuring 5 mm without hemodynamic stenosis, normal vertebral artery flow (Mild Atherosclerosis).",
    },
    {
        "raw_name": "أمال مختار.pdf",
        "new_name": "CARD_2026-11-05_Holter24h_Complete_29Pages_Summary_HRV_and_FullDisclosure.pdf",
        "folder": "06_Cardiology_24h_Holter_ECG",
        "category": "CARD",
        "test_date": "2026-11-05",
        "summary": "Complete 29-page 24-hour Holter ECG report (Portsaid Military Hospital, 2026-11-05 09:00 start): Summary, 24h Trends, Absolute ST, QTc, HRV (SDNN 101 ms), and Full Disclosure Ch.1 (09:00-08:59); 857 PVCs (311 between 06:00-08:59), 4 V-runs, 153 SVEs, ST depression -2.1 mm CH3 @ 13:11, normal QTc max 326 ms.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-43.jpg",
        "new_name": "CARD_2026-11-05_Holter24h_Page01_Summary_857PVCs_153SVEs_STdepression.jpg",
        "folder": "06_Cardiology_24h_Holter_ECG",
        "category": "CARD",
        "test_date": "2026-11-05",
        "summary": "Photo of Holter Page 1 Report Summary (Avg HR 65 bpm, Min 51, Max 99; 857 VEs, 22 V-pairs, 4 V-runs, 153 SVEs, Max ST -2.1 mm CH3, SDNN 101 ms).",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-46.jpg",
        "new_name": "CARD_2026-11-05_Holter24h_Page02_24HourTrends_HR51to99_MorningPVCSurge.jpg",
        "folder": "06_Cardiology_24h_Holter_ECG",
        "category": "CARD",
        "test_date": "2026-11-05",
        "summary": "Photo of Holter Page 2 24-Hour Trends Report showing hourly HR, SDNN, ST deviations, and 06:00-08:59 AM PVC surge (164 PVCs at 08:00-08:59).",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-44.jpg",
        "new_name": "CARD_2026-11-05_Holter24h_Page03_Absolute_ST_Analysis_Minus2.1mm_CH3.jpg",
        "folder": "06_Cardiology_24h_Holter_ECG",
        "category": "CARD",
        "test_date": "2026-11-05",
        "summary": "Photo of Holter Page 3 Absolute ST Analysis (160 ST min CH3, max depression -2.1 mm @ 13:11, max elevation +6.7 mm @ 21:27 baseline/artifact step).",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-43 (2).jpg",
        "new_name": "CARD_2026-11-05_Holter24h_Page04_QTc_and_QT_3CH_Analysis_NormalQTc.jpg",
        "folder": "06_Cardiology_24h_Holter_ECG",
        "category": "CARD",
        "test_date": "2026-11-05",
        "summary": "Photo of Holter Page 4 QTc & QT 3-CH Analysis (100% QTc <= 450 ms, max QTc 326 ms @ 14:27, max QT 287 ms @ 12:13).",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-46 (2).jpg",
        "new_name": "CLIN_2026-11-03_Neurosurgery_Rx_DrAymanGalhom_Inderal10_Ciprapra10_CobalF_Brainox.jpg",
        "folder": "08_Clinical_Notes_and_Prescriptions",
        "category": "CLIN",
        "test_date": "2026-11-03",
        "summary": "Neurosurgery prescription by Prof. Dr. Ayman Elsayed Ali Galhom (2026-11-03, No. 25334) noting DM, HTN, Essential/Intention Tremor; prescribing Inderal 10 mg BID, Ciprapra (Escitalopram) 10 mg 1/2 tab post-dinner, Cobal F 1 tab qAM, Brainox 1 cap post-lunch.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-47.jpg",
        "new_name": "CLIN_2026-11-03_Neurosurgery_ClinicalNote_DrAymanGalhom_EssentialTremor_Titubation.jpg",
        "folder": "08_Clinical_Notes_and_Prescriptions",
        "category": "CLIN",
        "test_date": "2026-11-03",
        "summary": "Handwritten clinical exam notepad by Prof. Dr. Ayman Galhom documenting Essential tremor and Titubation (head tremor) with no lower-limb involvement note.",
    },
    {
        "raw_name": "photo_2026-11-15_13-44-47 (2).jpg",
        "new_name": "CLIN_2026-11-02_Neurosurgery_Order_DrAymanGalhom_MRI_MRV_MRA_Duplex_Holter.jpg",
        "folder": "08_Clinical_Notes_and_Prescriptions",
        "category": "CLIN",
        "test_date": "2026-11-02",
        "summary": "Neurosurgery diagnostic workup order by Prof. Dr. Ayman Galhom (2026-11-02, No. 25334) requesting MRI brain, MRV brain, MRA brain, Color Doppler neck vessels carotid and vertebral, and Holter.",
    },
    {
        "raw_name": "photo_2026-11-15_13-48-43.jpg",
        "new_name": "CLIN_2021-09-01_MOH_MedicalCouncil_LegalBlindness_LessThan1of60_RetinalOpticAtrophy.jpg",
        "folder": "08_Clinical_Notes_and_Prescriptions",
        "category": "CLIN",
        "test_date": "2021-09-01",
        "summary": "Egyptian Ministry of Health Specialized Medical Councils certificate (session 2021-09-01, decision 2021-09-19, National ID 26304090300188 confirming DOB 1963-04-09): legal blindness (<1/60 in better eye) due to retinal and optic nerve atrophy, unfit to drive.",
    },
    # --- DEFINITIVE FOLLOW-UP AUDIOLOGY & TYMPANOMETRY BASELINE (NOV 3, 2026) ---
    {
        "raw_name": "photo_2026-11-15_14-53-22 (2).jpg",
        "new_name": "AUD_2026-11-03_PTA_Masked_AC_BC_FollowUp_RtWRS84_LtWRS76_PureSNHL.jpg",
        "folder": "04_Audiology_PTA_and_Tympanometry/2026_11_03_FollowUp",
        "category": "AUD",
        "test_date": "2026-11-03",
        "summary": "Definitive Masked PTA (Al-Hayah Port Fouad Hospital, Dr. Shahenda el-KHamesy): Rt AC 20/25/45/50/50/70 dB (SRT 25 dB, WRS 84% @ 65 dB); Lt AC 50/50/55/55-60/70/80 dB (SRT 50 dB, WRS 76% @ 90 dB); masked BC matches AC (0 dB ABG, pure asymmetrical SNHL). Handwritten note says 'Rt As'—disproven by accompanying GSI-39 thermal printout.",
    },
    {
        "raw_name": "photo_2026-11-15_14-53-22.jpg",
        "new_name": "AUD_2026-11-03_Tympanometry_GSI39_FollowUp_Rt1.5cm3_Lt3.2cm3_ExtremeTypeAd.jpg",
        "folder": "04_Audiology_PTA_and_Tympanometry/2026_11_03_FollowUp",
        "category": "AUD",
        "test_date": "2026-11-03",
        "summary": "Definitive GSI-39 Tympanometry printout (226 Hz): Rt Peak 1.5 cm3 @ -5 daPa (ECV 1.3 cm3, GR 25 daPa -> normal/upper-normal Type A, contradicting handwritten 'Rt As'); Lt Peak 3.2 cm3 @ 0 daPa exceeding 3.0 cm3 graph ceiling (ECV 1.2 cm3, GR 10 daPa -> extreme Type Ad); Ipsi 1000 Hz reflexes marked NT.",
    },
    # --- CT BRAIN DOCUMENTS (JUL 2024) ---
    {
        "raw_name": "photo_2026-11-15_14-55-10 (2).jpg",
        "new_name": "RAD_2024-07-17_CT_Brain_Requisition_IMC_Reason_Tremors.jpg",
        "folder": "03_CT_Brain_Scans",
        "category": "RAD",
        "test_date": "2024-07-17",
        "summary": "IMC Referral Order #474510 (Prof. Dr. Ahmed Fahmy, Age 61y 3m 8d): Non-contrast CT Brain requisition ordered explicitly for 'tremors'.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-10 (3).jpg",
        "new_name": "RAD_2024-07-27_CT_Brain_NonContrast_Report_Involutional_LtEyeSiliconeOil.jpg",
        "folder": "03_CT_Brain_Scans",
        "category": "RAD",
        "test_date": "2024-07-27",
        "summary": "El-Zohor Specialized Hospital Non-Contrast CT Brain report (Pat ID A900000367, e-signed Dr. Emad Mohamed MSc): age-related involutional changes, normal posterior fossa/ventricles, no hemorrhage, plus explicit 'left vitreous hyperdensity likely due to intra-ocular silicon oil' (proving pre-existing silicone oil 27 months before Nov 2026 MRI).",
    },
    # --- PROLONGED 2-HOUR EEG REPORT & 9 TIMESTAMPED EPOCH TRACINGS (JUL 17, 2024) ---
    {
        "raw_name": "photo_2026-11-15_14-55-10.jpg",
        "new_name": "NEURO_2024-07-17_Prolonged_2h_EEG_Report_IMC_Normal9HzAlpha.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "IMC Prolonged 2h EEG Report (Prof. Dr. Ahmed Fahmy, Study 2024-07-17): symmetric 9 C/Sec alpha background, unremarkable HV & photic stimulation, zero epileptiform discharges; confirms benign EEG ruling out cortical seizure etiology for tremors/spells.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-12.jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch01_11-42-44_AM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 01 (11:42:44 AM, 42s into recording, DOB header 9/4/1963): awake 9 Hz posterior alpha background with mild frontopolar EMG muscle artifact, no spikes/sharp waves.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-11 (4).jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch02_11-48-49_AM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 02 (11:48:49 AM): symmetric 9 Hz background rhythm with intermittent movement/EMG contamination, zero epileptiform discharges.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-11 (3).jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch03_12-01-10_PM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 03 (12:01:10 PM): high-frequency fronto-temporal EMG/tremor bursts superimposed on normal background without cortical spike-wave correlates.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-11 (2).jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch04_12-10-07_PM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 04 (12:10:07 PM): intact symmetric cerebral background with bilateral EMG/tremor interference, zero paroxysmal slowing.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-11.jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch05_12-35-17_PM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 05 (12:35:17 PM): mid-recording epoch showing preserved symmetric cerebral rhythms with mild anterior muscle artifact.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-12 (4).jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch06_12-55-51_PM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 06 (12:55:51 PM): dense high-frequency somatic EMG/tremor bursts (no post-burst suppression or cortical epileptiform sharp waves).",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-12 (2).jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch07_13-21-46_PM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 07 (13:21:46 PM): marked bilateral fronto-temporal EMG/tremor bursts confirming peripheral/subcortical tremor generator rather than cortical myoclonus.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-12 (3).jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch08_13-24-34_PM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 08 (13:24:34 PM): normal symmetric cerebral activity with fading muscle artifact, no ictal evolution.",
    },
    {
        "raw_name": "photo_2026-11-15_14-55-10 (4).jpg",
        "new_name": "NEURO_2024-07-17_EEG_Tracing_Epoch09_13-28-39_PM_AlphaBackground_MuscleArtifact.jpg",
        "folder": "07_Neurophysiology_Prolonged_EEG",
        "category": "NEURO",
        "test_date": "2024-07-17",
        "summary": "Prolonged EEG Tracing Epoch 09 (13:28:39 PM): late 2h recording epoch confirming normal background and absence of epileptiform abnormalities.",
    },
]

CLINICAL_RECORDS_FROM_HISTORY_ONLY = [
    {
        "category": "AUD",
        "test_date": "2024-11-02",
        "modality": "Historical Baseline PTA & GSI-39 Tympanometry Note (Al-Hayah Port Fouad — Superseded by 2026-11-03 Definitive Baseline)",
        "local_file_status": "Historical narrative comparison only (no raw image in workspace; superseded by verified 2026-11-03 primary scans)",
        "summary": "Prior unmasked AC-only PTA (BC omitted): Rt PTA 35 dB, Lt PTA 55 dB; prior GSI-39 Rt Peak 0.3 cm3 (+10 daPa, Type As), Lt Peak 1.7 cm3 (-5 daPa, Type Ad), Ipsi 1 kHz reflex 105 dB HL (+Metz recruitment, 45 dB SL).",
    },
    {
        "category": "CLIN",
        "test_date": "2026-11-02",
        "modality": "EHA Family Medicine Note (PMH, Meds & Audiology Referral)",
        "local_file_status": "Documented in clinical history (raw image not in local workspace)",
        "summary": "T2DM on Toujeo 30 IU + Apidra 50 IU + Gliptus Plus; HTN on Angiopress Comp; Dyslipidemia on Zatorva 20 mg; Aggrex 75 mg; Sinemet for PD diagnosis; prior Lt retinal detachment surgery.",
    },
    {
        "category": "CLIN",
        "test_date": "2026-11-03",
        "modality": "ENT Prescription (Dr. Mohamed Salem Abdel-Maksoud)",
        "local_file_status": "Documented in clinical history (raw image not in local workspace)",
        "summary": "Suspected vascular etiology behind left ear; prescribed Meclizine qPM x 1 mo & Cetrovit Ginkgo qAM x 2 mo; ordered VNG and vascular imaging.",
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


def ensure_directories() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for sub in SUBFOLDERS:
        (INPUTS_DIR / sub).mkdir(parents=True, exist_ok=True)
    for d in OTHER_TOP_DIRS:
        (WORKSPACE_ROOT / d).mkdir(parents=True, exist_ok=True)


def run_organization() -> dict:
    ensure_directories()
    manifest_entries = []

    for item in RAW_FILE_CATALOG:
        raw_name = item["raw_name"]
        raw_src = WORKSPACE_ROOT / raw_name
        backup_dst = BACKUP_DIR / raw_name
        renamed_dst = INPUTS_DIR / item["folder"] / item["new_name"]

        # Resolve source: either still at root or already backed up on a re-run
        if raw_src.exists():
            src_md5, src_sha256 = file_hashes(raw_src)
            src_size = raw_src.stat().st_size
            if not backup_dst.exists():
                shutil.copy2(raw_src, backup_dst)
            bkp_md5, _ = file_hashes(backup_dst)
            assert bkp_md5 == src_md5, f"Backup MD5 mismatch for {raw_name}"

            if not renamed_dst.exists():
                shutil.copy2(raw_src, renamed_dst)
            ren_md5, _ = file_hashes(renamed_dst)
            assert ren_md5 == src_md5, f"Renamed MD5 mismatch for {raw_name}"

            # Safe to remove raw root file after verifying both backup and renamed copy
            raw_src.unlink()
        elif backup_dst.exists():
            src_md5, src_sha256 = file_hashes(backup_dst)
            src_size = backup_dst.stat().st_size
            if not renamed_dst.exists():
                shutil.copy2(backup_dst, renamed_dst)
            ren_md5, _ = file_hashes(renamed_dst)
            assert ren_md5 == src_md5, f"Renamed MD5 mismatch on re-run for {raw_name}"
        else:
            raise FileNotFoundError(f"Neither root nor backup file found for {raw_name}")

        manifest_entries.append(
            {
                "original_raw_filename": raw_name,
                "new_semantic_filename": item["new_name"],
                "folder_path": f"01_Structured_and_Renamed_Inputs/{item['folder']}/{item['new_name']}",
                "backup_path": f"00_Raw_Backup_Archive/{raw_name}",
                "category_code": item["category"],
                "date_of_test": item["test_date"],
                "size_bytes": src_size,
                "md5": src_md5,
                "sha256": src_sha256,
                "one_line_clinical_summary": item["summary"],
            }
        )

    manifest = {
        "patient_name": "Amal Mokhtar Mohamed Ibrahim (آمال مختار محمد إبراهيم)",
        "dob_verified_from_national_id": "1963-04-09",
        "dob_clerical_variant_on_mri_film": "1963-03-27",
        "national_id": "26304090300188",
        "definitive_audiology_baseline_date": "2026-11-03",
        "definitive_holter_date": "2026-11-05",
        "generated_at": datetime.now().isoformat(),
        "total_raw_files_organized": len(manifest_entries),
        "files": manifest_entries,
        "clinical_history_records_without_raw_local_files": CLINICAL_RECORDS_FROM_HISTORY_ONLY,
    }

    manifest_path = INPUTS_DIR / "case_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Write FILE_RENAMING_INDEX.md
    index_lines = [
        "# Workspace File Renaming Index & Clinical Inventory",
        "",
        f"**Patient:** {manifest['patient_name']} | **Verified DOB:** 1963-04-09 (National ID `26304090300188`; MRI DICOM header lists `27-Mar-1963`)",
        f"**Generated:** {manifest['generated_at']} | **Total Raw Local Files Organized & Checksum-Verified:** {len(manifest_entries)}",
        "",
        "## 1. Local Workspace Raw Files Mapping Table",
        "",
        "| # | Original Raw Filename | New Semantic Filename | Folder Path | Date of Test | Size (Bytes) | One-Line Clinical Summary |",
        "|---|---|---|---|---|---:|---|",
    ]

    for idx, entry in enumerate(manifest_entries, start=1):
        index_lines.append(
            f"| {idx} | `{entry['original_raw_filename']}` | `{entry['new_semantic_filename']}` | `{entry['folder_path']}` | {entry['date_of_test']} | {entry['size_bytes']:,} | {entry['one_line_clinical_summary']} |"
        )

    index_lines.extend(
        [
            "",
            "## 2. Historical Narrative Records Without Raw Local Scans (Retained for Timeline Continuity)",
            "",
            "> Note: Folders `03_CT_Brain_Scans`, `04_Audiology_PTA_and_Tympanometry/2026_11_03_FollowUp`, and `07_Neurophysiology_Prolonged_EEG` are now populated with raw primary scans (`37` total raw files organized). The `2026-11-03` Masked PTA & GSI-39 Tympanometry scans serve as the **definitive audiology baseline**, replacing the unmasked `2024-11-02` baseline narrative below (retained solely for longitudinal comparison).",
            "",
            "| Category | Date of Test | Modality / Source | Workspace Status | Clinical Summary |",
            "|---|---|---|---|---|",
        ]
    )

    for rec in CLINICAL_RECORDS_FROM_HISTORY_ONLY:
        index_lines.append(
            f"| {rec['category']} | {rec['test_date']} | {rec['modality']} | {rec['local_file_status']} | {rec['summary']} |"
        )

    index_path = INPUTS_DIR / "FILE_RENAMING_INDEX.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    return manifest


if __name__ == "__main__":
    result = run_organization()
    print(f"Successfully organized {result['total_raw_files_organized']} raw files.")
    print(f"Manifest written to: {INPUTS_DIR / 'case_manifest.json'}")
    print(f"Index written to: {INPUTS_DIR / 'FILE_RENAMING_INDEX.md'}")
