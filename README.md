# Neurovascular & Neuro-Otological MDT Clinical Case Study Dashboard

Multidisciplinary Medical Board (6 Consultants) audit, biomedical signal/radiology quantification pipeline, shareable fellowship PDF dossiers, and interactive HTML5 dashboard (**including embedded Neuroimaging Video Suite**) for patient **Amal Mokhtar Mohamed Ibrahim** (Age 63.5).

## Quick Launch Links
- **Hosted Web Entry:** [index.html](./index.html) -> auto-redirects to [`06_Clinical_Dashboard/index.html`](./06_Clinical_Dashboard/index.html)
- **Interactive Dashboard:** [`06_Clinical_Dashboard/index.html`](./06_Clinical_Dashboard/index.html)
- **2-Page Handover Brief (PDF):** [`05_Shareable_PDF_Reports/01_MDT_Executive_Summary_and_Action_Plan.pdf`](./05_Shareable_PDF_Reports/01_MDT_Executive_Summary_and_Action_Plan.pdf)
- **6-Consultant Compendium (PDF):** [`05_Shareable_PDF_Reports/02_Consultant_6_Board_Audit_Reports.pdf`](./05_Shareable_PDF_Reports/02_Consultant_6_Board_Audit_Reports.pdf)
- **Complete Case Monograph (PDF):** [`05_Shareable_PDF_Reports/03_Patient_Case_Study_Full_Report.pdf`](./05_Shareable_PDF_Reports/03_Patient_Case_Study_Full_Report.pdf)
- **Master Markdown Report (EN + Egyptian AR):** [`04_Final_MDT_Deliverables/Master_MDT_Case_Study_Report.md`](./04_Final_MDT_Deliverables/Master_MDT_Case_Study_Report.md)

## Repository Structure
| Directory | Purpose |
|---|---|
| `01_Structured_and_Renamed_Inputs/` | All **37 semantically renamed clinical inputs**, `case_manifest.json`, `FILE_RENAMING_INDEX.md`, CLAHE ROI plates (`ROI1a`–`ROI5b`), and H.264 streaming MP4s in `01_Neuroimaging_MRI_MRA_MRV_Videos/` |
| `02_Custom_Skills_and_Scripts/` | Reproducible biomedical pipeline scripts (`skill_semantic_organizer_and_renamer.py`, `skill_video_frame_extractor.py`, `skill_radiology_enhancer.py`, `skill_clinical_quantifier.py`, `skill_pdf_generator.py`, `skill_dashboard_builder.py`) |
| `03_Individual_Consultant_Audits/` | Independent audit reports from all 6 specialist consultants |
| `04_Final_MDT_Deliverables/` | Consolidated bilingual MDT fellowship report |
| `05_Shareable_PDF_Reports/` | Publication-grade A4 PDF deliverables |
| `06_Clinical_Dashboard/` | Standalone responsive clinical command dashboard (`index.html`) |

## Neuroimaging Video Suite (MRI, MRA, MRV)
The top hero section of `06_Clinical_Dashboard/index.html` embeds 3 responsive HTML5 `<video controls preload="metadata">` players bound via relative paths:
1. **3D-MRV Venography Video** (`MRV_2026-11-05_Video03_DerivedCine_3D_PCA_Attenuated_Lt_Transverse_Sigmoid_Sinus.mp4` — derived 18-projection rotational cine from primary `Film06` showing Right-dominant drainage and `0.48` L/R attenuated Lt transverse/sigmoid/IJV without thrombosis).
2. **3D-TOF MRA Video** (`MRA_2026-11-05_Video01_3D_TOF_Axial_Source_Slices_Sc11_T1FFE_LtEyeSiliconeOil_WebH264.mp4` — lossless faststart H.264 stream of raw Sc 11 1.0-mm slices showing patent intracranial arteries & `2.59x` Lt intraocular silicone oil).
3. **Axial T2/FLAIR MRI Video** (`MRI_2026-11-05_Video02_DerivedCine_Axial_T2_FLAIR_BasalGanglia_CSVD.mp4` — synchronized 20-level T2/FLAIR sweep from primary `Film09 + Film07` showing Rt basal ganglia chronic lacunar infarct & Fazekas 1–2 CSVD).

> *Note on `.gitignore`:* Intermediate regenerable frame dumps (`Video01_Source_Slices/`, `Extracted_Pages/`, `Full_Sheets_CLAHE/`) and raw hash duplicates (`00_Raw_Backup_Archive/`) are omitted from Git to keep repository & static-site deployment lean (~140 MB); run `python 02_Custom_Skills_and_Scripts/skill_video_frame_extractor.py` locally if you wish to regenerate the 513 individual PNG slices.
