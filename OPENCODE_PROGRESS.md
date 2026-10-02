# OPENCODE_PROGRESS.md

Progress log for the **AI Waste Classifier** project.
**Always read this file first when starting a new session, then resume from
"Next step" at the bottom.**

Project root: `C:\Python files\waste classifier`
Last updated: after smoke test (training not yet started)

---

## Current phase

**PHASE 4 — ready to launch full training run.** Phases 1–3 are complete.
Training has NOT been started yet. It requires explicit go-ahead.

---

## Completed work

| Phase | Status | Notes |
|---|---|---|
| 1. State inspection / dataset + model selection | DONE | Recovered a crashed session |
| 2. Dataset download | DONE | 19,762 images, 753 MB, verified |
| 2. Data preparation | DONE | 7,004 / 1,502 / 1,502 split |
| 3. Training pipeline code | DONE | prepare_data, dataset, models, train |
| 3. Smoke test | PASS | all 7 checks green |
| 4. Full training | **NOT STARTED** | ~65 min expected |
| 5. Evaluation / ONNX export | NOT STARTED | `evaluate.py`, `export_model.py` not yet written |
| 6. FastAPI backend | NOT STARTED | empty dir tree only |
| 7. React frontend | NOT STARTED | empty dir tree only |
| 8. Tests | NOT STARTED | |
| 9. Render + Vercel config | NOT STARTED | |
| 10. README | NOT STARTED | |

---

## Key decisions

- **Dataset (primary):** *Garbage Classification Dataset* by Suman Kunwar
  (D.Waste.app) — `omasteam/waste-garbage-management-dataset`, **MIT**,
  19,762 images, 10 folders. Academic reference: "Managing Household Waste
  Through Transfer Learning".
- **Dataset (supplement):** *WasteSnap Dataset* —
  `Kishore2412/Multi-Class-Waste-Image-Classification-Dataset`, **CC-BY-2.5**.
  Used **only** to fill the `E-Waste` class (100 images), which the primary
  dataset lacks.
- **Taxonomy (10 canonical classes):** Cardboard, Paper, Plastic, Glass, Metal,
  Organic, Textile, E-Waste, Battery, Other/Unknown.
  - `clothes` + `shoes` → **Textile**
  - `biological` → **Organic**, `trash` → **Other/Unknown**
- **Backbone:** `mobilenet_v3_small` (1,528,106 params). Benchmarked on this
  4-thread machine: **64 ms/img train**, **13 ms/img inference**. EfficientNet-B0
  (353 ms) and ResNet18 were too slow; MobileNetV3-Large is 175 ms.
- **Sampling:** capped at 1,200 images/class to fix the 7,304-vs-1,020 imbalance
  and keep CPU training time reasonable. E-Waste is uncapped (only 100 images).
- **Class index order** is fixed by `CANONICAL_CLASSES`, never by alphabetical
  directory listing, so the ONNX output layer always matches `class_names.json`.
- **`Other/Unknown` on disk is `Other-Unknown`.** The display name contains a
  `/`, which silently created a nested directory. Fixed with
  `config.class_dir_name()`. Do not "simplify" this back.

---

## Environment

- Python 3.12.3, Node v24.21.0, npm 11.19.0, git 2.55.0
- venv at `.venv/` (repo root) — **complete, do not recreate**:
  torch 2.5.1+cpu, torchvision 0.20.1+cpu, onnx 1.17.0, onnxruntime 1.19.2,
  onnxslim, fastapi 0.115.6, uvicorn 0.32.1, pillow 11.0.0, numpy 1.26.4,
  scikit-learn 1.5.2, matplotlib, pytest 8.3.4, httpx, pydantic 2.10.3,
  pydantic-settings, python-multipart, python-dotenv, huggingface_hub 0.27.0
- **Pretrained ImageNet weights are cached locally** at
  `C:\Users\ABCD\.cache\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`
  (10,306,551 bytes). No network fetch is needed. Do not disable SSL anywhere.

---

## Datasets (do not delete, do not re-download)

Raw, in `%TEMP%\opencode\` (outside the repo, gitignored):

- `garbage_dataset\` — 19,762 images, 753 MB, 10 class folders, COMPLETE
- `wastesnap\extracted\WasteSnap_Dataset\` — 770 images, 9 folders

Prepared copy in the repo (gitignored): `data\prepared\`
`{train,val,test}/{class}/*.jpg` + `manifest.json` (records every file, its
source and its split). Rebuild with
`python training/prepare_data.py --force`.

Per-class prepared counts:

| Class | train | val | test |
|---|---|---|---|
| Cardboard | 840 | 180 | 180 |
| Paper | 840 | 180 | 180 |
| Plastic | 840 | 180 | 180 |
| Glass | 840 | 180 | 180 |
| Metal | 714 | 153 | 153 |
| Organic | 697 | 150 | 150 |
| Textile | 840 | 180 | 180 |
| E-Waste | 70 | 15 | 15 |
| Battery | 660 | 142 | 142 |
| Other-Unknown | 663 | 142 | 142 |
| **Total** | **7004** | **1502** | **1502** |

---

## Files created so far

```
.gitignore
training/__init__.py
training/config.py        # paths, CANONICAL_CLASSES, source maps, TrainingConfig
training/prepare_data.py  # raw -> data/prepared + manifest.json
training/dataset.py       # transforms, WasteImageDataset, class weights
training/models.py        # build_model() shared by train/evaluate/export
training/train.py         # fine-tune loop, early stopping, checkpointing
```

Empty directories that exist but have no code yet:
`backend/app/{model,api,utils,data}`, `backend/models`, `artifacts/{metrics,checkpoints}`.

---

## Smoke test results (Windows-safe version)

Script: `%TEMP%\opencode\smoke_test.py` — has an `if __name__ == "__main__"`
guard and uses `num_workers=0` so DataLoader spawn workers cannot re-execute it.

| Check | Result |
|---|---|
| dataset loading | **PASS** (train=7004, val=1502, labels 0–9) |
| DataLoader | **PASS** (batch 8×3×224×224) |
| model creation | **PASS** (mobilenet_v3_small, 1,528,106 params) |
| forward pass | **PASS** (logits 8×10, loss 2.4386) |
| backward pass | **PASS** (grad_sum 2973.5) |
| evaluation | **PASS** (acc 0.156, macro-F1 0.119, confusion 10×10) |
| pretrained weights | **AVAILABLE** (local cache, no download) |
| certificate error | **RESOLVED** — no network call was ever needed |

The earlier 15-minute "timeout" was **not** a certificate error: the unguarded
smoke script was re-executed by every DataLoader worker (Windows uses `spawn`),
spawning processes forever. Any new ad-hoc script touching a DataLoader must
have a `__main__` guard and `num_workers=0`.

Observed class weights: `[0.456 ×6, 0.537, 0.550, 0.456, 5.474, 0.581, 0.578]`
— E-Waste (index 7) is weighted ~10× higher. Watch the confusion matrix for
E-Waste over-prediction once training finishes.

---

## Current errors

None outstanding. Two issues were found and fixed:
1. Dataset download was interrupted (Wi-Fi loss) leaving the entire `trash`
   class missing — **fixed**, download resumed from HF cache and verified.
2. `Other/Unknown` created a nested `train/Other/Unknown/` directory — **fixed**
   with `class_dir_name()`; data regenerated (`MANIFEST_VERSION = 3`).

---

## API / port notes

The old `%TEMP%\opencode\uvicorn_*.log` files are from a **different, older
project** (port **8765**, routes `/api/predict`, `/static/styles.css`). They are
**not** this project. This project expects:

- Backend: `http://localhost:8000`
- Frontend (Vite): `http://localhost:5173`
- Swagger: `http://localhost:8000/docs`

Any "cannot connect to API" message should be checked against port **8000**.

---

## Next step

Awaiting go-ahead to start the full training run:

```powershell
& ".\.venv\Scripts\python.exe" training\train.py --epochs 8
```

Expect ~8 min/epoch × 8 = ~65 min. The production `TrainingConfig` uses
`num_workers=4`, which is safe because `train.py` has a `__main__` guard.
Runs will be slow; run it in the background and poll
`artifacts/checkpoints/best_model.pth` + `artifacts/metrics/training_history.json`.

After training:
1. Write `training/evaluate.py` (accuracy, per-class precision/recall/F1,
   confusion matrix, classification report → `artifacts/metrics/`)
2. Write `training/export_model.py` (ONNX → `backend/models/waste_classifier.onnx`
   + `backend/models/class_names.json`)
3. Verify ONNX output matches PyTorch to within 1e-4
4. Then build the FastAPI backend, then the React frontend

## Deployment status

Render: not configured. Vercel: not configured. Git: repo initialised on `main`,
**zero commits**, `.venv/` still untracked (needs `.gitignore` verification —
`.gitignore` is written but nothing has been committed yet).
