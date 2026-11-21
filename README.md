# Parameter-Efficient Adaptation of DINOv2-Small for Dental Radiograph Classification

Code, split files and analysis notebooks for the paper

> **Parameter-Efficient Adaptation of DINOv2-Small for Dental Radiograph Classification: A Comparative Study of Fine-Tuning Strategies, Data Efficiency, and Interpretability**
> Sabbir Sharif, Fateha Hossain Anushka, MD Sazzadur Ahamed — Daffodil International University
> *[EDIT: journal / arXiv link and year once available]*

The study compares five adaptation strategies for DINOv2-Small (linear probe, VPT-Shallow, LoRA on attention, LoRA on attention + MLP, bottleneck adapters), full fine-tuning of DINOv2-Small, and a fully fine-tuned ResNet-50 on four-class single-tooth periapical radiograph classification (Caries, Crowned, Root Canal, Healthy). Full fine-tuning, adapters, LoRA (attention + MLP) and ResNet-50 are also retrained on 50% and 20% of the training images. Linear probes on RAD-DINO and BiomedCLIP are used to check the backbone choice. The analysis covers per-class results, patient-level paired bootstrap tests with Holm correction, calibration and ensemble Grad-CAM.

---

## 1. Repository contents

> **[EDIT]** Adjust the file names to the ones in this repository.

| Path | Purpose |
|---|---|
| `splits/train_split.csv`, `val_split.csv`, `test_split.csv` | Patient-level partition used for every experiment |
| `splits/train_split_50pct.csv`, `train_split_20pct.csv` | Reduced training subsets (class-wise sampling, `random_state=2025`) |
| `notebooks/patient_level_split.ipynb` | Creates the patient-level split (seed 42) and verifies it |
| `notebooks/train_*.ipynb` | Training and test evaluation, one notebook per method (see Section 5) |
| `notebooks/patient_level_comparison_all_models.ipynb` | Patient-level paired bootstrap, sign-flip test, Holm correction |
| `notebooks/calibration_analysis*.ipynb` | ECE, Brier score, mean confidence, reliability diagrams |
| `notebooks/gradcam_ensemble.ipynb` | Ensemble Grad-CAM figure |
| `results/` | Saved CSV outputs (per-seed metrics, training curves, test predictions) |

The images are **not** included. They are available from the DentIRO dataset repository (see Section 2).

---

## 2. Data

- **Dataset:** DentIRO, 5,300 single-tooth periapical radiographs from 3,243 patients at two clinics. Shoib *et al.*, DOI: `10.6084/m9.figshare.32086377.v1`.
- Download the dataset and place the images so that the paths in the `image_path` column of the split files resolve.
- **[EDIT]** Note any path change that users have to make (for example the Drive folder used in the notebooks: `/content/drive/MyDrive/DentIRO`).

### Patient-level partition

- The 3,243 unique patient identifiers are sorted and split with scikit-learn `train_test_split` (random seed 42): 60% of the patients for training, and the remaining 40% divided equally into validation and test.
- Result: **train 3,200 images / 1,945 patients, validation 1,074 / 649, test 1,026 / 649**. All images of a patient belong to the same set. The split is random and not stratified by class.
- `notebooks/patient_level_split.ipynb` creates the files and checks that no patient occurs in more than one set.
- **Do not regenerate the split after experiments have started.** Use the provided CSV files for every model and every training fraction.

---

## 3. Environment

Experiments were run in Google Colab on a T4 GPU (16 GB).

| Package | Version |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| torchao | 0.18.0 |
| PEFT | 0.21.2 |
| scikit-learn | 1.6.1 |
| NumPy | 2.1.3 |
| SciPy | 1.16.3 |
| pandas | 2.2.3 |

The DINOv2-Small weights are loaded with `torch.hub` (`facebookresearch/dinov2`, `dinov2_vits14`). ResNet-50 uses torchvision `IMAGENET1K_V2` weights. RAD-DINO: `microsoft/rad-dino`. BiomedCLIP: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.

```bash
pip install torch torchvision peft scikit-learn numpy scipy pandas matplotlib open_clip_torch transformers
```
*(Install the versions above to reproduce the reported numbers as closely as possible.)*

---

## 4. Experimental setup (summary)

| Item | Setting |
|---|---|
| Input | 224 × 224, gray channel replicated to 3 channels, ImageNet normalization |
| Augmentation (training only) | Random horizontal flip (p = 0.5), random rotation within ±10° |
| Loss | Weighted cross-entropy with label smoothing 0.1; inverse-frequency class weights recomputed for each training subset |
| Optimizer | AdamW, weight decay 0.01, batch size 32, no gradient clipping, no mixed precision |
| Schedule | Linear warm-up over 10% of the steps, then cosine decay to zero, defined over 30 epochs |
| Stopping | At most 30 epochs; early stopping on validation macro-F1 (patience 7); the checkpoint with the best validation macro-F1 is used for the test set |
| Seeds | 42, 123, 2024 for every configuration |
| Head | Dropout 0.1 followed by one linear layer (384 → 4 for DINOv2-Small, 2048 → 4 for ResNet-50) |

**Learning rates** (chosen from common defaults and not tuned per method):

| Method | Module LR | Head LR |
|---|---|---|
| DINOv2-S full fine-tune | 1e-5 | 1e-3 |
| ResNet-50 (full FT) | 1e-4 | 1e-3 |
| LoRA (attention; attention + MLP), r = 16, α = 32, dropout 0.1 | 1e-4 | 1e-3 |
| Bottleneck adapters (width 64, dropout 0.1) | 1e-3 | 1e-3 |
| VPT-Shallow (10 prompt tokens) | 1e-3 | 1e-3 |
| Linear probes (DINOv2-S, RAD-DINO, BiomedCLIP) | – | 1e-3 |

BiomedCLIP probe: embeddings computed once from non-augmented images and cached. Full details are in Section 3 of the paper.

**Reported quantities.** Accuracy, per-class results, paired tests and calibration use the three-seed ensemble (mean of the three softmax outputs). Macro-F1, balanced accuracy and the other overall metrics in Table 5 are the mean ± SD over the three single-seed runs.

---

## 5. How to run

> **[EDIT]** Replace the `train_*.ipynb` names with the actual notebook names.

1. **Split** — run `patient_level_split.ipynb` only if you want to re-create the split (it writes to a new folder and does not overwrite existing files). Otherwise use the provided files in `splits/`.
2. **Train and evaluate (100% data)** — one notebook per method, for the three seeds. Each notebook saves, per method: per-epoch curves (`training_curve_seed*.csv`), the best weights (`*_weights_seed*.pth`) and `test_evaluation_data_multiseed.npz` (keys include `y_true`, `avg_probs` and `seed_<seed>_probs`).
3. **Reduced data (50%, 20%)** — run the notebooks for full fine-tuning, adapters, LoRA (attention + MLP) and ResNet-50 with `train_split_50pct.csv` and `train_split_20pct.csv`. Outputs go to folders with the suffix `_50pct` / `_20pct`.
4. **Statistics** — `patient_level_comparison_all_models.ipynb` reads the `.npz` files and writes the patient-level bootstrap (10,000 resamples, seed 42), the sign-flip test and the Holm-adjusted results to CSV and LaTeX files.
5. **Calibration** — `calibration_analysis*.ipynb` computes ECE (15 equal-width bins), Brier score, mean confidence and the reliability diagrams.
6. **Grad-CAM** — `gradcam_ensemble.ipynb` loads the saved weights of the three seeds and produces the ensemble Grad-CAM figure.

Training time and GPU memory were not recorded, so no timing claims are made in the paper.

---

## 6. Mapping from paper to outputs

| Paper item | Produced by |
|---|---|
| Table 3 (parameters, model size) | training notebooks (parameter counts, checkpoint sizes) |
| Tables 5–7 (overall and per-class results, Caries counts) | training notebooks and `.npz` files |
| Table 8 (reduced data) | training notebooks for 50% and 20% |
| Table 9 and reliability diagrams (calibration) | `calibration_analysis*.ipynb` |
| Section 4.6 (statistics) | `patient_level_comparison_all_models.ipynb` |
| Figure 3 (Grad-CAM) | `gradcam_ensemble.ipynb` |

---

## 7. Notes

- Bootstrap intervals describe test-set sampling variability and do not include variability between training seeds.
- The results come from one dataset, one fixed patient-level split (100 Caries test images) and three seeds. Grad-CAM maps are descriptive and were not reviewed by clinicians.
- Non-deterministic GPU operations may cause small numerical differences even with the same seeds.

## 8. Citation

<!-- *[EDIT: add the BibTeX entry when the paper is published.]* -->

<!-- ```bibtex
@article{sharif2026dinov2dental,
  title   = {Parameter-Efficient Adaptation of DINOv2-Small for Dental Radiograph Classification: A Comparative Study of Fine-Tuning Strategies, Data Efficiency, and Interpretability},
  author  = {Sharif, Sabbir and Anushka, Fateha Hossain and Ahamed, MD Sazzadur},
  year    = {2026}
}
``` -->

## 9. License

*[EDIT: add a license, for example MIT for the code. The DentIRO images follow the license of the dataset repository.]*
