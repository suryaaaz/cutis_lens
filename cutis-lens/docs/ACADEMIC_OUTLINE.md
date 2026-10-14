# Cutis Lens — Project report outline (fill the bracketed parts with YOUR measured results)

1. **Abstract** — Cutis Lens grades acne severity from facial images using an ImageNet-pretrained EfficientNet-B0 trained on ACNE04 and explains predictions with Grad-CAM. On a held-out test set the model achieved [accuracy / macro-F1 / QWK from outputs/evaluation_summary.json].
2. **Introduction & problem statement** — acne prevalence and manual counting burden (Zhang et al., 2024); black-box models are hard to trust clinically (Noronha et al., 2023).
3. **Existing systems** — YOLO-based lesion detection (Zhang et al.), CNN ensembles for multi-disease classification (Balasundaram et al.), limited-data strategies (Mohanty et al.). Gap: little emphasis on explaining predictions to the user.
4. **Proposed system & objectives** — upload → validate → classify → explain → report; privacy-preserving, locally runnable.
5. **Scope** — image-level severity, single dataset, educational use only.
6. **Literature survey** — use the comparison table in README.md.
7. **System architecture** — Mermaid diagrams in README.md (architecture, training workflow); add an inference + Grad-CAM sequence if required by your department.
8. **Modules** — dataset.py, preprocessing.py, model.py, train.py, evaluate.py, predict.py, explainability.py, report.py, app.py.
9. **Dataset** — ACNE04 (Wu et al., ICCV 2019); [count of images/classes from inspect_dataset.py]; label source and verification; split seed and sizes from splits.json; licence note.
10. **Methodology & model** — transfer learning, class-weighted CE, AdamW + cosine schedule, early stopping on validation macro-F1, mild augmentation. Justify EfficientNet-B0: small, strong ImageNet baseline, CPU-friendly; compare with MobileNetV3 if you run it.
11. **XAI methodology** — Grad-CAM on the last conv block: α_k = mean gradient of class score w.r.t. feature map k; CAM = ReLU(Σ α_k A_k); upsample and overlay. Limits: coarse, model-centric, not proof of lesions.
12. **Hardware/software** — Python 3.10/3.11, PyTorch, Flask, ReportLab, OpenCV; CPU or GPU.
13. **Implementation & testing** — pytest suite (counts from your run); distinguish mock-model tests from trained-model tests.
14. **Results** — insert training_curves.png, confusion_matrix.png, classification_report.txt. Report only measured numbers; discuss errors between adjacent severity levels.
15. **Limitations, future work** — lesion detection stage (YOLO) using ACNE04 boxes, calibration, skin-tone diversity, ensembles (Balasundaram et al.), Grad-CAM++ / faithfulness tests.
16. **Conclusion; References** (see README.md; verify all citation details against the PDFs).
