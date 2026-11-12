# SchizoDetect: MRI-Based Schizophrenia Stage Classification

## Project overview

SchizoDetect is a research prototype for analyzing structural MRI and assigning
one of four cohort-defined labels: **Healthy**, **Early**, **Moderate**, or
**Chronic**. Its machine-learning pipeline combines:

1. A ResNet-18 convolutional neural network (CNN) that analyzes 2D slices of a
   3D T1-weighted MRI and produces a binary Healthy/Schizophrenia probability
   and a 512-value image embedding.
2. Hand-engineered structural MRI features and fMRI connectivity features.
3. A graph attention network (GAT) that uses the subject features as graph
   nodes and phenotype-based similarity to build graph edges.
4. A Flask website for cohort visualization and single-scan demonstration.

This is a **research/educational prototype, not a medical device**. Its four
stage labels are derived from symptom-score percentiles in the available
cohort. They are not independently established clinical diagnoses, disease
progression stages, or treatment recommendations.

## Data and labels

The code expects two local datasets:

- `dataset_115/`: DS115 participant metadata, structural MRI, and available
  letter-2-back fMRI.
- `dataset_ucla/`: UCLA participant metadata and phenotype tables, structural
  MRI, and available resting-state fMRI.

`01_stage_labeling.py` joins participant and symptom information to the
participants who have a T1 MRI. Controls are labeled `Healthy`. For patients,
the code sums selected SANS/SAPS ratings and normalizes the score to a
percentage. It then uses the 33rd and 66th percentiles of the MRI-available
patients' scores to assign `Early`, `Moderate`, and `Chronic` labels.

The script undersamples the healthy cohort toward the median patient-stage
count, preferring UCLA controls. These are cohort-relative proxy labels; the
same symptom scores help define the targets and later help define graph
connections.

The checked-in/generated metadata reports **97 subjects** in total. The
saved split has **77 training subjects** and **20 test subjects**:

| Stage | Train | Test | Total |
|---|---:|---:|---:|
| Healthy | 19 | 5 | 24 |
| Early | 19 | 5 | 24 |
| Moderate | 20 | 5 | 25 |
| Chronic | 19 | 5 | 24 |
| **Total** | **77** | **20** | **97** |

These are the counts in the saved `npy_metadata.csv` and CNN embedding CSV;
they may change if the labeling or preprocessing scripts are rerun against
different source data.

## Pipeline and code map

Run the numbered scripts in this order to regenerate the model pipeline:

### 1. `01_stage_labeling.py` — labels and cohort metadata

**Main functions:** `load_ds115()`, `load_ucla()`, `safe_sum()`,
`assign_stage()`, and `main()`.

Reads participants and SANS/SAPS TSV files, checks which participants have
structural MRI, assigns the four labels, balances the healthy group, and writes
`Outputs/metadata.csv`.

### 2. `02_preprocess_npy_3d.py` — structural MRI preprocessing

**Main functions:** `preprocess_mri()`, `crop_nonzero_brain()`,
`normalize_volume()`, `resize_volume()`, `save_subject()`, and `main()`.

Loads each T1 NIfTI, crops around non-background voxels, clips intensities at
the 1st/99th percentiles, min-max normalizes, and resizes to
`(1, 64, 64, 64)`. It creates a stratified 80/20 split using seed 42, saves
volumes under `Outputs/npy_dataset/{train,test}/<stage>/`, and writes
`Outputs/npy_dataset/npy_metadata.csv`.

### 3. `03_train_binary_cnn_embeddings.py` — binary CNN and embeddings

**Main components:** `MRISliceDataset`, `BinaryMRIModel()`,
`volume_to_slices()`, `evaluate_subject_level()`,
`extract_embeddings()`, and `main()`.

Converts each MRI volume into 45 central 2D slices (15 per axis), resizes them
to 224 by 224, and trains an ImageNet-initialized ResNet-18 for
Healthy-versus-Schizophrenia classification. The backbone's later layers are
fine-tuned. Training uses rotations, AdamW, cosine learning-rate scheduling,
early stopping, and the subject-level validation accuracy to select a
checkpoint. Slice probabilities and embeddings are averaged per subject.

Outputs:

- `Outputs/npy_dataset/best_binary_cnn_for_embeddings.pth`
- `Outputs/npy_dataset/cnn_embeddings_4class.csv`
- `Outputs/npy_dataset/binary_cnn_embedding_results.txt`

The embeddings are 512 values per participant; the CSV also contains binary
probabilities, demographics, split, and labels.

### 4. `04_extract_structural_features.py` — regional MRI features

**Main functions:** `safe_stats()`, `region_features()`,
`extract_features()`, and `main()`.

Computes intensity and volume proxies, left/right differences, coarse
frontal/temporal/parietal/occipital and central regions, hippocampus-like
proxies, and a 4-by-4-by-4 spatial grid. These are simple coordinate-based
proxies, **not** anatomical segmentation or validated measurements of
individual brain structures.

Output: `Outputs/npy_dataset/structural_features_4class.csv`.

### 5. `05_extract_fmri_features.py` — functional connectivity

**Main functions:** `get_func_file()`, `extract_region_timeseries()`,
`extract_fmri_features()`, and `main()`.

Loads up to 120 fMRI time points, resizes each volume to 32 cubed, calculates
time series over a 4-by-4-by-4 grid, and computes Pearson correlations among
the 64 regions. It produces global connectivity statistics, regional
connectivity summaries, and up to 500 edge values. Subjects without a matching
functional scan are skipped. The available DS115 task fMRI and UCLA resting
fMRI are different acquisition paradigms, so the resulting features should not
be assumed to be directly equivalent.

Output: `Outputs/npy_dataset/fmri_features_4class.csv`.

### 6. `06_gnn_classifier.py` — four-class GAT

**Main components:** `create_masks()`, `build_node_features()`,
`build_phenotypic_graph()`, `GenuineGNN`, `evaluate()`, and `main()`.

The node features concatenate:

- 512 CNN embedding values;
- CNN Healthy/Schizophrenia probabilities;
- age and encoded sex;
- up to 20 selected structural/fMRI values.

The feature scaling and supervised feature selection are fitted on training
subjects. The graph connects each subject to five nearest phenotype neighbors
using cosine similarity over age, sex, and normalized symptom score. The GAT
has three graph-attention layers, hidden dimension 64, dropout 0.5, class-
weighted cross-entropy, AdamW, cosine scheduling, and validation-based
checkpoint selection (up to 400 epochs).

The source comments describe clinical scores as excluded from node features.
That is true for `raw_score`/`norm_score` columns in the feature vector, but
`norm_score` is used to construct graph edges, while symptom scores are also
used to define the target stages. Therefore, the graph topology contains
target-related information. This should be disclosed as a methodological
limitation; the implementation should not be described as proving zero
target leakage or fully independent clinical prediction. Graph construction
also uses the full cohort, including held-out subjects, which is a
transductive evaluation setup.

Outputs:

- `Outputs/gnn_results/best_gnn_model.pth`
- `Outputs/gnn_results/gnn_predictions.csv`
- `Outputs/gnn_results/training_loss_curve.png`
- `Outputs/gnn_results/accuracy_curves.png`
- `Outputs/gnn_results/confusion_matrix.png`

### 7. `07_predict_patient.py` — command-line prediction demo

**Main components:** `load_data()`, `GenuineGNN`, `predict_patient()`, and
`main()`.

Loads the prepared feature CSV and saved GNN model, then prompts for an
existing participant ID and prints the predicted stage and class probabilities.
It is a prediction demo over participants already in the CSV; it does not
process a new NIfTI upload.

### 8. `08_cnn_evaluation.py` — binary CNN evaluation

Evaluates the saved CNN probabilities on rows marked `test`, reports accuracy
and classification metrics, and saves:

- `Outputs/cnn_results/cnn_confusion_matrix.png`
- `Outputs/cnn_results/cnn_roc_curve.png`
- `Outputs/cnn_results/cnn_probability_distribution.png`
- `Outputs/cnn_results/cnn_f1_scores.png`

### 9. `09_gnn_evaluation.py` — GNN evaluation

Reads the test predictions created by script 6, prints a classification
report, and saves the multiclass confusion matrix and per-stage F1 chart under
`Outputs/gnn_results/`.

### Web application

- `app.py`: Flask routes, MRI preprocessing, loads the CNN checkpoint at
  startup, performs upload inference, and serves the cohort APIs.
- `templates/index.html`: page structure and dashboard/analysis panels.
- `static/style.css`: page styling and responsive layout.
- `static/script.js`: tab navigation, NIfTI upload interaction, prediction
  request/results, MRI slice viewer, and D3 cohort graph.

Routes:

- `GET /`: serves the dashboard.
- `GET /api/cohort-stats`: cohort summary statistics.
- `GET /api/cohort-graph`: nodes and phenotype-similarity links for the graph.
- `POST /api/predict`: accepts a NIfTI scan plus age, sex, and symptom ratings;
  returns CNN/GNN probabilities, a predicted class, structural proxies,
  neighbor information, and generated MRI slices.

The web app needs both saved checkpoints and
`Outputs/npy_dataset/fmri_features_4class.csv`. The CNN checkpoint is loaded
when `app.py` starts; the GNN checkpoint is loaded during each prediction.

## Reported results

The following values are from the result artifacts currently present in the
project. Both test sets contain only 20 subjects (five per four-class stage,
or five healthy and 15 patient subjects for the binary task).

### Binary CNN

From `Outputs/npy_dataset/binary_cnn_embedding_results.txt`:

- Best validation accuracy: **87.50%** at epoch 11.
- Held-out test accuracy: **90.00% (18/20)**.
- Healthy: precision 0.80, recall 0.80, F1 0.80 (5 subjects).
- Schizophrenia: precision 0.93, recall 0.93, F1 0.93 (15 subjects).
- Confusion matrix, rows=true and columns=predicted, order Healthy /
  Schizophrenia: `[[4, 1], [1, 14]]`.

### Four-class GNN

The saved `Outputs/gnn_results/gnn_predictions.csv` was independently
summarized from its 20 rows:

- Held-out test accuracy: **80.00% (16/20)**.
- Healthy: precision 0.83, recall 1.00, F1 0.91 (5 subjects).
- Early: precision 1.00, recall 0.80, F1 0.89 (5 subjects).
- Moderate: precision 0.67, recall 0.80, F1 0.73 (5 subjects).
- Chronic: precision 0.75, recall 0.60, F1 0.67 (5 subjects).
- Macro-average F1: 0.80.
- Confusion matrix, rows=true and columns=predicted, order Healthy, Early,
  Moderate, Chronic:
  `[[5, 0, 0, 0], [1, 4, 0, 0], [0, 0, 4, 1], [0, 0, 2, 3]]`.

These are small, single-split results and do not establish clinical
performance, generalization to other hospitals/scanners, or diagnostic
validity. The CNN scores are binary; the GNN scores are four-class. They are
not interchangeable.

## Dependencies and running locally

The Python dependencies listed in `requirements.txt` cover the scientific
pipeline (NumPy, pandas, scikit-learn, nibabel, SciPy, PyTorch,
torchvision, PyTorch Geometric, and plotting utilities). The Flask app also
imports `flask` and `flask-cors`, which are currently **not listed** in
`requirements.txt`; install them separately for local web testing:

```powershell
python -m pip install -r requirements.txt
python -m pip install Flask flask-cors
```

To open the website on Windows when the prepared feature CSV and model files
are available:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then visit `http://127.0.0.1:5000`. Keep the terminal open while using the
site; `Ctrl+C` stops the server.

To regenerate all pipeline outputs from source data, run scripts 1 through 6
in order, then optionally scripts 8 and 9:

```powershell
.\.venv\Scripts\python.exe 01_stage_labeling.py
.\.venv\Scripts\python.exe 02_preprocess_npy_3d.py
.\.venv\Scripts\python.exe 03_train_binary_cnn_embeddings.py
.\.venv\Scripts\python.exe 04_extract_structural_features.py
.\.venv\Scripts\python.exe 05_extract_fmri_features.py
.\.venv\Scripts\python.exe 06_gnn_classifier.py
.\.venv\Scripts\python.exe 08_cnn_evaluation.py
.\.venv\Scripts\python.exe 09_gnn_evaluation.py
```

Script 7 is an optional interactive participant-ID prediction demo. Training
the CNN may download pretrained ResNet-18 weights and can take substantial
time; GPU availability can speed it up.

## Current workspace state and deployment notes

- The saved CNN/GNN model weights, binary CNN result text, 97-row NPY metadata,
  CNN embeddings, and 20 GNN test predictions are present in the inspected
  workspace.
- The intermediate files
  `Outputs/npy_dataset/structural_features_4class.csv` and
  `Outputs/npy_dataset/fmri_features_4class.csv` were **not present** at the
  time this report was prepared. Consequently, the saved web app cannot
  currently serve cohort statistics or make predictions until these are
  generated (and the correct final CSV is made available). Existing model
  weights alone are not sufficient.
- Several scripts spell the output directory as lowercase `outputs`, while
  other scripts and the app use `Outputs`. This works on typical Windows
  filesystems but can fail on case-sensitive Linux deployment. Normalize paths
  consistently before deploying.
- `requirements.txt` does not include Flask/Flask-CORS; add them to the
  deployment dependency list if deploying the website.
- The application has not been packaged here with a production WSGI server,
  upload-size limits, or deployment-specific storage configuration. Do not
  expose the development Flask server directly to the public internet.
- The dataset and model files can be large and may have separate usage terms.
  Before publishing or deploying, verify dataset redistribution permissions
  and avoid committing identifiable or sensitive scans, uploads, or generated
  scan slices.

## Technology summary

| Area | Tools |
|---|---|
| Data handling | Python, NumPy, pandas |
| MRI/fMRI | nibabel, SciPy |
| CNN | PyTorch, torchvision ResNet-18 |
| Graph learning | PyTorch Geometric `GATConv` |
| Preprocessing/metrics | scikit-learn |
| Charts | Matplotlib, Seaborn |
| Web backend | Flask, Flask-CORS |
| Web frontend | HTML, CSS, JavaScript, D3.js |
