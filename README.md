# Multistage Schizophrenia Detection using CNN and GNN

Live demo: https://varshaaaaaaaaaaa-schizodetect.hf.space/ (hosted on a Hugging Face Space)

A deep learning pipeline that looks at structural brain MRI scans and estimates how severe a person's schizophrenia symptoms are. Instead of the usual two-way answer (patient or healthy), it sorts people into four groups: Healthy, Early, Moderate and Chronic.

This is a research prototype, not a medical tool. The four groups are not clinical stages. I made them by splitting patients into thirds based on their SANS/SAPS symptom scores, so they only mean something inside this dataset. Please don't upload real patient scans to the demo.

## About schizophrenia and the idea behind this

Schizophrenia is a long-term brain disorder that affects how a person thinks, perceives things and feels. Doctors usually describe the symptoms in two groups. Positive symptoms are things added to a person's experience, like hallucinations, delusions and disorganized thinking. Negative symptoms are things taken away, like low motivation, flat emotion and withdrawing from other people. The SAPS and SANS scales are standard checklists clinicians use to rate each group.

Brain MRI studies have found small average differences between people with schizophrenia and healthy people, such as enlarged fluid-filled spaces in the brain and less gray matter in some regions. These differences overlap a lot from person to person, so an MRI alone can't diagnose it. A lot of existing machine learning work just predicts patient versus healthy. Severity matters too, since symptoms change over time and doctors need to track them, so I wanted to see how far a model could go in estimating severity from structural MRI.

## How it works

1. Take a T1 brain MRI, crop and normalize it, and resize it to 64x64x64.
2. Cut out 45 2D slices and run them through a ResNet-18 (pretrained on ImageNet, fine-tuned on this data). It gives a patient-versus-healthy probability and a 512-number summary of each scan, averaged over the slices.
3. Combine that summary with age, sex and some simple brain-region and fMRI connectivity features.
4. Build a graph where every person is a node connected to their 5 most similar people, and train a graph attention network (GAT) to predict one of the four groups.
5. A Flask dashboard shows the cohort graph and lets you upload a scan to see a prediction.

## What each file does

- `01_stage_labeling.py` reads the participant tables and symptom scores from both datasets and keeps only people who have a T1 MRI. Healthy controls get the Healthy label. For patients it adds up the SANS/SAPS ratings, turns that into a percentage, and splits patients into thirds to label them Early, Moderate or Chronic. It also balances the number of people in each group.
- `02_preprocess_npy_3d.py` loads each MRI, crops away the empty background, clips extreme values, normalizes the intensities, and resizes the scan to 64x64x64. It then makes the 80/20 train and test split and saves everything as NumPy files.
- `03_train_binary_cnn_embeddings.py` fine-tunes the ResNet-18 to tell patients from healthy controls using 45 slices per scan, then saves a probability and a 512-number embedding for every person. These embeddings are what the graph model uses later.
- `04_extract_structural_features.py` calculates simple numbers from each scan, like intensity and volume in different regions, left versus right differences, and a coarse grid over the brain.
- `05_extract_fmri_features.py` takes the fMRI scans, splits the brain into a grid, and measures how strongly different regions' activity moves together over time.
- `06_gnn_classifier.py` puts it all together. It builds the node features, connects each person to their 5 most similar people, and trains the graph attention network to predict the four groups. It saves the best model, the predictions and the result plots.
- `07_predict_patient.py` is a command-line demo. You enter the ID of someone already in the dataset and it prints the predicted group and probabilities.
- `08_cnn_evaluation.py` and `09_gnn_evaluation.py` make the confusion matrices, ROC curve and F1 charts for the CNN and the GNN.
- `app.py`, `templates/` and `static/` are the Flask web app: the cohort statistics, the population graph, and the scan upload with its slice viewer and results.
- `Outputs/` holds the saved models, predictions and plots. `docs/PROJECT_REPORT.md` is a longer write-up.

## Data

I combined two public datasets from OpenNeuro:

- **DS115**: "Working memory in healthy and schizophrenic individuals" (OpenNeuro ds000115)
- **UCLA CNP**: "UCLA Consortium for Neuropsychiatric Phenomics LA5c Study" (OpenNeuro ds000030)

After keeping people with a T1 scan and balancing the groups, I had 97 subjects (74 from UCLA and 23 from DS115). I split them 80/20 with a fixed random seed, giving 77 for training and 20 for testing, with the same mix of groups in both. The raw scans aren't in this repo. Download them from OpenNeuro and put them in `dataset_115/` and `dataset_ucla/` if you want to rerun everything.

## Results

On the 20 held-out test subjects:

- Patient versus healthy (CNN): 90% accuracy, 18 of 20 correct
- Four groups (GNN): 80% accuracy, 16 of 20 correct

In the GNN, Healthy and Early did best (F1 about 0.9) and Moderate and Chronic were harder to separate (F1 0.73 and 0.67). Plots are in `Outputs/`.

## Limitations

- The test set is only 20 people, so one person changes the accuracy by 5 points. These numbers are a rough signal, not proof.
- The graph uses age, sex and the same symptom score that defines the four groups, so some information about the answer ends up in the graph. It is also built using all subjects, including the test ones. Rebuilding the graph without the symptom score is the first thing I'd try next.
- Controls and patients come from the two datasets in different proportions, so the CNN may partly be picking up scanner or site differences.
- The structural features are simple region-based numbers, not real anatomical segmentation.

## Running it locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. The saved models and `Outputs/npy_dataset/fmri_features_4class.csv` are included, so the app runs without the raw data. To regenerate everything from the raw data, run scripts 01 to 06 in order, then 08 and 09.

There's also a `Dockerfile` if you want to host it somewhere.

## Notes

I used AI tools (ChatGPT and VS Code) to help write the code. The datasets come from OpenNeuro, and anything derived from them should follow their original licenses.
