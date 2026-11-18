# Concrete Crack Detection and Severity Estimation for Campus Infrastructure

A deep-learning prototype for detecting visible cracks in concrete surface images and estimating a preliminary image-based crack-coverage indicator.

## Team

| Name | Roll No. |
|---|---:|
| Shashwat Sinha | 43 |
| Uzayr Sheikh | 56 |
| Suhani Mishra | 48 |
| Zaid Shaikh | 63 |

**Group:** 14 (V-D)  
**Department:** Electronics Engineering, Ramdeobaba University, Nagpur, Maharashtra, India  
**Academic session:** 2026–27

## Dataset

Use the **Concrete Crack Images for Classification** dataset. After extracting the downloaded archives, arrange images as:

```
data/concrete/
  Negative/
  Positive/
```

Do not upload the dataset archive or extracted images to this repository. Keep the data locally.

## Important scope note

The commonly used dataset provides binary crack / negative labels, not expert-annotated severity grades. This implementation therefore classifies crack presence and calculates a rough image-processing crack-coverage estimate. It is **not** a validated structural severity or safety rating and must not replace inspection by a qualified civil/structural engineer.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python src/train.py --data-dir data/concrete --epochs 10 --batch-size 32
python src/evaluate.py --data-dir data/concrete --checkpoint outputs/best_model.pth
```

The scripts create reproducible train/validation/test splits and report metrics from the actual test run. Do not report unmeasured accuracy, F1-score, or AUC values.

## Run the demo

```bash
streamlit run app.py
```

## Repository structure

```
.
├── app.py
├── requirements.txt
├── .gitignore
└── src/
    ├── model.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## Reproducibility checklist

Record the dataset source/version, split seed, split proportions, image size, batch size, learning rate, optimizer, epochs, hardware, and actual test metrics in the final report.
