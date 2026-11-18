# Dataset setup (Windows)

The uploaded course archive was a ZIP containing a nested RAR archive. Both layers must be extracted before Python can read the images.

## 1. Extract both archive layers

1. Install 7-Zip from https://www.7-zip.org/ or use WinRAR.
2. Right-click `5y9wdsg2zt-2.zip` and choose **7-Zip → Extract to "5y9wdsg2zt-2\\"**.
3. Open the extracted folder. It should contain `Concrete Crack Images for Classification.rar`.
4. Right-click that RAR and choose **7-Zip → Extract to "Concrete Crack Images for Classification\\"**.
5. Locate the folders named `Negative` and `Positive`. They may be inside one more enclosing folder.

## 2. Arrange the folders

From the repository root, create:

```
data/
  concrete/
    Negative/
    Positive/
```

Move/copy all negative-class images into `data/concrete/Negative/` and all positive-class images into `data/concrete/Positive/`. Keep the original class labels unchanged. Do not put the RAR/ZIP files inside these folders.

## 3. Check the image count

Open PowerShell in the repository root and run:

```powershell
(Get-ChildItem data/concrete/Negative -File -Recurse).Count
(Get-ChildItem data/concrete/Positive -File -Recurse).Count
```

For the full CCIC v2 dataset, the expected published count is 20,000 images per class (40,000 total). If your extracted archive has fewer, check that both archive layers were fully extracted and that you have the complete dataset.

## 4. Install and train

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py --data-dir data/concrete --epochs 10 --batch-size 32
python src/evaluate.py --data-dir data/concrete --checkpoint outputs/best_model.pth
```

If PowerShell blocks activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 5. Save the real results

After evaluation, copy the printed metrics and the contents of `outputs/test_metrics.json` into the report and presentation. Do not use example or previously generated figures. If training is interrupted, report it as incomplete and do not claim a final test score.

## Important scope note

CCIC has only image-level Crack/No-Crack labels. It cannot train a pixel-level crack segmentation model or a supervised Low/Moderate/Severe classifier. Those extensions need separate pixel masks, expert-reviewed severity labels, and calibrated image capture. The prototype is not an engineering safety assessment.
