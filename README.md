# CIFAR-10 Image Classifier

A simple convolutional neural network (CNN) that trains on the CIFAR-10 dataset and classifies images into one of 10 categories: **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**.

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | Builds, trains, evaluates, and saves the model |
| `test.py` | Loads the saved model and classifies a new image |

---

## Requirements

Install dependencies before running either script:

```bash
pip install tensorflow numpy matplotlib pillow
```

---

## Usage

### Step 1 — Train the model

Run this once. Training takes a few minutes depending on your hardware.

```bash
python train.py
```

**What it does:**
- Downloads and preprocesses the CIFAR-10 dataset automatically
- Builds a 3-block CNN with a Dense head
- Trains for 10 epochs with a 20% validation split
- Prints test accuracy and loss after training
- Saves the trained model to `cifar10_model.keras`
- Saves a training/validation accuracy and loss plot to `training_curves.png`

**Expected output:**
```
Test accuracy : 0.7134
Test loss     : 0.8512

Model saved to "cifar10_model.keras"
Training curves saved to "training_curves.png"
```

> You only need to run `train.py` once. After that, use `test.py` for all predictions.

---

### Step 2 — Classify an image

```bash
python test.py your_image.jpg
```

Pass any image file (PNG, JPG, WEBP). It will be automatically resized to 32×32 before classification.

**Example output:**
```
Image : your_image.jpg
Input size fed to model: 32×32 px

── Predictions ───────────────────────────
1. cat          62.3%  ██████████████████
2. dog          18.1%  █████
3. deer          9.4%  ██
4. frog          4.8%  █
5. bird          3.2%  
...

→ Predicted: cat  (62.3% confidence)
```

If you prefer, you can hardcode the image path at the top of `test.py`:

```python
IMAGE_PATH = 'your_image.jpg'   # edit this line
```

Then run without an argument:

```bash
python test.py
```
You can also use the images download.jpg and download1.jpg which are already present in the directory to test the model after running train.py 
---

## Model Architecture

```
Input: (32, 32, 3)
  ↓
Conv2D(32 filters, 3×3, relu)
MaxPooling2D(2×2)
  ↓
Conv2D(64 filters, 3×3, relu)
MaxPooling2D(2×2)
  ↓
Conv2D(64 filters, 3×3, relu)
Flatten
  ↓
Dense(64, relu)
Dense(10)           ← logits, one per class
```

**Optimizer:** Adam  
**Loss:** Sparse Categorical Crossentropy (from logits)  
**Epochs:** 10 | **Batch size:** 32  
**Expected test accuracy:** ~70–75%

---

## Output Files

After running `train.py` you will find these files in the same directory:

| File | Description |
|------|-------------|
| `cifar10_model.keras` | Saved trained model — required by `test.py` |
| `training_curves.png` | Accuracy and loss plots across epochs |

---

## Notes

- `test.py` will exit with a clear error message if `cifar10_model.keras` is not found, so always run `train.py` first.
- CIFAR-10 images are natively 32×32. Larger images are downscaled before classification, which may reduce accuracy on detailed photos.
- To improve accuracy beyond 75%, consider adding `Dropout`, data augmentation, or training for more epochs.

# Image-Classification-Practical
