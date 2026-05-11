import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = 'cifar10_model.keras'   # path to your saved model
IMAGE_PATH = 'your_image.jpg'        # change to your image path (or pass as CLI arg)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f'Error: model file "{MODEL_PATH}" not found.')
    print('Run train.py first to train and save the model.')
    sys.exit(1)

print(f'Loading model from "{MODEL_PATH}" ...')
model = tf.keras.models.load_model(MODEL_PATH)
print('Model loaded.\n')

# ── Load & preprocess image ───────────────────────────────────────────────────
# Accept image path from command line: python test.py my_photo.jpg
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print(f'Error: image file "{IMAGE_PATH}" not found.')
    sys.exit(1)

img = Image.open(IMAGE_PATH).convert('RGB').resize((32, 32))
img_array = np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]
img_array = np.expand_dims(img_array, axis=0)          # add batch dimension → (1, 32, 32, 3)

# ── Inference ─────────────────────────────────────────────────────────────────
logits = model.predict(img_array, verbose=0)
probabilities = tf.nn.softmax(logits[0]).numpy()

# ── Results ───────────────────────────────────────────────────────────────────
top_indices = np.argsort(probabilities)[::-1]   # sort highest → lowest

print(f'Image : {IMAGE_PATH}')
print(f'Input size fed to model: 32×32 px\n')
print('── Predictions ──────────────────────────')

for rank, idx in enumerate(top_indices, start=1):
    bar = '█' * int(probabilities[idx] * 30)
    print(f'{rank}. {class_names[idx]:<12} {probabilities[idx]*100:5.1f}%  {bar}')

predicted_class = class_names[np.argmax(probabilities)]
confidence = probabilities[np.argmax(probabilities)] * 100
print(f'\n→ Predicted: {predicted_class}  ({confidence:.1f}% confidence)')
