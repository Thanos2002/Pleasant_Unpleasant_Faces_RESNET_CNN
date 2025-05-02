from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the model
model = load_model("resnet50_finetuned_final.keras")

# Load and preprocess the image
img_path = 'image.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0)  
img = preprocess_input(img)

# Predict
pred = model.predict(img)
label = 'pleasant' if pred[0] > 0.5 else 'unpleasant'

# Display prediction
print(f"Prediction: {label} (Confidence: {pred[0][0]:.4f})")

# Optionally, show the image
plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
plt.title(f"Predicted: {label}")
plt.axis('off')
plt.show()
