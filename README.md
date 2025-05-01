# Recognizing Pleasant and Unpleasant Faces

## 💡 How to Load and Use the Model

Follow these steps to load and use the pre-trained ResNet50 model in your project for **predicting new images**.
You can find the following code in `run_model.py`

### ✔️ 1. Install Dependencies
```bash
pip install tensorflow opencv-python numpy matplotlib
```

### ✔️ 2. Load the saved model
```bash
model = load_model('best_resnet50.keras')
```

### ✔️ 3. Load the image and resize it to 224x224 pixels (same as ResNet50 input size)
```bash
img_path = 'path_to_your_image.jpg'  # Replace with your image path
img = cv2.imread(img_path)
```
### ✔️ 4. Resize and convert the image from BGR to RGB (as Keras expects RGB)
```bash
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
### ✔️ 5. Normalize the image to match ResNet50's preprocessing
```bash
from tensorflow.keras.applications.resnet50 import preprocess_input
img = np.expand_dims(img, axis=0) 
img = preprocess_input(img)
```
## ⚡ How to Make the Prediction

### ✔️ 1. Predict using the model
```bash
pred = model.predict(img)
```
### ✔️ 2. Convert prediction to readable result
```bash
label = 'pleasant' if pred[0] > 0.5 else 'unpleasant'
print(f"Prediction: {label} (Confidence: {pred[0][0]:.4f})")
```
