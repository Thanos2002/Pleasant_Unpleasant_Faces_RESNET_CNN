<h1 align="center">Pleasant or Unpleasant? : Teaching ResNet to Read Faces</h1>

| ![Image](https://github.com/user-attachments/assets/924bc45f-08ad-422c-8b82-21e7c31a52a0) | ![Image](https://github.com/user-attachments/assets/d3d28742-e084-4d43-86e8-1d48cc7b7947) | ![Image](https://github.com/user-attachments/assets/13c603dc-09b6-4de2-b4f5-6ae56a052b65)
|:----------------------------:|:----------------------------:|:----------------------------:|
##  Project Overview
A simple image classification project using a *fine-tuned* ResNet50 model (via transfer learning) to identify whether a face appears pleasant or unpleasant. The objective of this assignment is to develop a decision system that classifies facial images into two categories (binary classification). You can find the code on how we extract features (training only the custom head) and fine-tune the model (by unfreezing a few of the top layers of the frozen model base) in [resnet-cnn.ipynb](resnet-cnn.ipynb)

### :memo: **Approach:**  

1. **Transfer Learning with ResNet50:**  
   - We Leverage a pre-trained ResNet50 model (trained on ImageNet) as a feature extractor, initializing it with frozen weights to preserve learned patterns.  
   - Then, we replace the original classifier head with a **custom binary classification layer** (`GlobalAveragePooling2D → Dropout → Sigmoid-activated Dense`).  

2. **Two-Phase Training:**  
   - **Phase 1 (Feature Extraction):** We train only the custom head while keeping the ResNet50 base frozen, using a moderate learning rate (`Adam, LR=1e-4`).  
   - **Phase 2 (Fine-Tuning):** Then, we unfreeze the top 30 layers of ResNet50 to adapt high-level features to the target task, training with a lower learning rate (`1e-5`) for stability.  

3. **Regularization & Optimization:**  
   - Employ **Dropout (30%)** to mitigate overfitting.  
   - Monitor validation loss and **AUC (Area Under the Curve)** for robust performance evaluation.  
   - Use **Early Stopping** and **Model Checkpointing** to save the best model and prevent overfitting.  

### :gear: **Technical Highlights:**  
- **Architecture:** ResNet50 (fine-tuned) + custom binary classifier.  
- **Metrics:** Binary cross-entropy loss, accuracy, and AUC.  
- **Training:** 20 epochs (frozen base) + 10 epochs (fine-tuning) with batch size 32.  
- **Tools:** TensorFlow/Keras, ResNet50 preprocessing.
  
### :chart_with_upwards_trend:	 **Testing Results:**
- **Accuracy:** 97.18%
- **AUC:** 99.50%
- **Loss:** 0.0882

## :bulb: How to Load and Use the Model

Follow these steps to load and use the pre-trained ResNet50 model in your project for **predicting new images**.
You can find the following code in [run_model.py](run_model.py)

### :heavy_check_mark: 1. Install Dependencies
```bash
pip install tensorflow opencv-python numpy matplotlib
```

### :heavy_check_mark: 2. Load the saved model
```bash
model = load_model('best_resnet50.keras')
```

### :heavy_check_mark: 3. Load the image and resize it to 224x224 pixels (same as ResNet50 input size)
```bash
img_path = 'path_to_your_image.jpg'  # Replace with your image path
img = cv2.imread(img_path)
```
### :heavy_check_mark: 4. Resize and convert the image from BGR to RGB (as Keras expects RGB)
```bash
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
### :heavy_check_mark: 5. Normalize the image to match ResNet50's preprocessing
```bash
from tensorflow.keras.applications.resnet50 import preprocess_input
img = np.expand_dims(img, axis=0) 
img = preprocess_input(img)
```
## :zap: How to Make the Prediction

### :heavy_check_mark: 1. Predict using the model
```bash
pred = model.predict(img)
```
### :heavy_check_mark: 2. Convert prediction to readable result
```bash
label = 'pleasant' if pred[0] > 0.5 else 'unpleasant'
print(f"Prediction: {label} (Confidence: {pred[0][0]:.4f})")
```
