
# 🐶🐱 Cat vs Dog Image Classification – CNN Project

A complete end-to-end deep learning project using a Convolutional Neural Network (CNN) to classify images of cats and dogs. This project demonstrates how to preprocess image datasets, train a CNN model, and deploy it to predict new unseen images using Python, TensorFlow/Keras, and OpenCV.

## 📝 Project Overview

This computer vision project aims to build a binary classifier that can accurately predict whether a given image is of a **cat** or a **dog**. It uses a custom CNN model trained on a clean and balanced dataset. The project also includes functionality to upload a new image and get a prediction result in real time.

## 🧾 Dataset Overview

- **Source**: [Kaggle - Cats and Dogs Mini Dataset](https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset)
- **Structure**: Two folders: `cat_set` and `dog_set`, each containing labeled images
- **Size**: ~1000 images (balanced between cats and dogs)
- **Format**: JPEG images only

### 📁 Folder Structure

```
cats_vs_dogs_mini/
├── cat_set/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
├── dog_set/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
```

## ⚙️ Tools & Technologies Used

- 🐍 **Python**
- 🧠 **TensorFlow / Keras** – Model training and prediction
- 📸 **OpenCV** – Image loading and preprocessing
- 🧪 **scikit-learn** – Train-test splitting
- 🎨 **Matplotlib** – Visualization
- 🌐 **Google Colab** – Model training and testing
- 📂 **Google Drive / `files.upload()`** – Image upload and prediction

## 🧠 Model Architecture Summary

- Input Layer – 128x128x3 image
- Conv2D → ReLU → MaxPooling (x2)
- Flatten → Dense → Dropout
- Output: Sigmoid (binary)

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

## 🧹 Data Preprocessing

1. **Load Images** from `cat_set/` and `dog_set/`
2. **Resize** to 128x128 pixels
3. **Normalize** pixel values to [0, 1]
4. **Label Encoding**: 0 for Cat, 1 for Dog
5. **Split** into training and testing sets (80-20)

## 🔮 Prediction Flow (New Image)

- Upload image using Google Colab’s `files.upload()`
- Display image using `matplotlib`
- Preprocess: Resize, Normalize
- Predict using `model.predict()`
- Show result as **“Cat”** or **“Dog”**

## 🚀 How to Use

1. **Download the Dataset**
   ```bash
   kaggle datasets download -d aleemaparakatta/cats-and-dogs-mini-dataset
   unzip cats-and-dogs-mini-dataset.zip
   ```

2. **Train the CNN model in Colab or VS Code**

3. **Upload a New Image for Prediction**
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

4. **Predict**
   ```python
   result = predict_image("your_image.jpg")
   print("Prediction:", result)
   ```

## 📊 Results

- Accuracy: ~90%+ on clean mini dataset
- Loss: Low training and validation loss
- Fast inference (<1 sec) for new images

## ✅ Future Improvements

- Add data augmentation
- Use pretrained models (Transfer Learning: MobileNetV2, ResNet)
- Build a web interface using Flask or Streamlit
- Train on full Dogs vs Cats dataset (25,000+ images)

## 📌 Credits

- **Dataset by**: [Aleema Parakatta on Kaggle](https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset)
- **Developed with**: TensorFlow, Keras, OpenCV, Google Colab
