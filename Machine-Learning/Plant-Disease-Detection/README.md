# Plant Disease Detection Using MobileNetV2

## About the Project

Plant diseases can affect crop quality and agricultural production. Identifying diseases at an early stage can help farmers take appropriate action.

This project focuses on detecting plant diseases from leaf images using deep learning. It uses the MobileNetV2 model with transfer learning to classify images into 38 different plant disease and healthy leaf categories.

A Streamlit web application called **LeafCare** was also developed. It allows users to upload a leaf image and view the predicted category along with confidence scores.

## Project Objectives

- Detect plant diseases using leaf images.
- Apply transfer learning using MobileNetV2.
- Classify images into 38 categories.
- Evaluate the model using a separate testing dataset.
- Develop a simple web application for predictions.

## Dataset

The project uses the **PlantVillage dataset**, which contains images of healthy and diseased plant leaves.

Dataset details:

- Total images: 54,305
- Number of categories: 38
- Image type: Colour images
- Input image size: 150 × 150 pixels

### Dataset Split

The dataset was divided into three sets:

| Dataset | Percentage | Number of Images |
|---|---:|---:|
| Training | 70% | 38,013 |
| Validation | 15% | 8,146 |
| Testing | 15% | 8,146 |
| **Total** | **100%** | **54,305** |

Stratified sampling was used with a random state of 42 to maintain the class distribution across the three sets.

## Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow (PIL)
- Streamlit
- Jupyter Notebook

## Model Information

MobileNetV2 was selected as the deep learning architecture for this project. It is a pre-trained convolutional neural network that was adapted for plant disease classification using transfer learning.

Model configuration:

- Model: MobileNetV2
- Pre-trained weights: ImageNet
- Input shape: 150 × 150 × 3
- Output categories: 38
- Training epochs: 8

## Model Evaluation Results

The trained model was evaluated using the testing dataset.

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 92.57% |
| Macro Precision | 92.55% |
| Macro Recall | 89.43% |
| Macro F1-Score | 90.41% |
| Weighted F1-Score | 92.59% |

These results represent the model's performance on the held-out testing images.

## LeafCare Web Application

LeafCare is a simple web application developed using Streamlit.

### Main Features

- Upload leaf images in JPG, JPEG, or PNG format.
- View the uploaded image.
- Predict the plant disease category.
- Display the prediction confidence.
- Show the top three possible predictions.

## Project Structure

```text
Plant-Disease-Detection/
│
├── README.md
├── app.py
│
├── models/
│   └── plant_disease_mobilenetv2.keras
│
├── results/
│   ├── class_mapping.json
│   ├── dataset_split_seed42.csv
│   └── confusion_matrix_mobilenetv2.png
│
└── notebooks/
    └── plant_disease_detection.ipynb
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NidhiBorasia103/Academic-Projects.git
```

### 2. Navigate to the Project Folder

```bash
cd Academic-Projects/Machine-Learning/Plant-Disease-Detection
```

### 3. Install the Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn pillow streamlit
```

### 4. Download the Dataset

Download the PlantVillage dataset separately. The dataset is not included in this repository because of its size.

Update the dataset path in the Jupyter Notebook according to your computer.

### 5. Run the Web Application

Make sure the trained model and class mapping file are available in the expected folders.

Run the following command:

```bash
streamlit run app.py
```

The LeafCare application will open in your web browser.

## Limitations

- The model was trained using PlantVillage images, which may differ from real-world leaf photographs.
- Lighting, image quality, and background conditions can affect predictions.
- The model may produce incorrect predictions for unfamiliar images.
- Predictions should be treated as an initial reference, not a confirmed plant disease diagnosis.

## Future Improvements

- Test the model with more real-world leaf images.
- Improve prediction performance under different lighting conditions.
- Include additional plant species and disease categories.
- Explore other deep learning models.
- Add more features to the LeafCare application.

## Conclusion

This project demonstrates how deep learning and transfer learning can be used for plant disease classification. MobileNetV2 was trained using the PlantVillage dataset, and a Streamlit application was developed to make image-based predictions easier.

Through this project, practical experience was gained in dataset preparation, image classification, transfer learning, model evaluation, and web application development.

---

**Project Name:** Plant Disease Detection  
**Model:** MobileNetV2  
**Dataset:** PlantVillage  
**Web Application:** LeafCare  
**Framework:** Streamlit