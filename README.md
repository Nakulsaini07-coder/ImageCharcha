# 🖼️ Image Charcha

**An intelligent image captioning system that generates descriptive captions for any image using deep learning.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📝 Overview

**Image Charcha** is a cutting-edge deep learning application that automatically generates meaningful captions for images. Using a combination of **DenseNet201** for feature extraction and **LSTM** for sequence generation, the model can understand visual content and describe it in natural language.

The application provides an intuitive web interface powered by Streamlit, making it easy for users to upload images and receive AI-generated captions instantly.

---

## 📊 About the Dataset

The **Flickr8k dataset** is used for training and evaluating the image captioning system. It consists of **8,091 images**, each with **five captions** describing the content of the image. The dataset provides a diverse set of images with multiple captions per image, making it suitable for training caption generation models.

### Dataset Structure

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and organize the files as follows:

```
flickr8k/
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.txt
```

---

## ✨ Key Features

- 🎯 **Image Captioning** - Generate descriptive captions using trained deep learning models
- 🚀 **Fast Inference** - Quick response times with optimized model loading
- 🎨 **User-Friendly Interface** - Easy-to-use Streamlit web application
- 📊 **Robust Architecture** - Combines CNN (DenseNet201) and RNN (LSTM) for optimal performance
- 🔄 **Pre-trained Weights** - Includes ready-to-use model weights (model1.weights.h5)
- 📱 **File Upload Support** - Supports JPG, JPEG, and PNG formats

---

## 🛠️ Architecture

The project uses a sophisticated neural network architecture:

### Visual Feature Extraction

- **DenseNet201** - Pre-trained convolutional neural network to extract visual features (1920-dimensional vectors)

### Caption Generation

- **Embedding Layer** - Converts word indices to dense vectors
- **LSTM Network** - Processes sequential caption tokens and learns temporal dependencies
- **Attention-like Fusion** - Combines visual features with generated text through residual connections

### Model Layers

```
Input Image (224×224)
    ↓
DenseNet201 (feature extraction)
    ↓
Dense Layer (256 dims) → Reshaped to (1, 256)
    ↓
[Image Features] + [Word Embeddings]
    ↓
LSTM Layer (256 units)
    ↓
Residual Connection + Dense Layers
    ↓
Output Vocabulary Distribution
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ImageCharcha
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### How to Use

1. Open the web interface
2. Click on **"Choose an image"** to upload an image (JPG, JPEG, or PNG)
3. Wait for the model to process the image
4. View the generated caption describing your image

---

## 📋 Project Structure

```
ImageCharcha/
├── app.py                    # Main Streamlit application
├── training.ipynb            # Jupyter notebook for model training
├── model1.weights.h5         # Pre-trained model weights
├── tokenizer.json            # Word tokenizer configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📚 Dependencies

| Package    | Version  | Purpose                   |
| ---------- | -------- | ------------------------- |
| streamlit  | 1.52.2   | Web application framework |
| tensorflow | 2.20.0   | Deep learning framework   |
| numpy      | ≥ 2.1.0  | Numerical computing       |
| pandas     | 2.2.3    | Data manipulation         |
| Pillow     | 11.3.0   | Image processing          |
| matplotlib | ≥ 3.5.0  | Visualization             |
| seaborn    | ≥ 0.12.0 | Statistical visualization |
| h5py       | ≥ 3.11.0 | HDF5 file handling        |

---

## 🔧 Model Details

### Input Specifications

- **Image Input**: 224 × 224 pixels (preprocessed by DenseNet201)
- **Caption Input**: Sequences padded to 34 tokens
- **Vocabulary Size**: Dynamic based on tokenizer

### Output

- **Prediction**: Probability distribution over vocabulary for next word
- **Max Caption Length**: 34 words

### Training Configuration

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Model Architecture**: Custom CNN-RNN hybrid

---

## 📖 Training

The model was trained using a captioning dataset with the following approach:

1. **Feature Extraction**: DenseNet201 pre-trained on ImageNet
2. **Sequence Learning**: LSTM trained to generate captions token-by-token
3. **Joint Training**: Image features combined with word embeddings

For detailed training code and methodology, refer to `training.ipynb`

---

### Dataset Details

- **Total Images**: 8,091
- **Captions per Image**: 5
- **Total Captions**: 40,455
- **Image Types**: Diverse real-world scenes
- **Format**: JPG images with associated text captions

---

## 🎯 Use Cases

- 📸 **Content Creation** - Generate captions for social media posts
- ♿ **Accessibility** - Create alt text for images automatically
- 📚 **Document Processing** - Describe images in document databases
- 🏷️ **Image Tagging** - Automatically tag and organize images
- 🎓 **Educational Tools** - Learn about image understanding and NLP

---

## ⚙️ Technical Highlights

- **Efficient Loading**: Uses Streamlit caching for fast model initialization
- **Memory Optimization**: Manual weight loading from HDF5 format
- **Pre-processing**: Proper image normalization using DenseNet's preprocessing
- **Error Handling**: Robust tokenizer loading and weight mapping

---

## 🚀 Future Scope

The project has several exciting avenues for enhancement and expansion:

### 🔧 Model Improvements

- **Fine-tuning** - Experiment with fine-tuning the captioning model architecture and hyperparameters for improved performance
- **Beam Search** - Implement beam search decoding for generating multiple captions and selecting the best one
- **Attention Mechanisms** - Add attention layers to focus on specific image regions when generating captions

### 📈 Dataset Expansion

- **Dataset Diversification** - Incorporate additional datasets to increase the diversity and complexity of the trained model
- **Flickr30k Integration** - Train the model on the Flickr30k dataset with 30,000 images for enhanced capabilities
- **Domain-Specific Training** - Create specialized models for specific domains (medical imagery, products, etc.)

### 🎨 User Interface Enhancements

- **Image Previews** - Display uploaded images before caption generation
- **Confidence Scores** - Show confidence levels for generated captions
- **Caption History** - Allow users to view history of previously captioned images
- **Batch Processing** - Enable processing of multiple images at once

### 🌍 Multilingual Support

- **Multilingual Captioning** - Extend the model to generate captions in multiple languages by incorporating multilingual datasets
- **Language Selection** - Allow users to choose their preferred language for captions
- **Cross-lingual Transfer** - Leverage transfer learning across different languages

### 📊 Advanced Features

- **Caption Ranking** - Implement algorithms to rank and select the best captions
- **User Feedback Loop** - Collect user feedback to continuously improve model performance
- **Real-time Analytics** - Track and analyze caption generation patterns and user preferences

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Enhance documentation

---

---

**Happy Captioning! 🎉**
