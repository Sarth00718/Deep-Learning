# Deep Learning Repository 🧠

A comprehensive collection of practical implementations and exercises covering Computer Vision, Deep Learning, and Natural Language Computing. This repository contains hands-on projects, datasets, and Jupyter notebooks for learning and experimenting with modern AI techniques.

## 📁 Repository Structure

### 🖼️ Computer Vision & Deep Learning (CVDL)
The `CVDL/` directory contains 10 comprehensive practicals covering fundamental to advanced computer vision concepts:

- **CVDL_Practical_1.ipynb** - Image Processing Fundamentals
- **CVDL_Practical_2.ipynb** - Image Enhancement Techniques
- **CVDL_Practical_3.ipynb** - Filtering and Convolution Operations
- **CVDL_Practical_4.ipynb** - Edge Detection and Feature Extraction
- **CVDL_Practical_5.ipynb** - Morphological Operations
- **CVDL_Practical_61.ipynb** - Introduction to Neural Networks
- **CVDL_Practical_62.ipynb** - Convolutional Neural Networks (CNNs)
- **CVDL_Practical_63.ipynb** - Advanced CNN Architectures
- **CVDL_Practical_7.ipynb** - Object Detection Techniques
- **CVDL_Practical_8.ipynb** - Image Segmentation
- **CVDL_Practical_9.ipynb** - Transfer Learning
- **CVDL_Practical_10.ipynb** - Advanced Deep Learning Applications
- **cifar_10_cnn.ipynb** - CIFAR-10 Classification with CNNs

#### Sample Images & Datasets
- Classic computer vision test images (Lena, Baboon, Cameraman, etc.)
- Custom datasets for various experiments
- CIFAR-10 and CIFAR-100 datasets
- Penn-Fudan pedestrian detection dataset

### 🔤 Natural Language Computing (NLC)
The `NLC/` directory contains 11 practicals covering NLP fundamentals to advanced techniques:

- **pract1/** - Text Preprocessing and Basic NLP
  - Disaster tweet classification dataset
  - Text cleaning and tokenization
  - Feature extraction techniques
- **pract2/** - Text Classification
- **pract3/** - Sentiment Analysis
- **pract4/** - Named Entity Recognition (NER)
- **pract5/** - Part-of-Speech Tagging
- **pract6/** - Language Modeling
- **pract7/** - Word Embeddings (Word2Vec, GloVe)
- **pract8/** - Sequence-to-Sequence Models
- **pract9/** - Attention Mechanisms
- **pract10/** - Transformer Architecture
- **pract11/** - Advanced NLP Applications

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.7+
pip install jupyter notebook
pip install numpy pandas matplotlib seaborn
pip install opencv-python pillow
pip install tensorflow keras pytorch
pip install nltk spacy transformers
pip install scikit-learn
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Sarth00718/Deep-Learning-.git
cd Deep-Learning-
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # If available
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 📊 Key Features

### Computer Vision & Deep Learning
- **Image Processing**: Filtering, enhancement, morphological operations
- **Feature Detection**: Edge detection, corner detection, SIFT/SURF
- **Deep Learning**: CNN architectures, transfer learning, object detection
- **Datasets**: CIFAR-10/100, custom image datasets
- **Applications**: Image classification, object detection, segmentation

### Natural Language Processing
- **Text Preprocessing**: Tokenization, stemming, lemmatization
- **Feature Engineering**: TF-IDF, word embeddings, n-grams
- **Classification**: Sentiment analysis, spam detection, topic modeling
- **Advanced NLP**: Transformers, BERT, attention mechanisms
- **Real-world Applications**: Tweet classification, text generation

## 🛠️ Technologies Used

- **Languages**: Python
- **Deep Learning**: TensorFlow, Keras, PyTorch
- **Computer Vision**: OpenCV, PIL, scikit-image
- **NLP**: NLTK, spaCy, Transformers, Gensim
- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn
- **Development**: Jupyter Notebook, Git LFS

## 📈 Learning Path

### Beginner
1. Start with CVDL Practical 1-3 for image processing basics
2. Explore NLC Practical 1-2 for text preprocessing fundamentals
3. Practice with provided sample datasets

### Intermediate
1. Dive into CNN architectures (CVDL Practical 6-7)
2. Implement text classification models (NLC Practical 3-5)
3. Experiment with transfer learning techniques

### Advanced
1. Explore advanced architectures and attention mechanisms
2. Work on end-to-end projects combining multiple techniques
3. Implement state-of-the-art models from recent papers

## 📝 Usage Examples

### Computer Vision Example
```python
# Load and process an image
import cv2
import numpy as np

# Load image
img = cv2.imread('CVDL/lenaimg.png')

# Apply Gaussian blur
blurred = cv2.GaussianBlur(img, (15, 15), 0)

# Display results
cv2.imshow('Original vs Blurred', np.hstack([img, blurred]))
```

### NLP Example
```python
# Text preprocessing pipeline
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('NLC/pract1/train.csv')

# Preprocess text
def preprocess_text(text):
    # Remove special characters, lowercase, etc.
    return processed_text

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course instructors and teaching assistants
- Open-source community for datasets and tools
- Research papers and tutorials that inspired these implementations

## 📞 Contact

- **Author**: Sarth Patel
- **GitHub**: [@Sarth00718](https://github.com/Sarth00718)
- **Repository**: [Deep-Learning-](https://github.com/Sarth00718/Deep-Learning-.git)

---

⭐ **Star this repository if you find it helpful!** ⭐
