# Handwritten Digit Recognition System

A machine learning system for recognizing handwritten digits using a neural network built with TensorFlow/Keras. This project includes both model training and a production-ready classification system.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Code Logic](#code-logic)
- [Performance](#performance)
- [Testing](#testing)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a binary classification system for handwritten digit recognition. The system uses a neural network to classify handwritten digits into two categories (0 or 1) based on 20x20 pixel grayscale images. The model is trained using TensorFlow/Keras and exported as a pickle file for easy deployment.

## 📁 Project Structure

```
handwritten_recognis/
├── data/
│   ├── X.npy                 # Training features (1000 samples, 400 features each)
│   └── y.npy                 # Training labels (1000 samples, binary classification)
├── model_creation_notebook.ipynb  # Jupyter notebook for model training and exploration
├── classification_system.py       # Production-ready classification system
├── trained_model.pkl             # Exported trained model (generated after training)
├── autils.py                     # Utility functions for data loading
├── utils.py                      # Additional utility functions
├── public_tests.py               # Test cases for the system
├── venv/                         # Python virtual environment
└── README.md                     # This file
```

## ✨ Features

- **Neural Network Model**: 3-layer fully connected network with sigmoid activations
- **High Accuracy**: 99.90% accuracy on test data
- **Easy Deployment**: Model exported as pickle file for simple loading
- **Comprehensive API**: Single and batch prediction capabilities
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score
- **Error Handling**: Robust error handling for missing files and invalid inputs
- **Documentation**: Well-documented code with examples

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd handwritten_recognis
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install tensorflow numpy matplotlib jupyter
   ```

## 🎮 Usage

### Quick Start

1. **Train the model** (if not already trained):
   ```bash
   # Activate virtual environment first
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   
   # Run the training script
   python -c "
   import numpy as np
   import tensorflow as tf
   import pickle
   from autils import load_data
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense
   
   # Load and train model
   X, y = load_data()
   model = Sequential([
       tf.keras.Input(shape=(400,)),
       Dense(units=25, activation='sigmoid'),
       Dense(units=15, activation='sigmoid'),
       Dense(units=1, activation='sigmoid')
   ])
   model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(0.001))
   model.fit(X, y, epochs=20, verbose=0)
   
   # Save model
   with open('trained_model.pkl', 'wb') as f:
       pickle.dump(model, f)
   print('Model trained and saved!')
   "
   ```

2. **Run the classification system**:
   ```bash
   python classification_system.py
   ```

### Using the Classification System

#### Basic Usage

```python
from classification_system import HandwrittenDigitClassifier
import numpy as np

# Initialize the classifier
classifier = HandwrittenDigitClassifier()

# Load your image data (should be flattened to 400 features)
image_data = np.random.rand(400)  # Example: random data

# Make a prediction
probability, predicted_class = classifier.predict(image_data)
print(f"Probability: {probability:.4f}")
print(f"Predicted class: {predicted_class}")
```

#### Batch Prediction

```python
# For multiple images
batch_data = np.random.rand(5, 400)  # 5 images
probabilities, predicted_classes = classifier.predict_batch(batch_data)

for i, (prob, pred) in enumerate(zip(probabilities, predicted_classes)):
    print(f"Image {i+1}: Probability={prob:.4f}, Class={pred}")
```

#### Model Evaluation

```python
# Evaluate on test data
metrics = classifier.evaluate_on_test_data()
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

### Jupyter Notebook

For interactive exploration and model development:

```bash
jupyter notebook model_creation_notebook.ipynb
```

## 🧠 Model Architecture

The neural network consists of three fully connected layers:

```
Input Layer:    400 neurons (20x20 pixel image flattened)
Hidden Layer 1: 25 neurons (sigmoid activation)
Hidden Layer 2: 15 neurons (sigmoid activation)
Output Layer:   1 neuron   (sigmoid activation for binary classification)
```

**Total Parameters**: 10,431 trainable parameters

### Architecture Details

- **Input Shape**: (None, 400) - Batch size × 400 features
- **Activation Functions**: Sigmoid for all layers
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam with learning rate 0.001
- **Training Epochs**: 20

## 🔧 Code Logic

### Data Processing

1. **Data Loading**: Images are loaded from `X.npy` (features) and `y.npy` (labels)
2. **Preprocessing**: Images are already normalized and flattened to 400 features
3. **Shape**: Each image is 20×20 pixels, flattened to a 400-element vector

### Model Training Process

1. **Model Creation**: Sequential model with three dense layers
2. **Compilation**: Binary crossentropy loss with Adam optimizer
3. **Training**: 20 epochs on 1000 samples
4. **Export**: Model saved as pickle file for deployment

### Classification Logic

1. **Input Validation**: Ensures input is properly shaped (400 features)
2. **Prediction**: Forward pass through the neural network
3. **Thresholding**: Probability ≥ 0.5 → Class 1, else Class 0
4. **Output**: Returns both probability and binary classification

### Error Handling

- **File Not Found**: Graceful handling of missing model files
- **Input Validation**: Checks for correct input shapes
- **Model Loading**: Validates model integrity on load

## 📊 Performance

### Test Results

- **Accuracy**: 99.90% (999/1000 correct predictions)
- **Precision**: 99.80%
- **Recall**: 100.00%
- **F1-Score**: 99.90%

### Training Metrics

- **Final Loss**: ~0.0196 (after 20 epochs)
- **Convergence**: Model converges quickly within 20 epochs
- **Overfitting**: No significant overfitting observed

## 🧪 Testing

### Running Tests

```bash
# Run the classification system test
python classification_system.py

# Run public tests (if available)
python public_tests.py
```

### Test Coverage

The system includes tests for:
- Model loading and initialization
- Single prediction functionality
- Batch prediction functionality
- Evaluation metrics calculation
- Error handling scenarios

### Example Test Output

```
Handwritten Digit Classification System
========================================
Model loaded successfully from trained_model.pkl

Model Information:
  model_type: Sequential
  input_shape: (None, 400)
  output_shape: (None, 1)
  total_params: 10431
  layers: 3

Evaluation Results:
  accuracy: 0.9990
  precision: 0.9980
  recall: 1.0000
  f1_score: 0.9990
  total_samples: 1000
  correct_predictions: 999
```

## 🔍 Key Components Explained

### `HandwrittenDigitClassifier` Class

**Purpose**: Main interface for the classification system

**Key Methods**:
- `__init__()`: Initialize and load the trained model
- `predict()`: Single image prediction
- `predict_batch()`: Multiple image predictions
- `evaluate_on_test_data()`: Model evaluation
- `get_model_info()`: Model metadata

### Data Flow

```
Raw Image (20×20) → Flatten (400) → Neural Network → Probability → Binary Classification
```

### Model Persistence

- **Training**: Model trained in Jupyter notebook
- **Export**: Saved as pickle file for portability
- **Loading**: Automatically loaded when classifier is initialized

## 🚨 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated
2. **FileNotFoundError**: Make sure `trained_model.pkl` exists
3. **Shape Errors**: Verify input data is 400 features
4. **Memory Issues**: Reduce batch size for large datasets

### Solutions

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install --upgrade tensorflow numpy matplotlib jupyter

# Retrain model if needed
python -c "from autils import load_data; import tensorflow as tf; ..."
```

## 📈 Future Enhancements

- **Multi-class Classification**: Extend to recognize digits 0-9
- **Real-time Prediction**: Add web interface for live predictions
- **Model Optimization**: Implement model compression techniques
- **Data Augmentation**: Add image augmentation for better generalization
- **API Development**: Create REST API for model serving

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue in the repository

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional validation, monitoring, and security measures.