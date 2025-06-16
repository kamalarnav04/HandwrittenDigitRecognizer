# Handwritten Digit Recognizer 🔢

A neural network-based handwritten digit recognition system built with TensorFlow/Keras that can classify handwritten digits (0-9) from the MNIST dataset with **97.8% validation accuracy**.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.12+-orange.svg)

## 🎯 Project Overview

This project implements a feedforward neural network to recognize handwritten digits using the famous MNIST dataset. The model can process both the original MNIST dataset and custom PNG images uploaded by users.

### Key Features
- **High Accuracy**: Achieves 97.8% validation accuracy
- **Fast Training**: Trains in under 15 seconds (5 epochs)
- **Custom Image Support**: Can predict digits from user-uploaded PNG images
- **Lightweight Architecture**: Simple yet effective 3-layer neural network
- **Easy to Use**: Interactive prediction interface

## 📊 Model Performance

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 98.55% | 97.80% |
| **Loss** | 0.0442 | 0.0854 |
| **Training Time** | ~12 seconds | - |

### Training Progress
```
Epoch 1/5: loss: 0.2573 - accuracy: 0.9249 - val_accuracy: 0.9665
Epoch 2/5: loss: 0.1086 - accuracy: 0.9663 - val_accuracy: 0.9762
Epoch 3/5: loss: 0.0766 - accuracy: 0.9763 - val_accuracy: 0.9745
Epoch 4/5: loss: 0.0572 - accuracy: 0.9816 - val_accuracy: 0.9760
Epoch 5/5: loss: 0.0442 - accuracy: 0.9855 - val_accuracy: 0.9780
```

## 🏗️ Model Architecture

The neural network consists of:

```
Input Layer:     784 neurons (28×28 flattened image)
Hidden Layer 1:  128 neurons (ReLU activation)
Hidden Layer 2:  64 neurons (ReLU activation)  
Output Layer:    10 neurons (Softmax activation)
```

**Total Parameters**: ~101,770 trainable parameters

### Architecture Highlights
- **Activation Functions**: ReLU for hidden layers, Softmax for output
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Sparse Categorical Crossentropy
- **Regularization**: 10% validation split for monitoring

## 📁 Dataset

The project uses the **MNIST Dataset** with the following files:
- `train-images-idx3-ubyte`: 60,000 training images (28×28 pixels)
- `train-labels-idx1-ubyte`: 60,000 training labels
- `t10k-images-idx3-ubyte`: 10,000 test images
- `t10k-labels-idx1-ubyte`: 10,000 test labels

### Data Preprocessing
1. **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
2. **Reshaping**: Images flattened from 28×28 to 784-dimensional vectors
3. **Type Conversion**: Images converted to float32 for efficient computation

## 📝 Usage

### Training the Model
The model trains automatically when you run all cells in the notebook. Training takes approximately 12 seconds on modern hardware.

### Making Predictions on Custom Images
```python
# Example usage for custom PNG images
file_path = "path/to/your/digit_image.png"
predicted_class, confidence = predict_from_png(file_path)

print(f"Predicted digit: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
```

### Image Requirements for Custom Predictions
- **Format**: PNG
- **Content**: Single handwritten digit on light background
- **Automatic Processing**: The function handles resizing to 28×28 and grayscale conversion

## 🔬 Technical Details

### Data Loading
Custom functions to read MNIST's binary format:
- `load_images()`: Reads image data from idx3-ubyte files
- `load_labels()`: Reads label data from idx1-ubyte files

### Model Configuration
```python
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') 
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Image Preprocessing Pipeline
```python
def predict_from_png(file_path):
    img = Image.open(file_path).convert('L')  # Grayscale
    img = img.resize((28, 28))                # Resize to MNIST dimensions
    img_array = np.array(img) / 255.0         # Normalize
    img_array = img_array.reshape(1, 784)     # Flatten
    return model.predict(img_array)
```

## 📈 Results Analysis

### Strengths
- ✅ **High Accuracy**: 97.8% validation accuracy
- ✅ **Fast Training**: Converges quickly in 5 epochs
- ✅ **Robust Performance**: Consistent across training and validation
- ✅ **Practical Application**: Works with real-world PNG images

## 🛠️ File Structure
```
HandwrittenDigitRecognizer/
├── model.ipynb                    # Main Jupyter notebook
├── README.md                      # This file
├── train-images-idx3-ubyte        # Training images
├── train-labels-idx1-ubyte        # Training labels  
├── t10k-images-idx3-ubyte         # Test images
├── t10k-labels-idx1-ubyte         # Test labels
```

## 📚 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Sequential Model Guide](https://keras.io/guides/sequential_model/)

---