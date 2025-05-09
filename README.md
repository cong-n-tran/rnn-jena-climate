# RNN Jena Climate

A Recurrent Neural Network (RNN) trained on the Jena climate dataset to predict the temperature of the next day. This project includes data preprocessing, model training, evaluation, and visualization tools for time-series forecasting.

---

## Features

- **Data Normalization**: Normalize raw climate data for better training performance.
- **Input and Target Creation**: Generate time-series inputs and targets for training, validation, and testing.
- **Model Architectures**: Train and evaluate two types of models:
  - A Dense Neural Network (DNN)
  - A Bidirectional LSTM-based Recurrent Neural Network (RNN)
- **Evaluation Metrics**:
  - Confusion Matrix for classification performance.
  - Accuracy and loss metrics during training and testing.

---

## Getting Started

### Prerequisites

- Python (>=3.8)
- TensorFlow (>=2.0)
- NumPy
