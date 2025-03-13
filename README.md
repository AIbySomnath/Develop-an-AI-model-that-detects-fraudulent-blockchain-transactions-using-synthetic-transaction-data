# Fraud Detection Using Deep Learning and SVM

## Overview
This project focuses on detecting fraudulent blockchain transactions using two machine learning techniques:
1. **Multi-Layer Perceptron (MLP) with Focal Loss and Learning Rate Scheduling**
2. **Support Vector Machine (SVM) Classifier**

## Techniques and Results
### 1. Multi-Layer Perceptron (MLP)
The MLP model was designed with multiple layers, batch normalization, dropout, and L2 regularization to improve generalization. Additionally, **Focal Loss** was used to handle class imbalance, and **learning rate scheduling** helped stabilize training.

#### **MLP Model Architecture**
- Input layer: 6 features (transaction amount, gas fee, transaction count, wallet age, hour, day of the week)
- Hidden layers:
  - **256 neurons** (ReLU, L2 regularization, BatchNorm, Dropout 0.4)
  - **128 neurons** (ReLU, L2 regularization, BatchNorm, Dropout 0.4)
  - **64 neurons** (ReLU, L2 regularization, BatchNorm, Dropout 0.3)
  - **32 neurons** (ReLU, L2 regularization, BatchNorm, Dropout 0.3)
- Output layer: **Sigmoid activation** for binary classification

#### **Training Enhancements**
- **Early stopping** (patience = 15)
- **ReduceLROnPlateau** (adaptive learning rate reduction)
- **Adam optimizer** (learning rate = 0.001)

#### **Results**
| Metric      | Score  |
|------------|--------|
| Accuracy   | 95.27% |
| Precision  | 92.07% |
| Recall     | 94.98% |
| F1-score   | 93.50% |
| ROC-AUC    | 95.20% |

### 2. Support Vector Machine (SVM)
A traditional **SVM classifier** was also implemented as a baseline model. It performed reasonably well but was outperformed by the deep learning model.

#### **SVM Results**
| Metric      | Score  |
|------------|--------|
| Accuracy   | 75.00% |

## Usage
### **Requirements**
Ensure you have the necessary dependencies installed:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### **Running the MLP Model**
```bash
python train_mlp.py
```

### **Running the SVM Model**
```bash
python train_svm.py
```

## Conclusion
The **MLP model significantly outperforms the SVM classifier**, achieving a **20% higher accuracy** and **better fraud detection performance**. The combination of **Focal Loss, Batch Normalization, Dropout, and Adaptive Learning Rate** contributed to the success of the deep learning model.

🚀 **Next Steps**:
- Implement **Autoencoder + MLP Hybrid Model**
- Try **Ensemble Learning with XGBoost and MLP**
- Deploy the model using **Streamlit API**

📌 **For any improvements, feel free to contribute!**

