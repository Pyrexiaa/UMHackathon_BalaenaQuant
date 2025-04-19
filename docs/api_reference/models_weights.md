# 🧠 Model Weights Module

The `model_weights` directory serves as a centralized and organized location for storing pretrained model parameters and associated data scalers. It is designed to support easy retrieval and reuse of trained models across different pipelines and use cases.

---

## 📦 Module Structure

```
models_weights/
│
└── <model_name>/          # Folder for each specific model (e.g., tcn/)
   ├── model.pth /.pkl     # Trained model weights (PyTorch / Pickle file format)
   └── scaler.pkl          # Corresponding data scaler for preprocessing
```

# Description

1. model.pth or model.pkl
Contains the pretrained model's weights. The format depends on the model implementation (PyTorch models use .pth, traditional models like XGBoost may use .pkl).

2. scaler.pkl
Stores the fitted scaler object (e.g., StandardScaler, MinMaxScaler) used during training for feature normalization. Ensures consistency when preprocessing new data during inference.

---
