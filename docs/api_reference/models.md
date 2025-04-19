# 🤖 Models Module

The `models` module provides a structured and extensible framework for developing, training, and deploying machine learning models tailored to financial time series prediction tasks. It supports modular design for easy integration of various deep learning architectures and includes utilities for model management, preprocessing, and evaluation.

---

## 📦 Module Structure
```
models/
│
├── base_model.py               # Abstract base class for all models
├── utils.py                    # Utility functions for saving/loading models and retrieving model instances
└── <model_name>/               # Implementation of a specific model architecture
    ├── model_architecture.py   # Custom neural network architecture definition
    └── <model_name>_model.py   # Model class extending BaseModel with training/prediction logic
```

### BaseModel Class `base_model.py`

Key Methods:

-   `fit(df)` – Fits a DataFrame for model training and returned a trained model.

-   `predict(df)` – Fits a Dataframe for prediction with a trained model.

#### Usage Example

```py
class TCNModel(BaseModel):
    def predict(self, data):
        """
        Make predictions based on raw input data.

        :param data: Raw input data as a DataFrame
        :return: Numpy array of predicted probabilities
        """
        df_feat = self.prepare_features(data)
        df_scaled = self.normalize(df_feat)
        X = self.preprocess(df_scaled)

        if len(X) == 0:
            return np.array([]), np.array([])

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            print(logits.shape)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs
```

### Utilities Class `(utils.py)`

Core Methods:

-   `save_model(model, path)` – Save the model to a file (.pkl and .pth are supported)

-   `load_model(path)` – Load the model from a file (.pkl and .pth are supported)

-   `get_model(name, **kwargs)` – Retrieve a model by name with the default settings


#### Usage Example

```py
tcn = TCNModel(ModelUtils)
tcn.get_model("tcn")
```


### Model Architecture Class `model_architecture.py`

Contains the customizable model architecture used in pretrained model. This allows the user to modify the architecture to cater to their own needs if it's needed.

#### Usage Example

```py
class TCNClassifier(nn.Module):
    """TCN model for multiclass classification."""
    def __init__(self, input_features, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(TCNClassifier, self).__init__()
        self.tcn = TCN(input_features, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        tcn_output = self.tcn(x)
        last_output = tcn_output[:, :, -1]
        return self.linear(last_output)
```

### Model Main Class `model.py`

This file allows to user to train and utilize the model by calling the "fit" or "predict" functions.

---
```py
probs = self.model.predict(X_test)
model = self.model.fit(X_train)
```

### Supported Models for NOW

| Method  | Description |
| ------------- | ---------------|
| XGBoost                                | Uses mutual information regression to rank features based on dependency with the target. |
| Temporal Convolutional Network (TCN)   | Employs dilated causal convolutions for time series classification.                              |
| Convolutional Neural Network (CNN)     | Extracts local patterns from multi-feature time windows using 1D convolutions.                               |
| Graph Neural Network (GNN)   | Leverages graph structure and correlation to learn feature interactions.                                |
