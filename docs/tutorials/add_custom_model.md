## Create A Custom Model


### 1. Directory Structure
Create a new folder inside the models/ directory for your model:
```
models/
└── my_awesome_model/
    ├── model_architecture.py     # Define your neural network architecture here
    └── my_awesome_model_model.py # Implements the training and prediction logic
```

### 2. Define Your Model Architecture
In model_architecture.py, define your model using PyTorch (or any other supported backend):
```py
# models/my_awesome_model/model_architecture.py

import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        super(MLPClassifier, self).__init__()
        layers = []

        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```
### 3. Create the Model Class
In my_awesome_model_model.py, extend the BaseModel to plug in training and prediction logic:

```py
# models/mlp_classifier/mlp_classifier_model.py

from models.base_model import BaseModel
from models.mlp_classifier.model_architecture import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPModel(BaseModel):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2, lr=0.001, epochs=50, patience=5, **kwargs):
        super().__init__(**kwargs)
        self.model = MLPClassifier(input_dim, hidden_dims, num_classes, dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.patience = patience

    def fit(self, df):
        X, y = self.prepare_training_data(df)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        self.model.train()
        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(best_model_state)
        return self

    def predict(self, df):
        X = self.prepare_features(df)
        if len(X) == 0:
            return np.array([])

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs


```

### 4. Register Your Model in utils.py
Add an entry in your get_model() function:
```py
# models/utils.py

from models.my_awesome_model.my_awesome_model_model import MyAwesomeModel

def get_model(name, **kwargs):
    models_dict = {
        "xgboost": XGBoostModel,
        "tabnet": TabNetModel,
        "tcn": TCNModel,
        "gnn": GNNModel,
        "mlp": MLPModel  # 👈 Add this line
    }
    return models_dict[name.lower()](**kwargs)
```

### 5. Saving and Loading
To save and load your model, use:
```py
from models.utils import save_model, load_model

save_model(my_model_instance, "path/to/model.pth")
loaded_model = load_model("path/to/model.pth")
```

### Usage Example
```py
from models.utils import get_model

model = get_model("mlp", input_dim=30, hidden_dim=64, num_classes=3)
model.fit(train_df)
probs = model.predict(test_df)
```