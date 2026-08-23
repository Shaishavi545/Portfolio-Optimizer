import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

class VolatilityLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(VolatilityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last sequence output
        return self.sigmoid(out)

def prepare_lstm_data(returns_df, sequence_length=20, train_ratio=0.6, val_ratio=0.2):
    """
    Prepare chronologically split data for LSTM training.
    Target: Will the next 10-day realized volatility exceed the 80th percentile?
    Percentile is calculated ONLY on the training set.
    """
    # Calculate rolling 10-day realized volatility (annualized)
    volatility = returns_df.rolling(window=10).std() * np.sqrt(252)
    market_vol = volatility.median(axis=1)
    
    # Shift market_vol backward by 10 days to make it the target for today
    target_vol = market_vol.shift(-10)
    
    market_returns = returns_df.mean(axis=1)
    
    # Combine to ensure index alignment and drop NAs
    df = pd.DataFrame({
        'return': market_returns,
        'target_vol': target_vol
    }).dropna()
    
    # Split chronologically to find training threshold
    n = len(df)
    train_end = int(n * train_ratio)
    train_df = df.iloc[:train_end]
    
    # Compute threshold STRICTLY on training data
    threshold_80 = np.percentile(train_df['target_vol'], 80)
    
    # Create target: 1 if vol > threshold, 0 otherwise
    target = (df['target_vol'] > threshold_80).astype(int)
    
    X, y, dates = [], [], []
    returns_arr = df['return'].values
    target_arr = target.values
    
    for i in range(sequence_length, len(df)):
        seq = returns_arr[i-sequence_length:i]
        X.append(seq)
        y.append(target_arr[i])
        dates.append(df.index[i])
        
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y).reshape(-1, 1)
    
    # Split X, y chronologically
    n_samples = len(X)
    t_end = int(n_samples * train_ratio)
    v_end = t_end + int(n_samples * val_ratio)
    
    X_train, y_train = X[:t_end], y[:t_end]
    X_val, y_val = X[t_end:v_end], y[t_end:v_end]
    X_test, y_test = X[v_end:], y[v_end:]
    
    # Convert to PyTorch tensors
    tensors = {
        "train": (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        "val": (torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        "test": (torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    }
    
    return tensors, threshold_80, dates

def train_lstm(tensors, epochs=50, lr=0.01):
    """
    Train the LSTM model.
    """
    X_train, y_train = tensors["train"]
    X_val, y_val = tensors["val"]
    
    model = VolatilityLSTM()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    # Evaluate on val
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val).item()
        
        preds_np = val_preds.numpy()
        y_val_np = y_val.numpy()
        
        try:
            auc = roc_auc_score(y_val_np, preds_np)
            f1 = f1_score(y_val_np, (preds_np > 0.5).astype(int))
        except ValueError:
            auc = 0.5
            f1 = 0.0
            
    metrics = {"val_loss": val_loss, "roc_auc": auc, "f1": f1}
    return model, metrics

def predict_volatility_regime(model, recent_returns_seq):
    """
    Predict the probability of high volatility regime for the next period.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(recent_returns_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        prob = model(x).item()
    return prob
