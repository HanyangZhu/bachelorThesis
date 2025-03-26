import torch
import numpy as np
from transformers import Trainer, TrainingArguments

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_wts = model.state_dict()
            torch.save(self.best_model_wts, self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    logging_dir='./logs',
)

early_stopping = EarlyStopping(patience=3, verbose=True)

for epoch in range(training_args.num_train_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{training_args.num_train_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break