import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, learning_rate, weight_decay, epochs, device):
    """
    Train the fine-tuned model and evaluate on the validation dataset.
    
    Args:
        model (nn.Module): The fine-tuned model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay parameter.
        epochs (int): Number of training epochs.
        device (str): Device to run training on ('cuda' or 'cpu').
    
    Returns:
        model (nn.Module): The trained model.
        history (dict): Dictionary containing training and validation metrics.
    """
    # Move model to the specified device
    model.to(device)

    # We only update parameters that have requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        # Calculate training loss and accuracy
        train_loss /= total_train
        train_acc = correct_train / total_train

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        # Calculate validation loss and accuracy
        val_loss /= total_val
        val_acc = correct_val / total_val

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.1f}s - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
        
    return model, history
