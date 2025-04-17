import torch
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    """
    Evaluate the fine-tuned model on the test set and compute accuracy.
    
    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader.
        device (str): Device to run evaluation on.
    
    Returns:
        test_accuracy (float): Accuracy on the test set.
    """
    # Move model to the specified device
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return test_accuracy