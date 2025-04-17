import torch.nn as nn
import torchvision.models as models

def get_finetuned_resnet50(num_classes=10, fine_tune_blocks=2):
    """
    Load a pre-trained ResNet50 model and modify it for fine-tuning.
    Only the top layers (i.e., the last fine_tune_blocks blocks) and the final classification layer 
    are unfrozen and updated during training.

    Args:
        num_classes (int): Number of classes in the target dataset.
        fine_tune_blocks (int): Number of top blocks to fine-tune (1: only layer4, 2: layer4 and layer3).

    Returns:
        model (nn.Module): Modified ResNet50 model ready for fine-tuning.
    """
    # Load the pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Replace the final fully connected layer to match the number of classes in the dataset
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Firstly, freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 'fine_tune_blocks' blocks
    # For ResNet50, blocks are organized in layer4, layer3, layer2, layer1.
    if fine_tune_blocks >= 1:
        # Unfreeze the final block: layer4
        for param in model.layer4.parameters():
            param.requires_grad = True
    if fine_tune_blocks >= 2:
        # Unfreeze the second block: layer3
        for param in model.layer3.parameters():
            param.requires_grad = True
    if fine_tune_blocks >= 3:
        # Unfreeze the third block: layer2
        for param in model.layer2.parameters():
            param.requires_grad = True
    if fine_tune_blocks >= 4:
        # Unfreeze the first block: layer1
        for param in model.layer1.parameters():
            param.requires_grad = True
    
    return model