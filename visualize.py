import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import iNaturalistDataset, CLASS_NAME_TO_IDX

def plot_prediction_grid(model, data_dir, device, samples_per_class=3, figsize_per_image=2):
    """
    Plot a (10 rows × samples_per_class cols) grid of test images with
    true & predicted labels.
    """
    # Reverse mapping from class index to class name
    idx_to_class = {v:k for k,v in CLASS_NAME_TO_IDX.items()}

    # Transformations for test images
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = iNaturalistDataset(
        root_dir=os.path.join(data_dir, 'val'),
        transform=test_transform
    )

    # Pick sample indices
    selected = {i: [] for i in range(len(CLASS_NAME_TO_IDX))}
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        if len(selected[label]) < samples_per_class: # if not full
            selected[label].append(idx)
        if all(len(v)==samples_per_class for v in selected.values()):
            break

    # Flatten into an order list
    ordered_indices = []
    for c in range(len(CLASS_NAME_TO_IDX)):
        ordered_indices += selected[c]

    # Gather images & labels
    images = [test_dataset[i][0] for i in ordered_indices]
    true_labels = [test_dataset[i][1] for i in ordered_indices]

    # Stack & move to device
    batch = torch.stack(images).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        preds = outputs.argmax(dim=1).cpu().tolist()

    # Inverse‐normalize for display
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    # Plot
    rows = len(CLASS_NAME_TO_IDX)
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols*figsize_per_image, rows*figsize_per_image))
    for i, ax in enumerate(axes.flat):
        img = batch[i].cpu()
        img = inv_norm(img)
        img = torch.clamp(img, 0, 1).permute(1,2,0).numpy()

        true_name = idx_to_class[true_labels[i]]
        pred_name = idx_to_class[preds[i]]

        ax.imshow(img)
        ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
