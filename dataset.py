import os
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
from PIL import Image


# Define a custom dataset class for the iNaturalist dataset
CLASS_NAME_TO_IDX = {
    "Amphibia": 0,
    "Animalia": 1,
    "Arachnida": 2,
    "Aves": 3,
    "Fungi": 4,
    "Insecta": 5,
    "Mammalia": 6,
    "Mollusca": 7,
    "Plantae": 8,
    "Reptilia": 9
}

class iNaturalistDataset(Dataset):
    """
    Custom Dataset for the iNaturalist data that uses a fixed class mapping.
    Assumes that images are organized in subdirectories named as in CLASS_NAME_TO_IDX.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized by class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Loop through each class specified in the custom mapping
        for class_name, class_idx in CLASS_NAME_TO_IDX.items():
            class_folder = os.path.join(root_dir, class_name)
            if os.path.exists(class_folder):
                # List all image files ending in common image extensions
                for img_name in os.listdir(class_folder):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_folder, img_name)
                        self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB mode
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders(data_dir, batch_size=32, val_split=0.2):
    """
    Create training, validation, and test data loaders for the iNaturalist dataset.
    Assumes data is organized into 'train' and 'test' folders, with subfolders per class.

    Args:
        data_dir (str): Root directory of the dataset.
        batch_size (int): Batch size for data loaders.
        val_split (float): Fraction of the training set to use for validation.
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation, and testing.
    """
    # Define transformations matching pre-trained model expectations
    # (Resize and center crop for consistency, plus ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the training dataset from a directory structured with sub-folders for each class
    train_dataset = iNaturalistDataset(root=os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = iNaturalistDataset(root=os.path.join(data_dir, 'val'), transform=transform)
    
    # Split the training dataset into a training set and validation set
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader