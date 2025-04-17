import argparse

def get_config():
    """
    Parse command line arguments for fine-tuning settings.
    Returns:
        args (Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Fine-tune ResNet50 on the iNaturalist Dataset')

    # Configuration for model and fine-tuning
    parser.add_argument('--num_classes', type=int, default=10, help='Number of target classes in iNaturalist dataset')
    parser.add_argument('--fine_tune_blocks', type=int, default=2, 
                        help='Number of top blocks (e.g., 1: only layer4, 2: layer4 and layer3) to fine-tune')

    # Configuration for training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for regularization')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')

    # Configuration for data directory location/path 
    parser.add_argument('--data_dir', type=str, default='data/inaturalist_12k', help='Directory containing the iNaturalist dataset')
    parser.add_argument('--plot_grid', action='store_true',
                        help='Whether to plot a 10×3 grid of test images & predictions') 

    args = parser.parse_args()
    return args