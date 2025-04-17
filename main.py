import torch
from config import get_config
from model_finetune import get_finetuned_resnet50
from dataset import get_data_loaders
from train import train_model
from evaluate import evaluate_model

def main():
    # Parse the configuration settings
    args = get_config()

    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.random_seed)
    
    # Load the data loaders
    train_loader, val_loader, test_loader = get_data_loaders(data_dir=args.data_dir,
                                                             batch_size=args.batch_size)
    
    # Load the pre-trained model with fine-tuning strategy
    model = get_finetuned_resnet50(num_classes=args.num_classes, fine_tune_blocks=args.fine_tune_blocks)

    # Train the model
    model, history = train_model(model, train_loader, val_loader,
                                 learning_rate=args.learning_rate,
                                 weight_decay=args.weight_decay,
                                 epochs=args.epochs,
                                 device=device)
    
    # Evaluate the model on the test set
    _ = evaluate_model(model, test_loader, device)

    # Optionally, plot a grid of
    if args.plot_grid:
        from visualize import plot_prediction_grid
        plot_prediction_grid(model, args.data_dir, device,
                             samples_per_class=3,
                             figsize_per_image=2)

if __name__ == "__main__":
    main()