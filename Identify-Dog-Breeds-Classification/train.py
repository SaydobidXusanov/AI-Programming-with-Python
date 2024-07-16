import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import create_model, train_model
from utils import load_data, save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument("data_directory", type=str, help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg13", help="Architecture for the model")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    # Load data
    trainloader = load_data(args.data_directory)

    # Create model
    model = create_model(arch=args.arch, hidden_units=args.hidden_units)

    # Train model
    train_model(model, trainloader, args.learning_rate, args.epochs, args.gpu)

    # Save checkpoint
    save_checkpoint(model, args.arch, args.save_dir)

if __name__ == "__main__":
    main()
