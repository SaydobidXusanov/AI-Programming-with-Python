import argparse
import torch
from torchvision import transforms
from model import predict
from utils import load_checkpoint, process_image

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability.")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to the mapping of categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load checkpoint
    model, class_to_idx = load_checkpoint(args.checkpoint)

    # Process image
    img = process_image(args.input)

    # Predict
    probs, classes = predict(img, model, topk=args.top_k)

    # Display results
    print("Top classes:", classes)
    print("Top probabilities:", probs)

if __name__ == "__main__":
    main()
