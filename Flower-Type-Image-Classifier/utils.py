# Imports here
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np

def load_data(data_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    return trainloader

def save_checkpoint(model, arch, save_dir):
    # TODO: Save the checkpoint 
    checkpoint = {'input_size': 25088,
                  'arch': arch,
                  'output_size': 102,
                  'optimer_state_dict': optimizer.state_dict(),
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # Open the image using PIL
    img = Image.open(image_path)
    
    # Resize the image to have the shortest side of 256 pixels
    img = img.resize((256, 256))
    
    # Crop out the center 224x224 portion of the image
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert color channels to floats 0-1
    np_image = np.array(img) / 255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions to have color channel as the first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image