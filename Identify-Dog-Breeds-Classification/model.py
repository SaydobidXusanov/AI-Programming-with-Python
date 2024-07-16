# Imports here
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models

def create_model(kwargs**):
    model = models.vgg16(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # TODO: Build and train your network
    classifier = nn.Sequential(nn.Linear(25088, 12544),
                      nn.ReLU(),
                      nn.Linear(12544, 6272),
                      nn.ReLU(),
                      nn.Linear(6272, 1568),
                      nn.ReLU(),
                      nn.Linear(1568, 128),
                      nn.ReLU(),
                      nn.Linear(128, 102),
                      nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    return model

def train_model(model, trainloader, learning_rate, epochs, gpu):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
    steps = 0

    train_losses, test_losses = [], []

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            test_loss = 0
            accuracy = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in testloader:
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Process the image
    img = process_image(image_path)
    
    # Convert NumPy array to PyTorch tensor
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    # Add batch of size 1 to image (unsqueeze)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    # Calculate class probabilities and indices of topk predictions
    probabilities = torch.exp(output)
    top_probs, top_classes = probabilities.topk(topk, dim=1)
    
    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_classes.numpy()[0]]
    
    return top_probs.numpy()[0], top_classes