import numpy as np
import time
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm, trange

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((int(300), int(300))),
    transforms.ToTensor()
])

data = datasets.ImageFolder(
    root='../datasets/adult_children_elderly/train',
    transform=transform
)

train_idx, valid_idx = train_test_split(list(range(len(data))), train_size=0.9)

dataset = {
    'train': torch.utils.data.Subset(data, train_idx),
    'val': torch.utils.data.Subset(data, valid_idx)
}

dataloader = {
    'train': torch.utils.data.DataLoader(
        dataset=dataset['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    ),
    'val': torch.utils.data.DataLoader(
        dataset=dataset['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    ),
}

transform = transforms.Compose([
    transforms.Resize((int(300), int(300))),
    transforms.ToTensor()
])

dataset_test = datasets.ImageFolder(
    root="../datasets/adult_children_elderly/test",
    transform=transform
)

dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=BATCH_SIZE)

datasets_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
class_names = np.array(data.classes)

def eff_net_train_model(model, criterion, optimizer, scheduler, epochs=25):
    start = time.time()

    use_gpu = torch.cuda.is_available()

    best_mode_wts = model.state_dict()
    best_accuracy = 0.0

    losses = {'train': [], 'val': []}
    accuracy = {'train': [], 'val': []}

    pbar = trange(epochs, desc='Epoch')

    for epoch in pbar:

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.eval()

            curr_loss = 0.0
            curr_corrects = 0

            for data in tqdm(dataloader[phase], leave=False, desc=f'{phase} iter'):
                inputs, labels = data
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = inputs, labels

                if phase == 'train':
                    optimizer.zero_grad()

                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                preds = torch.argmax(outputs, -1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                curr_loss += loss.item()
                curr_corrects += int(torch.sum(preds == labels.data))

            epoch_loss = curr_loss / datasets_sizes[phase]
            epoch_accuracy = curr_corrects / datasets_sizes[phase]

            losses[phase].append(epoch_loss)
            accuracy[phase].append(epoch_accuracy)

            pbar.set_description('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - start

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val Acc: {:.4f}'.format(best_accuracy))

    model.load_state_dict(best_model_wts)

    return model, losses, accuracy

def evaluate(model):
    model.eval()

    curr_correct = 0
    for data in dataloader['val']:
        inputs, labels = data

    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()

    output = model(inputs)
    _, preds = torch.max(output, 1)

    curr_correct += int(torch.sum(preds == labels))

    return curr_correct / datasets_sizes['val']

def predict(model, dataloader_test, class_names):
    probs = []
    model.eval()
    with torch.no_grad():

        for inputs, y in tqdm(dataloader_test):
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            preds = model(inputs).cpu()
            probs.append(preds)

    print(f'probs shape before softmax: {len(probs)}')
    probs = nn.functional.softmax(torch.cat(probs), dim=-1).numpy()
    print(f'probs shape after softmax: {probs.shape}')
    probs = np.argmax(probs, axis=1)
    probs = class_names[probs]
    return probs

def eff_net_final(epochs):
    model_efficientnet = models.efficientnet_b3(pretrained=True)

    for param in model_efficientnet.parameters():
        param.require_grad = False

    model_efficientnet.classifier
    model_efficientnet.classifier = nn.Sequential(nn.Dropout(p=0.3),
                                                nn.Linear(1536, len(data.classes))
                                                )

    model_efficientnet = model_efficientnet.cuda()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model_efficientnet.parameters()), lr=1e-4)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    model_efficientnet, losses_efficientnet, accs_efficientnet = eff_net_train_model(model_efficientnet, loss_func, optimizer, exp_lr_scheduler, epochs)


def eff_predict(file_path):
    model_path = "../models/adult_vs_children/efficientnet/adult_vs_children_30_epochs.pkl"
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(file_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    _, predicted_class = torch.max(output, 1)

    print(f"Predicted class: {('adult' if predicted_class.item() == 0 else ('child' if predicted_class.item() == 1 else 'elderly'))}")
    return f"Predicted class: {('adult' if predicted_class.item() == 0 else ('child' if predicted_class.item() == 1 else 'elderly'))}"
