from cmath import inf
import os
import numpy as np
from tqdm import tqdm
import random
import sys

import jittor as jt
import jittor.nn as nn
from jittor import dataset, models, transform

# Hyperparameter setting
exp = 'jt_dnn_patch_1'
batch_size = 16
train_epochs = 5
learning_rate = 1e-3

valid_ratio = 0.1
test_ratio = 0.1

seed = 42

# Use CUDA if possible
jt.flags.use_cuda = jt.has_cuda

# Set random stat
os.environ['PYTHONHASHSEED'] = str(seed)
jt.set_global_seed(seed)


# Prepare datasets
train_transform = transform.Compose([
    transform.RandomHorizontalFlip(),
    # transform.RandomVerticalFlip(),
    transform.RandomRotation(15),
    transform.RandomResizedCrop((224, 224)),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
])

inference_transform = transform.Compose([
    transform.Resize((224, 224)),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
])

train_loader = dataset.ImageFolder('dataset/train', transform=train_transform).set_attrs(batch_size=batch_size, shuffle=True)

valid_img_loaders = []
for fname in os.listdir(f'dataset/val/'):
    label = train_loader.class_to_idx[os.listdir(f'dataset/val/{fname}')[0]]
    # img_loader = DataLoader(datasets.ImageFolder(f'dataset/val/{fname}', transform=inference_transform), batch_size=batch_size, shuffle=False, drop_last=False)
    img_loader = dataset.ImageFolder(f'dataset/val/{fname}', transform=inference_transform).set_attrs(batch_size=batch_size, shuffle=False, drop_last=False)
    valid_img_loaders.append((img_loader, label))

test_img_loaders = []
for fname in os.listdir(f'dataset/test/'):
    label = train_loader.class_to_idx[os.listdir(f'dataset/test/{fname}')[0]]
    img_loader = dataset.ImageFolder(f'dataset/test/{fname}', transform=inference_transform).set_attrs(batch_size=batch_size, shuffle=False, drop_last=False)
    test_img_loaders.append((img_loader, label))


# Prepare model and optimizer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4, bias=True)

optimizer = jt.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Train
loss_best = sys.float_info.max
for epoch in range(train_epochs):
    # Train one epoch
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader)
    for images, labels in loop:
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()

        total += labels.shape[0]
        predictions = jt.argmax(logits.detach(), 1)[0]
        correct += (predictions == labels).sum().item()
        running_loss += loss.item()
        loop.set_description(f'loss: {running_loss/total:.4f} | accuracy: {correct/total:.2f}')

    print(f'Training Loss: {running_loss/total:.4f} | Accuracy(Patch): {correct / total:.4f}')

    # Evaluate
    model.eval()
    running_loss = 0.0
    correct_patch = 0
    total_patch = 0
    correct_img = 0
    total_img = 0

    loop = tqdm(valid_img_loaders)
    loop.set_description('Evaluating...')

    with jt.no_grad():
        for loader, label_gt in loop:
            logits_total = []
            for images, labels in loader:
                labels = (jt.ones(images.shape[0]) * label_gt).long()
                logits = model(images)
                loss = criterion(logits, labels)
                logits_total.append(logits.detach().cpu())

                total_patch += labels.shape[0]
                predictions = jt.argmax(logits.detach(), dim=1)[0]
                correct_patch += (predictions == labels).sum().item()
                running_loss += loss.item()

            logits_total = jt.concat(logits_total)
            prediction_img = jt.argmax(logits_total.mean(dim=0), dim=0)[0]
            correct_img += (prediction_img.item() == label_gt)
            total_img += 1

        print(f'Validation Loss: {running_loss / total_patch:.4f} | Accuracy(Patch): {correct_patch / total_patch:.2f} | Accuracy(Image): {correct_img / total_img:.2f}')
        if running_loss / total_patch < loss_best:
            loss_best = running_loss / total_patch
            print('Currently lowest loss reached, saving model...')
            model.save(f'{exp}_model_best.pkl')


# Test
print(f'Best loss: {loss_best}')
model.load(f'{exp}_model_best.pkl')

model.eval()
running_loss = 0.0
correct_patch = 0
total_patch = 0
correct_img = 0
total_img = 0

loop = tqdm(test_img_loaders)
loop.set_description('Testing...')

with jt.no_grad():
    for loader, label_gt in loop:
        logits_total = []
        for images, labels in loader:
            labels = (jt.ones(images.shape[0]) * label_gt).long()
            logits = model(images)
            loss = criterion(logits, labels)
            logits_total.append(logits.detach().cpu())

            total_patch += labels.shape[0]
            predictions = jt.argmax(logits.detach(), dim=1)[0]
            correct_patch += (predictions == labels).sum().item()
            running_loss += loss.item()

        logits_total = jt.concat(logits_total)
        prediction_img = jt.argmax(logits_total.mean(dim=0), dim=0)[0]
        correct_img += (prediction_img.item() == label_gt)
        total_img += 1

    print(f'Test Loss: {running_loss / total_patch:.4f} | Accuracy(Patch): {correct_patch / total_patch:.2f} | Accuracy(Image): {correct_img / total_img:.2f}')
