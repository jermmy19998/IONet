import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pdb


criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, train_loader, device, optimizer, mil_type, accumulate, epoch, epochs):
    model.train()
    loss_sum = 0.
    n = 0
    y_true = []
    y_pred = []
    loop = tqdm(train_loader, total=len(train_loader))

    for file_name, features, label in loop:
        label = label.to(device)
        features = features.squeeze(0).to(device)

        if mil_type == 'abmil':
            scores = model(features)
            loss = criterion(scores, label)
        elif mil_type == 'dsmil':
            scores, bag_prediction, _, _ = model(features)
            max_prediction, index = torch.max(scores, 0, True)
            loss_bag = criterion(bag_prediction, label)
            loss_max = criterion(max_prediction.view(1, -1), label)
            loss = 0.5 * loss_bag + 0.5 * loss_max
    

        if accumulate:
            loss = loss / 4
            loss.backward(retain_graph=True)
            if (n + 1) % 4 == 0 or (n + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred = torch.argmax(scores)
        y_pred.append(pred.item())
        y_true.append(label.item())

        n += 1
        loss_sum += loss.item()

        loop.set_description(f'Train [{epoch}/{epochs}]')
        loop.set_postfix(train_loss=loss.item())

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    avg_loss = loss_sum / len(train_loader)
    print(f"Train_balanced_acc: {balanced_acc}, Train_avg_loss: {avg_loss}")
    return avg_loss, balanced_acc


def valid_one_epoch(model, val_loader, device, mil_type, epoch, epochs):
    model.eval()
    loss_sum = 0.
    y_true = []
    y_pred = []

    with torch.no_grad():
        loop = tqdm(val_loader, total=len(val_loader))

        for file_name, features, label in loop:
            label = label.to(device)
            features = features.squeeze(0).to(device)

            if mil_type == 'abmil':
                scores = model(features)
                loss = criterion(scores, label)
                scores = torch.softmax(scores, 1)
            elif mil_type == 'dsmil':
                classes, bag_prediction, _, _ = model(features)
                max_prediction, index = torch.max(classes, 0, True)
                loss_bag = criterion(bag_prediction, label)
                loss_max = criterion(max_prediction.view(1, -1), label)
                loss = 0.5 * loss_bag + 0.5 * loss_max
                scores = 0.5 * torch.softmax(max_prediction, 1) + 0.5 * torch.softmax(bag_prediction, 1)

            pred = torch.argmax(scores)

            y_pred.append(pred.item())
            y_true.append(label.item())
            loss_sum += loss.item()

            loop.set_description(f'Val [{epoch}/{epochs}]')
            loop.set_postfix(valid_loss=loss.item())

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    avg_loss = loss_sum / len(val_loader)
    print(f"Validation loss:{avg_loss}, Validation balanced_acc: {balanced_acc}")

    return avg_loss, balanced_acc
