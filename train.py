import argparse
import glob
import os
import time
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from models.ECGCNet import Model
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from utils.metrics import calculate_metrics_from_logits, calculate_metrics_from_logits1
from torch.autograd import Variable
from utils import ecg_data
from models.visualization import visualize_weights_all_leads

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--train', type=bool, default=True, help='train and valid')
parser.add_argument('--test', type=bool, default=True, help='test')
parser.add_argument('--inference', type=bool, default=True, help='inference')
parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs')  
parser.add_argument('--batch_size', type=int, default=32, help='batch size')  
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')  
parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')  
parser.add_argument("--model", type=str, default="Model", help="Select the model to train")
parser.add_argument('--dataset', type=str, default='cad', help='dataset')  
args = parser.parse_args()


# Random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


setup_seed(42)


def print_datainfo(dataset):
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def train_epoch(
        model: nn.Module,
        optimizer: torch.optim,
        loss_func,
        train_loader,
        epoch: int,
        num_classes: int,
) -> Tuple[float, float, float]:
    model.train()
    pred_all = []
    loss_all = []
    gt_all = []
    for _, data in tqdm(enumerate(train_loader), desc="train"):
        data = data.to(device)
        pred = model(data.x.double(), data.edge_index, data.batch)
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)
        loss = loss_func(pred, y_true.to(device))
        loss_all.append(loss.cpu().detach().item())
        pred_all.append(pred.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gt_all.append(y_true.cpu().detach().numpy())
        torch.cuda.empty_cache()
    print("Epoch: {0}".format(epoch))
    pred_all = np.concatenate(pred_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0)
    acc1 = calculate_metrics_from_logits1(gt_all, pred_all)
    roc_score = roc_auc_score(np.array(gt_all), pred_all, average="macro")
    print(f'train_loss: {np.mean(loss_all)}, train_mean_accuracy: {acc1},train_roc_score: {roc_score}')
    return np.mean(loss_all), acc1, roc_score


def test_epoch(
        model: nn.Module,
        loss_func,
        loader,
        epoch: int,
        num_classes: int,
) -> Tuple[float, float, float]:
    model.eval()
    pred_all = []
    loss_all = []
    gt_all = []
    for _, data in tqdm(enumerate(loader), desc="valid"):
        data = data.to(device)
        pred = model(data.x.double(), data.edge_index, data.batch)  
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)
        loss = loss_func(pred, y_true.to(device))
        pred_all.append(pred.cpu().detach().numpy())
        gt_all.append(y_true.cpu().detach().numpy())
        loss_all.append(loss.cpu().detach().numpy())
    pred_all = np.concatenate(pred_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0)
    acc1 = calculate_metrics_from_logits1(gt_all, pred_all)
    roc_score = roc_auc_score(np.array(gt_all), pred_all, average="macro")
    print(f'test_loss: {np.mean(loss_all)}, test_mean_accuracy: {acc1},test_roc_score: {roc_score}')
    return np.mean(loss_all), acc1, roc_score


def train(
        train_loader,
        val_loader,
        model: nn.Module,
        epochs: int,  # 60
        name: str = "Model",
        num_classes: int = 2,  # 9
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)  
    loss_func = torch.nn.BCEWithLogitsLoss()  
    best_score = 0.0
    for epoch in range(epochs):
        train_results = train_epoch(model, optimizer, loss_func, train_loader, epoch,
                                    num_classes=num_classes)
        test_results = test_epoch(model, loss_func, val_loader, epoch, num_classes=num_classes)
        scheduler.step()
        if epoch >= 2 and best_score <= test_results[1]: 
            best_score = test_results[1]
            save_path = os.path.join(os.getcwd(), "checkpoints/", f"{name}_weights.pt")
            torch.save(model.state_dict(), save_path)



def test(
        model: nn.Module,
        test_loader,
        num_classes
) -> None:
    pred_all = []
    loss_all = []
    gt_all = []
    model.eval()
    for i, data in tqdm(enumerate(test_loader), desc="test"):
        data = data.to(device)
        pred = model(data.x.double(), data.edge_index, data.batch)
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)
        pred_all.append(pred.cpu().detach().numpy())
        gt_all.append(y_true.cpu().detach().numpy())
    pred_all = np.concatenate(pred_all, axis=0)
    y_test = np.array(np.concatenate(gt_all, axis=0))
    acc1, auc1, precision, recall, f1 = calculate_metrics_from_logits(y_test, pred_all)
    print(f"accuracy: {acc1}")
    print(f"roc_score : {auc1}")
    print(f"precision : {precision}")
    print(f"recall : {recall}")
    print(f"f1 score : {f1}")
    logs = dict()
    logs["accuracy"] = acc1
    logs["auc"] = auc1
    logs["precision"] = precision
    logs["recall"] = recall
    logs["f1"] = f1
    name = "output"
    logs_path = os.path.join(os.getcwd(), "logs", f"{name}_logs.json")
    jsObj = json.dumps(logs)
    fileObject = open(logs_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()


def visualize_lead(model, data1, idx):
    'visualization'
    outputs = model(data1.x.double(), data1.edge_index, data1.batch)
    lattentions = model.lweights.cpu().detach().numpy()
    lattentions = lattentions.reshape(12, )
    # draw bar graph
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    plt.figure()
    plt.bar(lead_names, lattentions, color='royalblue')
    plt.title('Lead-wise Attention Weights', fontsize=14)
    plt.xlabel('Leads', fontsize=12)
    plt.ylabel('Attention Weights', fontsize=12)
    plt.tight_layout()
    plt.savefig("bargraph.svg", bbox_inches='tight', pad_inches=0)
    # draw weight mapping
    rattentions = model.rweights.cpu().detach().numpy()
    rattentions = rattentions.reshape(12, 24)
    signal0 = data1.x.cpu().detach().numpy()
    visualize_weights_all_leads(signal0, rattentions, idx)


if __name__ == "__main__":

    data_root = "./data/"
    sampling_rate = 100
    # Load data
    data_path = os.path.join(data_root, args.dataset, "raw/")
    _, _, Y = ecg_data.load_dataset(data_path, sampling_rate)
    ecg_dataset = ecg_data.ECGDataset(os.path.join(data_root, args.dataset))
    print_datainfo(ecg_dataset)
    args.num_classes = 2
    args.num_features = ecg_dataset.num_features

    # Split data
    train_dataset, val_dataset, test_dataset = ecg_data.select_dataset(ecg_dataset, Y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Build Model
    model = Model(args).double().to(device)


    # # Model training
    # if args.train:
    #   print("<=============== Start Training ===============>")
    #   train(
    #       model=model,
    #       epochs=args.epochs,
    #       name=args.model,
    #       train_loader=train_loader,
    #       val_loader=test_loader,
    #       num_classes=args.num_classes,
    #    )

    # Model testing
    if args.test:
        print("<=============== Start Testing ===============>")
        model.load_state_dict(torch.load('./checkpoints/weight.pt'))
        test(model, test_loader, num_classes=args.num_classes)

    # Model inference
    if args.inference:
        print("<=============== Start inferring ===============>")
        model.load_state_dict(torch.load('./checkpoints/weight.pt'))
        model = model.to(device)
        idx = 238
        data1 = test_dataset[idx]
        data1 = data1.to(device)
        visualize_lead(model, data1, idx)

