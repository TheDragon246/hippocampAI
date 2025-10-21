import datetime
import os
import torch
import torch_geometric
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataListLoader

import time
# Start the timer
start_time = time.time()

dataset = []
seed = 14

path = "/blue/stevenweisberg/share/hippocampAI/8knn_isthmuscingulate_fs_181023"

for filename in sorted(os.listdir(path)):
    f = torch.load(os.path.join(path, filename))
    dataset += [f]
    
for i in range(len(dataset)):
    dataset[i].x = dataset[i].x.reshape(dataset[i].num_nodes, 1)

from torch.nn import Sequential, Linear, ReLU, LeakyReLU, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import GraphNorm

torch.cuda.empty_cache()

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels+3, out_channels),
                              # GraphNorm(out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels),
                              # GraphNorm(out_channels),
                              ReLU()
                              )
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]
        
        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input.float())  # Apply our final MLP.

import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool


class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.conv1 = PointNetLayer(1, 32)
        self.conv2 = PointNetLayer(32, 64)
        self.conv3 = PointNetLayer(64, 128)
        self.out = Linear(128, 1)
        
    def forward(self, data):

        h, pos, batch, edge_index = data.x, data.pos, data.batch, data.edge_index
        
        # 3. Start bipartite message passing.
        h = self.conv1(h=h, pos=pos, edge_index=edge_index)

        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        
        h = self.conv3(h = h, pos = pos, edge_index = edge_index)

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        
        # 5. Output.
        return self.out(h)

from torch_geometric.nn import DataParallel


lr = 1e-3
num_epochs = 100
k_folds = 5

# model = PointNet()
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available for training: {num_gpus}")
# model = DataParallel(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)


from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

import random
random.seed(seed)
random.shuffle(dataset)
kfold = KFold(n_splits=k_folds, shuffle=False)

fold_losses = []
fold_mses = []

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    num_batches = 0

    for data_list in loader:
        optimizer.zero_grad() # Clear gradients from previous iteration.
        output = model(data_list) # Forward pass. 

        # Get the target tensor
        y = [data.y.view(-1) for data in data_list]
        loss = 0

        for y_batch in y:
            loss += criterion(output.view(-1), y_batch.to(output.device)) # Loss computation.
        
        loss.backward() # Backpropagation.
        optimizer.step() # Update parameters after backpropagation. 

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches
    _, _, final_mse = test(model, train_loader)
    
    return final_mse


@torch.no_grad()
def test(model, loader):
    model.eval()

    logits_list = []
    data_y_list = []


    for data_list in loader:
        output = model(data_list)

        # Get the target tensor
        y = [data.y.view(-1) for data in data_list]
        loss = 0

        for y_batch in y:
            loss += criterion(output.view(-1), y_batch.to(output.device))

        logits_list.append(output.detach().cpu())
        data_y_list.append(torch.cat([y_batch.view(-1) for y_batch in y]).detach().cpu())

    logits_all = torch.cat(logits_list, dim=0)
    data_y_all = torch.cat(data_y_list, dim=0)
    
    logits_all = logits_all.reshape(-1, 1)
    data_y_all = data_y_all.reshape(-1, 1)

    mse = criterion(logits_all, data_y_all)

    return logits_all, data_y_all, mse

def worker_init_fn(worker_id):
    seed_worker = 70
    np.random.seed(seed_worker)
    random.seed(seed_worker)
    torch.manual_seed(seed_worker)
    torch.initial_seed(seed_worker)
    torch.cuda.manual_seed(seed_worker)
    torch.cuda.manual_seed_all(seed_worker)
    return

test_mse_values = []
train_mse_values = []
# Initialize your predictions and targets list
predictions_train = []
targets_train = []
predictions = []
targets = []

def correlation_coefficient(pred, actual):
    return np.corrcoef(pred, actual)[0, 1]

def convert_to_1d_tensor(tsr):
    tsr = tsr.reshape(-1)
    return tsr
    
for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
    torch.cuda.empty_cache()
    train_dataset = [dataset[i] for i in train_index]
    test_dataset = [dataset[i] for i in test_index]
    
    train_loader = DataListLoader(train_dataset, batch_size = 1, shuffle = False, num_workers=0, worker_init_fn=worker_init_fn)
    test_loader = DataListLoader(test_dataset, batch_size = 1, num_workers=0, worker_init_fn=worker_init_fn)
    
    print(f"Fold {fold + 1}")
    
    model = PointNet()
    model = DataParallel(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    test_mse_values.clear()
    train_mse_values.clear()
    
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        loss = train(model, optimizer, train_loader)
        pred_arr, data_y_arr, mse_test = test(model, test_loader)
        pred_arr_train, data_y_arr_train, mse = test(model, train_loader)
        pred_arr = convert_to_1d_tensor(pred_arr)
        data_y_arr = convert_to_1d_tensor(data_y_arr)
        pred_arr_train = convert_to_1d_tensor(pred_arr_train)
        data_y_arr_train = convert_to_1d_tensor(data_y_arr_train)
        r_value_train = correlation_coefficient(pred_arr_train, data_y_arr_train)
        r_value_test = correlation_coefficient(pred_arr, data_y_arr)
        pred_arr_temp, data_y_arr_temp, mse_second = test(model, train_loader)
        print(f'Epoch: {epoch:02d}, Training Loss: {loss:.4f}, Train MSE: {mse:.4f}, r (train): {r_value_train:.4f}, Test MSE: {mse_test:.4f}, r (test): {r_value_test:.4f}')
        test_mse_values.append(mse_test.item())
        train_mse_values.append(mse.item())
    
    fold_losses.append(loss)
    fold_mses.append(mse)
    
    print("Predicted tensors for training set: ")
    print(pred_arr_train)
    predictions_train.append(pred_arr_train)
    print("Target tensorsfor training set: ")
    print(data_y_arr_train)
    targets_train.append(data_y_arr_train)
    print(f"Correlation for training set: {correlation_coefficient(pred_arr_train, data_y_arr_train)}")
    print("Predicted tensors for test set: ")
    print(pred_arr)
    #predictions.append(pred_arr)
    # Convert the tensor to a numpy array and add it to the list
    predictions.append(pred_arr.detach().cpu().numpy())
    print("Target tensors for test set: ")
    print(data_y_arr)
    targets.append(data_y_arr.detach().cpu().numpy())
    print(f"Correlation for test set: {correlation_coefficient(pred_arr, data_y_arr)}")
    #torch.save(model.state_dict(), 'model_weights1.pth')
   #  print(f"Test MSE losses for this fold: {test_mse_values}")
    # print(f"Train MSE losses for this fold: {train_mse_values}")

print(f"Seed: {seed}")
print(f"Fold losses: {fold_losses}")
print(f"Fold MSE values: {fold_mses}")
print(f"Target training values: {targets_train}")
print(f"Prediction training values: {predictions_train}")
print(f"Target values: {targets}")
print(f"Prediction values: {predictions}")


# End the timer
end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time
# Print the time taken
print(f"Time taken: {time_taken} seconds")

# Save the model state
#torch.save(model.state_dict(), 'model_weights2.pth')
                                                                                                                                                                                                                                                                                                                                                                                     