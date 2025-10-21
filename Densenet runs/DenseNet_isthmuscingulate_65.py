from __future__ import annotations
import pandas as pd
import numpy as np
import torch
#torch.use_deterministic_algorithms(True)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import random
import time
# Start the timer
start_time = time.time()
seed = 65

class MedicalDataset(Dataset):
    def __init__(self, file_directory, label_file):
        self.file_directory = file_directory
        self.label_file = label_file
        self.filenames = self.get_filenames()
        random.seed(seed)
        random.shuffle(self.filenames)
        
    def get_filenames(self):
        filenames = []
        for filename in os.listdir(self.file_directory):
            if filename.endswith(".npy") and 'normalized' in filename:
                # Split the filename by underscores and take the part that starts with 'sub-'
                parts = filename.split("_")
                for part in parts:
                    if part.startswith("sub-"):
                        image_id = part.replace("sub-", "")
                        filenames.append((int(image_id), os.path.join(self.file_directory, filename)))

        filenames.sort(key=lambda x: x[0])
        return [filename for _, filename in filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = self.filenames[idx]
        label = self.load_label(image_path)
        image = self.load_medical_image(image_path)
        return image, label

    def load_medical_image(self, image_path):
        image_array = np.load(image_path, allow_pickle = True)
        image_tensor = torch.from_numpy(image_array).float()
        return image_tensor

    def load_label(self, image_path):
        # Split the image_path by underscores and take the first part as the ID
        parts = os.path.basename(image_path).split("_")
        image_id = parts[0]
        for part in parts:
            if part.startswith("sub-"):
                image_id = part.replace("sub-", "")
        label_df = pd.read_csv(self.label_file)
        matching_row = label_df.loc[label_df['ID'] == int(image_id)]
        if not matching_row.empty:
            intensity = matching_row['Model_Building_Total'].values[0]
            return intensity
        else:
            return None


file_directory = "/blue/stevenweisberg/ashishkumarsahoo/difumo/Data/freesurfer_121023/isthmuscingulate_standardized_normalized"
label_file = '/blue/stevenweisberg/ashishkumarsahoo/DataAnalysisWith90Participants_Jupyter.csv'

dataset = MedicalDataset(file_directory, label_file)
print(len(dataset))

#from sklearn.utils import shuffle

# Shuffle the dataset
#dataset.filenames = shuffle(dataset.filenames, random_state=8)

# Iterate over the dataset
for idx in range(len(dataset)):
    image, label = dataset[idx]
    print("Image shape:", image.shape)
    print("Label:", label)
    print()


import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F



import re
from collections import OrderedDict
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option

__all__ = [
    "DenseNet",
    "Densenet",
    "DenseNet121",
    "densenet121",
    "Densenet121",
    "DenseNet169",
    "densenet169",
    "Densenet169",
    "DenseNet201",
    "densenet201",
    "Densenet201",
    "DenseNet264",
    "densenet264",
    "Densenet264",
]


class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-deterministic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, 1)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x


def _load_state_dict(model: nn.Module, arch: str, progress: bool):
    """
    This function is used to load pretrained models.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.

    """
    model_urls = {
        "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
        "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    }
    model_url = look_up_option(arch, model_urls, None)
    if model_url is None:
        raise ValueError(
            "only 'densenet121', 'densenet169' and 'densenet201' are supported to load pretrained weights."
        )

    pattern = re.compile(
        r"^(.*denselayer\d+)(\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + ".layers" + res.group(2) + res.group(3)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


class DenseNet121(DenseNet):
    """DenseNet121 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            if spatial_dims > 2:
                raise NotImplementedError(
                    "Parameter `spatial_dims` is > 2 ; currently PyTorch Hub does not"
                    "provide pretrained models for more than two spatial dimensions."
                )
            _load_state_dict(self, "densenet121", progress)


class DenseNet169(DenseNet):
    """DenseNet169 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 32, 32),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            if spatial_dims > 2:
                raise NotImplementedError(
                    "Parameter `spatial_dims` is > 2 ; currently PyTorch Hub does not"
                    "provide pretrained models for more than two spatial dimensions."
                )
            _load_state_dict(self, "densenet169", progress)


class DenseNet201(DenseNet):
    """DenseNet201 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 48, 32),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            if spatial_dims > 2:
                raise NotImplementedError(
                    "Parameter `spatial_dims` is > 2 ; currently PyTorch Hub does not"
                    "provide pretrained models for more than two spatial dimensions."
                )
            _load_state_dict(self, "densenet201", progress)


class DenseNet264(DenseNet):
    """DenseNet264"""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 64, 48),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            raise NotImplementedError("Currently PyTorch Hub does not provide densenet264 pretrained models.")


Densenet = DenseNet
Densenet121 = densenet121 = DenseNet121
Densenet169 = densenet169 = DenseNet169
Densenet201 = densenet201 = DenseNet201
Densenet264 = densenet264 = DenseNet264
    
    
    
lr = 1e-4

# cnn = CNN3D()
criterion = nn.MSELoss()
# optim = torch.optim.Adam(params = cnn.parameters(), lr = lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.optim as optim

# model = generate_model(10)
# optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()


def train(model, optimizer, criterion, train_loader):
    model.train()
    model.to(device)
    
    total_loss = 0
    num_batches = 0

    for images, labels in train_loader:
        images = images.to(device, dtype=torch.float) # Convert input tensors to floating-point numbers.
        labels = labels.to(device, dtype=torch.float) # Convert target tensors to floating-point numbers.
        
        optimizer.zero_grad() # Clear weights before any forward and backward passes.
        
        images = images.view(images.shape[0], 1, images.shape[1], images.shape[2], images.shape[3])
        # Above: [batch_size, num_input_channels, depth, height, width].
        
        output = model(images) # Forward pass. 

        output = output.reshape(-1, 1) # Reshape to column tensor to match
        # labels tensor (prevent broadcasting issues).
        labels = labels.reshape(-1, 1) # Ensure the same as described above.
        
        loss = criterion(output, labels) # Loss calculation (MSE).
        
        loss.backward() # Backpropagation.
        
        optimizer.step() # Update parameters.

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches
    torch.cuda.empty_cache()
    
    mse, preds, labels = test(model, criterion, train_loader)
    
    average_loss = mse
    
    return average_loss


def test(model, criterion, test_loader):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float) # Convert input tensors to Float
            labels = labels.to(device, dtype=torch.float) # Convert target tensors to Float
            
            images = images.view(images.shape[0], 1, images.shape[1], images.shape[2], images.shape[3])
            
            output = model(images)
            
            output = output.reshape(-1, 1)
            labels = labels.reshape(-1, 1)

            loss = criterion(output, labels)

            all_predictions.append(output)
            all_labels.append(labels)

    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # predictions = predictions.reshape(-1, 1)
    # labels = labels.reshape(-1, 1)
    mse = F.mse_loss(predictions, labels)

    return mse.item(), predictions, labels
    


def correlation_coefficient(pred, actual):
    return np.corrcoef(pred, actual)[0, 1]

def convert_to_1d_tensor(tsr):
    tsr = tsr.cpu()
    tsr = tsr.reshape(-1)
    return tsr
    
    
from sklearn.model_selection import KFold

k_folds = 5
num_epochs = 100
kfold = KFold(n_splits= k_folds, shuffle=False)

test_mse_values = []
train_mse_values = []

test_correlations_pred = []
test_correlations_target = []

for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
    torch.cuda.empty_cache()
    train_dataset = [dataset[i] for i in train_index]
    test_dataset = [dataset[i] for i in test_index]

    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False, num_workers=0)
    
    print(len(train_loader), len(test_loader))
    print(f"Fold {fold + 1}")
    
    model =DenseNet(
    spatial_dims = 3,
    in_channels = 1, # Grayscale.
    out_channels= 1, # No classes since this is a regression problem. Only one output unit.
    init_features=64,
    growth_rate=32,
    block_config=(6, 12, 24, 16),
    bn_size=4,
    act=("relu", {"inplace": True}),
    norm="batch",
    dropout_prob=0.0,
).to(device)

    
    test_mse_values.clear()
    train_mse_values.clear()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        loss = train(model, optimizer, criterion, train_loader)
        mse_test, pred_arr, data_y_arr = test(model, criterion, test_loader)
        mse, pred_arr_train, data_y_arr_train = test(model, criterion, train_loader)
        pred_arr = convert_to_1d_tensor(pred_arr)
        data_y_arr = convert_to_1d_tensor(data_y_arr)
        pred_arr_train = convert_to_1d_tensor(pred_arr_train)
        data_y_arr_train = convert_to_1d_tensor(data_y_arr_train)
        r_value_train = correlation_coefficient(pred_arr_train, data_y_arr_train)
        r_value_test = correlation_coefficient(pred_arr, data_y_arr)
        print(f'Epoch: {epoch:02d}, Training Loss: {loss:.4f}, Train MSE: {mse:.4f}, r (train): {r_value_train:.4f}, Test MSE: {mse_test:.4f}, r (test): {r_value_test:.4f}')

    print("Predicted tensors for training set: ")
    print(pred_arr_train)
    print("Target tensorsfor training set: ")
    print(data_y_arr_train)
    print(f"Correlation for training set: {correlation_coefficient(pred_arr_train, data_y_arr_train)}")
    print("Predicted tensors for test set: ")
    print(pred_arr)
    print("Target tensorsfor test set: ")
    print(data_y_arr)
    test_correlations_pred.append(pred_arr)
    test_correlations_target.append(data_y_arr)
    print(f"Correlation for test set: {correlation_coefficient(pred_arr, data_y_arr)}")

print(f"Validation set predictions (100 folds): {test_correlations_pred}")
print(f"Validation set targets (100 folds): {test_correlations_target}")

# End the timer
end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time
# Print the time taken
print(f"Time taken: {time_taken} seconds")
                                                                                                                                                                                                                                                                                                                                                                                     