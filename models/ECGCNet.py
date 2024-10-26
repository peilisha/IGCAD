from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (global_mean_pool as gap, global_max_pool as gmp)
from torch_geometric.nn import GCNConv


class attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, enc_output):
        energy = torch.tanh(self.attn(enc_output))
        attention = self.v(energy)
        scores = F.softmax(attention, dim=1)
        out = enc_output * scores
        return torch.sum(out, dim=1), scores

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_features = 128 
        self.nhid = args.nhid  
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.lead_attn = attention(enc_hid_dim=64, dec_hid_dim=64)
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.lin3 = torch.nn.Linear(128, 64)
        self.lin4 = torch.nn.Linear(64, 32)
        self.lin = torch.nn.Linear(128, 32)

    def forward(self, data_x, data_edge_index, data_batch):
        x, edge_index, batch = data_x, data_edge_index, data_batch
        edge_attr = None
        x = F.relu(self.conv1(x, edge_index, edge_attr))# 384,64
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x=x.reshape(-1,12,self.nhid) 
        _, lweights = self.lead_attn(x) 
        x = x * (lweights)  
        x=x.reshape(-1,64) 
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin(x))
        return x,lweights

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7 if not downsample else 8,
            stride=1 if not downsample else 2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)  #
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )
        self.conv_down = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_down(x)
        out += identity
        out = self.relu(out)
        out = self.bn2(out)
        return out


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.beat_len = 80  
        self.start_filters = 32
        self.num_classes = args.num_classes
        self.conv1 = nn.Conv1d(
            in_channels=1,  
            out_channels=32,  
            kernel_size=7,  
            stride=1,
            padding=3,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=False)
        self.block = self._make_layer()
        self.beat_attn = attention(enc_hid_dim=128, dec_hid_dim=64)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.gcn = GCN(args)
        self.rweights = None
        self.lweights = None
        self.fc = nn.Linear(in_features=32, out_features=self.num_classes)  
    def _make_layer(self, blocks_list: list = [3, 3, 3]) -> List[ResBlock]:
        layers = []
        downsample = None
        num_filters = 32
        old_filters = num_filters
        for i in range(len(blocks_list)):
            num_blocks = blocks_list[i]
            for j in range(num_blocks):
                downsample = True if j == 0 and i != 0 else False
                layers.append(
                    ResBlock(
                        in_channels=old_filters,
                        out_channels=num_filters,
                        downsample=downsample,
                    )
                )
                old_filters = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)

    def forward(self, x, edge_index, batch):
        def segment_signal_tensor(signal, window_size, overlap):
            segments = []
            start = 0
            while start + window_size <= signal.size(1):
                segments.append(signal[:, start:start + window_size])
                start += int(window_size * (1 - overlap))
            return torch.stack(segments, dim=1)
        x = segment_signal_tensor(signal=x, window_size=self.beat_len, overlap=0.5)  
        beat_num = x.size(1)
        x = x.reshape(-1, self.beat_len).unsqueeze(1)
        x = self.conv1(x)  
        x = self.relu(x)  
        x = self.block(x)  
        x = self.global_avg_pool(x)  
        x = x.reshape(-1, beat_num, 128)  
        x, self.rweights = self.beat_attn(x)  
        gcn_out,self.lweights = self.gcn(x, edge_index, batch)  
        out = self.fc(gcn_out)

        return out
