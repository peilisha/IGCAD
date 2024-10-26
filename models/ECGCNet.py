# # from typing import List
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # # from gnn.gnn_models2 import GCN #这是9951的模型
# # from gnn.gnn_models import GCN
# #
# # class attention(nn.Module):
# #     def __init__(self, enc_hid_dim, dec_hid_dim):
# #         super().__init__()
# #         self.attn = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
# #         self.v = nn.Linear(dec_hid_dim, 1, bias=False)
# #     def forward(self, enc_output):
# #         energy = torch.tanh(self.attn(enc_output))
# #         attention = self.v(energy)
# #         scores = F.softmax(attention, dim=1)
# #         out = enc_output * scores
# #         return torch.sum(out, dim=1), scores
# #
# #
# # class ResBlock(nn.Module):
# #     def __init__(self, in_channels: int, out_channels: int, downsample: bool):
# #         super(ResBlock, self).__init__()
# #         self.conv1 = nn.Conv1d(
# #             in_channels=in_channels,
# #             out_channels=out_channels,
# #             kernel_size=7 if not downsample else 8,
# #             stride=1 if not downsample else 2,
# #             padding=3,
# #             bias=False
# #         )
# #         self.bn1 = nn.BatchNorm1d(num_features=out_channels)
# #         self.relu = nn.ReLU(inplace=True)  #
# #         self.conv2 = nn.Conv1d(
# #             in_channels=out_channels,
# #             out_channels=out_channels,
# #             kernel_size=7,
# #             stride=1,
# #             padding=3,
# #             bias=False
# #         )
# #         self.conv_down = nn.Conv1d(
# #             in_channels=in_channels,
# #             out_channels=out_channels,
# #             kernel_size=2,
# #             stride=2,
# #             padding=0,
# #             bias=False
# #         )
# #         self.bn2 = nn.BatchNorm1d(num_features=out_channels)
# #         self.downsample = downsample
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         identity = x
# #         out = self.conv1(x)
# #         out = self.relu(out)
# #         out = self.bn1(out)
# #         out = self.conv2(out)
# #         if self.downsample:
# #             identity = self.conv_down(x)
# #         out += identity
# #         out = self.relu(out)
# #         out = self.bn2(out)
# #         return out
# #
# #
# # class ECGNN(nn.Module):
# #     def __init__(self, args):
# #         super().__init__()
# #
# #         self.beat_len = 80  # 滑动窗口长度 目前100最优87.2
# #         self.start_filters = 32
# #         self.num_classes = args.num_classes
# #
# #         self.conv1 = nn.Conv1d(
# #             in_channels=1,  # 输入通道一定为1
# #             out_channels=32,  # 输出通道为卷积核的个数
# #             kernel_size=7,  # 卷积核大小为7 (实际为一个[1,7]大小的卷积核)  # 一般3/5/7
# #             stride=1,
# #             padding=3,
# #             bias=False,
# #         )
# #         self.relu = nn.ReLU(inplace=False)
# #         self.block = self._make_layer()
# #         self.beat_attn = attention(enc_hid_dim=128, dec_hid_dim=64)
# #         self.rhythm_attn = attention(enc_hid_dim=128, dec_hid_dim=64)
# #         self.gcn = GCN(args)
# #
# #         self.rweights = None  # 添加这个属性以保存注意力权重
# #         self.fc = nn.Linear(in_features=32, out_features=self.num_classes)  # need change  ##32
# #         # with gcn：160    only gcn：32     without gcn：128
# #         # self.fc11 = nn.Linear(in_features=128 * 12, out_features=2)  # 只有去掉GNN才用  ##128*12
# #         # self.fc1=nn.Linear(in_features=128*25, out_features=128)
# #         # self.fc2 = nn.Linear(in_features=128 * 19, out_features=128) #消融注意力
# #         # self.fc3 = nn.Linear(in_features=128 * 12, out_features=2)
# #
# #     def _make_layer(self, blocks_list: list = [3, 3, 3]) -> List[ResBlock]:
# #         layers = []
# #         downsample = None
# #
# #         num_filters = 32  # 第一个残差块输出通道数
# #         old_filters = num_filters  # old_filter用于跟踪上一个残差块输出通道数
# #         for i in range(len(blocks_list)):
# #             num_blocks = blocks_list[i]
# #             for j in range(num_blocks):
# #                 downsample = True if j == 0 and i != 0 else False
# #                 layers.append(
# #                     ResBlock(
# #                         in_channels=old_filters,
# #                         out_channels=num_filters,
# #                         downsample=downsample,
# #                     )
# #                 )
# #                 old_filters = num_filters
# #             num_filters *= 2
# #
# #         return nn.Sequential(*layers)
# #
# #     def forward(self, x, edge_index, batch):  # data
# #         def segment_signal_tensor(signal, window_size, overlap):
# #             segments = []
# #             start = 0
# #             while start + window_size <= signal.size(1):
# #                 segments.append(signal[:, start:start + window_size])
# #                 start += int(window_size * (1 - overlap))  # 根据重叠率计算步长
# #             return torch.stack(segments, dim=1)
# #
# #         # 输入x:
# #         # 384,1000 （384/12=32=batch，1次载入32个数据共384条导联）
# #
# #         # 滑动窗口分割x
# #         x = segment_signal_tensor(signal=x, window_size=self.beat_len, overlap=0.5)  # 384,39,50/384,19,100
# #         beat_num = x.size(1)
# #         # print("beat数量:",beat_num)
# #
# #         # 划分心段
# #         x = x.reshape(-1, self.beat_len).unsqueeze(1)
# #         # 上面384条导联一共有384*39=14976个分割段，每个段长50
# #         # 7291,1,100: 一共 7291 个元素，每个元素都是 1行100列 的数组
# #         # 提取特征
# #         x = self.conv1(x)  # 7296,32,100
# #         x = self.relu(x)  # (ReLU、Sigmoid Tanh等激活函数都不改变维度)
# #         x = self.block(x)  # 这是残差连接，会改变形状 14976,128,12
# #         # print("block后的x:", x.shape)
# #         # x=F.dropout(x, p=0.5, training=self.training)
# #         global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化降维
# #         x = global_avg_pool(x)  # 7296,128,1
# #
# #         # # 不用注意力（消融实验）
# #         # x = x.reshape(-1, beat_num * 128)
# #         # x = self.fc2(x)
# #         # 用注意力
# #         x = x.reshape(-1, beat_num, 128)  # 384,19,128 合并特征表示从片段到导联级别：384个导联,每个导联19个片段,每个片段特征长度128维
# #         x, self.rweights = self.rhythm_attn(x)  # 384,128 提炼导联特征
# #
# #         gcn_out = self.gcn(x, edge_index, batch)  # (384,128)→(32, 2)
# #         # with gnn
# #         out = self.fc(gcn_out)
# #
# #         # # 下面2句是去掉GNN的输出
# #         # x1 = x.reshape(-1, 12 * 128)
# #         # x1 = self.fc11(x1)
# #         # out = x1
# #         # 注意力不太对
# #         # x2=x.reshape(-1,12,128)
# #         # x2, _ = self.beat_attn(x2)
# #         # x2=self.fc11(x2)
# #         # 别的降维
# #
# #         # without gnn
# #         # out = x2
# #
# #         return out
#
#
# from typing import List
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from gnn.gnn_models2 import GCN #这是9951的模型
# # from gnn.gnn_models import GCN
# from torch_geometric.nn import (global_mean_pool as gap, global_max_pool as gmp, SAGPooling,
#                                 TopKPooling)
# from torch_geometric.nn import GCNConv, GATv2Conv
#
#
# class attention(nn.Module):
#     def __init__(self, enc_hid_dim, dec_hid_dim):
#         super().__init__()
#         self.attn = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
#         self.v = nn.Linear(dec_hid_dim, 1, bias=False)
#
#     def forward(self, enc_output):
#         energy = torch.tanh(self.attn(enc_output))
#         attention = self.v(energy)
#         scores = F.softmax(attention, dim=1)
#         out = enc_output * scores
#         return torch.sum(out, dim=1), scores
#
# class GCN(torch.nn.Module):
#     def __init__(self, args):
#         super(GCN, self).__init__()
#         self.args = args
#         self.num_features = 128 #args.num_features  # 128
#         self.nhid = args.nhid  # 64
#         self.num_classes = args.num_classes
#         # self.pooling_ratio = args.pooling_ratio
#         self.dropout_ratio = args.dropout_ratio
#         self.lead_attn = attention(enc_hid_dim=64, dec_hid_dim=64)
#
#         self.conv1 = GCNConv(self.num_features, self.nhid)
#         self.conv2 = GCNConv(self.nhid, self.nhid)
#         self.conv3 = GCNConv(self.nhid, self.nhid)
#
#         # 定义一个线性层，将输入维度为128，输出维度为32*32
#         self.lin3 = torch.nn.Linear(128, 64)
#         self.lin4 = torch.nn.Linear(64, 32)
#         self.lin = torch.nn.Linear(64, 32)
#
#
#     def forward(self, data_x, data_edge_index, data_batch):
#         x, edge_index, batch = data_x, data_edge_index, data_batch
#         edge_attr = None
#         x = F.relu(self.conv1(x, edge_index, edge_attr))# 384,64
#         x = F.relu(self.conv2(x, edge_index, edge_attr))
#         x = F.relu(self.conv3(x, edge_index, edge_attr))
#         x=x.reshape(-1,12,self.nhid) # 32,12,64
#         x1, lweights = self.lead_attn(x) # 32,64
#         # x = x * (lweights)  # 32,12,64 消融注意力要去掉
#         x=x1
#         # x=x.reshape(-1,64) # 384,64
#         # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 32,128
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)
#         x = F.relu(self.lin(x))
#
#         return x,lweights
#
# class ResBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, downsample: bool):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=7 if not downsample else 8,
#             stride=1 if not downsample else 2,
#             padding=3,
#             bias=False
#         )
#         self.bn1 = nn.BatchNorm1d(num_features=out_channels)
#         self.relu = nn.ReLU(inplace=True)  #
#         self.conv2 = nn.Conv1d(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=7,
#             stride=1,
#             padding=3,
#             bias=False
#         )
#         self.conv_down = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=2,
#             stride=2,
#             padding=0,
#             bias=False
#         )
#         self.bn2 = nn.BatchNorm1d(num_features=out_channels)
#         self.downsample = downsample
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x
#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.bn1(out)
#         out = self.conv2(out)
#         if self.downsample:
#             identity = self.conv_down(x)
#         out += identity
#         out = self.relu(out)
#         out = self.bn2(out)
#         return out
#
#
# class ECGNN(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#
#         self.beat_len = 80  # 滑动窗口长度 目前100最优87.2
#         self.start_filters = 32
#         self.num_classes = args.num_classes
#
#         self.conv1 = nn.Conv1d(
#             in_channels=1,  # 输入通道一定为1
#             out_channels=32,  # 输出通道为卷积核的个数
#             kernel_size=7,  # 卷积核大小为7 (实际为一个[1,7]大小的卷积核)  # 一般3/5/7
#             stride=1,
#             padding=3,
#             bias=False,
#         )
#         self.relu = nn.ReLU(inplace=False)
#         self.block = self._make_layer()
#         self.rhythm_attn = attention(enc_hid_dim=128, dec_hid_dim=64)
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#
#         self.gcn = GCN(args)
#
#
#         self.rweights = None  # 添加这个属性以保存注意力权重
#         self.lweights = None  # 添加这个属性以保存注意力权重
#         self.fc = nn.Linear(in_features=32, out_features=self.num_classes)  # need change  ##32
#         # with gcn：160    only gcn：32     without gcn：128
#         # self.fc11 = nn.Linear(in_features=128 * 12, out_features=2)  # 只有去掉GNN才用  ##128*12
#         # self.fc1=nn.Linear(in_features=128*25, out_features=128)
#         # self.fc2 = nn.Linear(in_features=128 * 24, out_features=128) #只有消融注意力才用
#         # self.fc3 = nn.Linear(in_features=128 * 12, out_features=2)
#         self.lin=nn.Linear(in_features=1000, out_features=128)
#
#     def _make_layer(self, blocks_list: list = [3, 3, 3]) -> List[ResBlock]:
#         layers = []
#         downsample = None
#
#         num_filters = 32  # 第一个残差块输出通道数
#         old_filters = num_filters  # old_filter用于跟踪上一个残差块输出通道数
#         for i in range(len(blocks_list)):
#             num_blocks = blocks_list[i]
#             for j in range(num_blocks):
#                 downsample = True if j == 0 and i != 0 else False
#                 layers.append(
#                     ResBlock(
#                         in_channels=old_filters,
#                         out_channels=num_filters,
#                         downsample=downsample,
#                     )
#                 )
#                 old_filters = num_filters
#             num_filters *= 2
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x, edge_index, batch):  # data
#         def segment_signal_tensor(signal, window_size, overlap):
#             segments = []
#             start = 0
#             while start + window_size <= signal.size(1):
#                 segments.append(signal[:, start:start + window_size])
#                 start += int(window_size * (1 - overlap))  # 根据重叠率计算步长
#             return torch.stack(segments, dim=1)
#
#         # 输入x:
#         # 384,1000 （384/12=32=batch，1次载入32个数据共384条导联）
#
#         # 滑动窗口分割x
#         x = segment_signal_tensor(signal=x, window_size=self.beat_len, overlap=0.5)  # 384,39,50/384,19,100
#         beat_num = x.size(1)
#         # print("beat数量:",beat_num)
#
#         # 划分心段
#         x = x.reshape(-1, self.beat_len).unsqueeze(1)
#         # 上面384条导联一共有384*39=14976个分割段，每个段长50
#         # 7291,1,100: 一共 7291 个元素，每个元素都是 1行100列 的数组
#         # 提取特征
#         x = self.conv1(x)  # 7296,32,100
#         x = self.relu(x)  # (ReLU、Sigmoid Tanh等激活函数都不改变维度)
#         x = self.block(x)  # 这是残差连接，会改变形状 14976,128,12
#         x = self.global_avg_pool(x)  # 7296,128,1
#
#         # # 不用注意力（消融实验）
#         # x = x.reshape(-1, beat_num * 128)
#         # x = self.fc2(x)
#         # 用注意力
#         x = x.reshape(-1, beat_num, 128)  # 384,19,128 合并特征表示从片段到导联级别：384个导联,每个导联19个片段,每个片段特征长度128维
#         x, self.rweights = self.rhythm_attn(x)  # 384,128 提炼导联特征
#
#
#
#         # x=self.lin(x)
#         gcn_out,self.lweights = self.gcn(x, edge_index, batch)  # (384,128)→(32, 2)
#         ## with gnn
#         out = self.fc(gcn_out)
#
#         # # 下面2句是去掉GNN的输出
#         # x1 = x.reshape(-1, 12 * 128)
#         # x1 = self.fc11(x1)
#         # out = x1
#
#         # without gnn
#         # out = x2
#
#         return out


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
        self.num_features = 128 #args.num_features  # 128
        self.nhid = args.nhid  # 64
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
        x=x.reshape(-1,12,self.nhid) # 32,12,64
        _, lweights = self.lead_attn(x) # 32,64
        x = x * (lweights)  # 32,12,64 消融注意力要去掉
        x=x.reshape(-1,64) # 384,64
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 32,128
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

        self.beat_len = 80  # 滑动窗口长度 目前100最优87.2
        self.start_filters = 32
        self.num_classes = args.num_classes

        self.conv1 = nn.Conv1d(
            in_channels=1,  # 输入通道一定为1
            out_channels=32,  # 输出通道为卷积核的个数
            kernel_size=7,  # 卷积核大小为7 (实际为一个[1,7]大小的卷积核)  # 一般3/5/7
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
        self.fc = nn.Linear(in_features=32, out_features=self.num_classes)  # need change  ##32
        # with gcn：160    only gcn：32     without gcn：128
        # self.fc11 = nn.Linear(in_features=128 * 12, out_features=2)  # 只有去掉GNN才用  ##128*12
        # self.fc2 = nn.Linear(in_features=128 * 24, out_features=128) #只有消融注意力才用

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
        x = segment_signal_tensor(signal=x, window_size=self.beat_len, overlap=0.5)  # 384,39,50/384,19,100
        beat_num = x.size(1)
        x = x.reshape(-1, self.beat_len).unsqueeze(1)
        x = self.conv1(x)  # 7296,32,100
        x = self.relu(x)  # (ReLU、Sigmoid Tanh等激活函数都不改变维度)
        x = self.block(x)  # 这是残差连接，会改变形状 14976,128,12
        x = self.global_avg_pool(x)  # 7296,128,1
        x = x.reshape(-1, beat_num, 128)  # 384,19,128 合并特征表示从片段到导联级别：384个导联,每个导联19个片段,每个片段特征长度128维
        x, self.rweights = self.beat_attn(x)  # 384,128 提炼导联特征
        gcn_out,self.lweights = self.gcn(x, edge_index, batch)  # (384,128)→(32, 2)
        out = self.fc(gcn_out)

        return out