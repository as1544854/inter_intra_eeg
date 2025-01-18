import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnableSelfLoopLaplacian(nn.Module):
    def __init__(self, num_nodes):
        super(LearnableSelfLoopLaplacian, self).__init__()
        self.self_loop_weight = nn.Parameter(torch.ones(num_nodes))

    def forward(self, adjacency_matrix):
        device = adjacency_matrix.device
        adjacency_matrix_with_self_loop = adjacency_matrix + torch.diag(self.self_loop_weight).to(device)
        degree_matrix = torch.diag(torch.sum(adjacency_matrix_with_self_loop, dim=1))
        degree_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree_matrix.diag()))
        laplacian_matrix = torch.eye(adjacency_matrix.size(0), device=device) - degree_inv_sqrt @ adjacency_matrix_with_self_loop @ degree_inv_sqrt
        laplacian_matrix = torch.nan_to_num(laplacian_matrix)
        return laplacian_matrix

class GCNLayer(nn.Module):
    def __init__(self, adj_matrix, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.laplacian_generator = LearnableSelfLoopLaplacian(adj_matrix.size(0))
        self.adj_matrix = adj_matrix
        self.linear = nn.Linear(input_dim, output_dim)
        self.transformer = TransformerEncoderLayer(input_dim, 4)
        self.alpha = 0.5
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, inputs):
        H_0 = inputs

        trans = self.transformer(inputs)

        batch_size, num_nodes, _ = trans.size()
        trans = trans.permute(1, 0, 2).contiguous().view(num_nodes, -1)


        laplacian = self.laplacian_generator(self.adj_matrix.float().to(inputs.device))

        laplacian_mul = torch.mm(laplacian, trans)
        laplacian_mul = laplacian_mul.view(num_nodes, batch_size, -1).permute(1, 0, 2)

        outputs = (1 - self.alpha) * laplacian_mul + self.alpha * H_0
        outputs = F.relu(self.linear(outputs))

        return outputs

class MultiLevelAttentionGRU(nn.Module):


    def __init__(self, input_size, hidden_size, num_layers=2, dropout_rate=0.3):
        super(MultiLevelAttentionGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout_rate)

        self.time_attention = nn.Linear(hidden_size * 2, 1)
        self.feature_attention = nn.Linear(hidden_size * 2, hidden_size * 2)

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):

        gru_output, _ = self.gru(x)
        gru_output = self.layer_norm(gru_output)

        time_attention_scores = F.softmax(self.time_attention(gru_output), dim=1)
        time_weighted = torch.sum(time_attention_scores * gru_output, dim=1)

        feature_attention_scores = torch.sigmoid(self.feature_attention(time_weighted))
        feature_weighted = feature_attention_scores * time_weighted

        return feature_weighted

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        weights = F.softmax(F.relu(self.attention(x)), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerEncoderLayer, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout_rate=0.3):
        super(GRUBlock, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        x, _ = self.gru(x)  # GRU 层
        x = self.layer_norm(x)
        return x

class Model(nn.Module):
    def __init__(self,  adj_matrix, input_dim, hidden_dim, output_dim, nhead=4):
        super(Model, self).__init__()
        self.gcn_layers = GCNLayer(adj_matrix, input_dim, hidden_dim)
        self.gcn_layers2=GCNLayer(adj_matrix, hidden_dim,hidden_dim)
        self.gcn_layers3=GCNLayer(adj_matrix, hidden_dim,output_dim)
        self.transformer = TransformerEncoderLayer(hidden_dim, nhead)
        self.gru = GRUBlock(input_size=20, hidden_size=hidden_dim // 2, num_layers=2, dropout_rate=0.3)
        self.multi_attention_gru = MultiLevelAttentionGRU(input_size=20, hidden_size=hidden_dim // 2,num_layers=2, dropout_rate=0.3)
        self.pooling = SelfAttentionPooling(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs
        x = self.gcn_layers(x)
        x = self.gcn_layers2(x)
        x = x.permute(0, 2, 1)
        x = self.multi_attention_gru(x)
        x = self.fc(x)
        return x.squeeze()

# 示例代码
if __name__ == "__main__":
    batch_size = 32
    num_channels = 20
    num_points = 512
    input_dim = num_points
    hidden_dim = 64
    output_dim = 1
    num_layers = 3

    adj_matrix = torch.eye(num_channels)
    model = Model(num_layers, adj_matrix, input_dim, hidden_dim, output_dim).cuda()
    inputs = torch.randn(batch_size, num_channels, num_points).cuda()
    output = model(inputs)
    print(output.shape)
