import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import math

# --- 1. Graph Layer (Graf Katmanı) ---
class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True):
        # node_dim=0: PyG'ye düğümlerin ilk boyutta olduğunu bildiriyoruz
        super(GraphLayer, self).__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
        # --- DÜZELTME BURADA (256 vs 128 Hatası) ---
        # GDN formülü: Embedding(d) + Feature(d) = 2d
        # Attention girişi: Source(2d) + Target(2d) = 4d
        # Bu yüzden parametre boyutu: 4 * out_channels olmalı (4 * 64 = 256)
        self.att = nn.Parameter(torch.Tensor(1, heads, 4 * out_channels))
        
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index, embedding):
        # x: [Batch, Node, Feature] -> [32, 51, 5]
        
        x = self.lin(x) 
        
        # [Batch, Node, Feature] -> [Node, Batch, Feature]
        x = x.permute(1, 0, 2)
        
        out = self.propagate(edge_index, x=x, embedding=embedding)
        
        if self.concat:
            out = F.relu(out)
            
        # [Node, Batch, Feature] -> [Batch, Node, Feature]
        return out.permute(1, 0, 2)

    def message(self, x_i, x_j, embedding_i, embedding_j, index):
        # Embedding'i Batch boyutuna genişlet
        # [Num_Edges, Dim] -> [Num_Edges, 1, Dim] -> [Num_Edges, Batch, Dim]
        embedding_i = embedding_i.unsqueeze(1).expand(-1, x_i.size(1), -1)
        embedding_j = embedding_j.unsqueeze(1).expand(-1, x_j.size(1), -1)

        # Denklem 6: g = Embedding || Feature
        g_i = torch.cat([embedding_i, x_i], dim=-1) # Boyut: 128
        g_j = torch.cat([embedding_j, x_j], dim=-1) # Boyut: 128
        
        # Denklem 7: Attention (g_i || g_j) -> Boyut: 256
        g_cat = torch.cat([g_i, g_j], dim=-1)
        
        # Hata burada çıkıyordu: 256'lık vektörü 256'lık parametreyle çarpmalıyız
        alpha = (g_cat * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)
        
        return x_j * alpha.unsqueeze(-1)


# --- 2. GDN Ana Model ---
class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, input_dim=5, 
                 out_layer_inter_dim=256, topk=15):
        
        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.embedding_dim = dim
        self.topk = topk

        self.embedding = nn.Parameter(torch.randn(node_num, dim))
        
        self.gnn_layer = GraphLayer(input_dim, dim, heads=1, concat=False)

        self.out_layer = nn.Sequential(
            nn.Linear(dim, out_layer_inter_dim),
            nn.ReLU(),
            nn.Linear(out_layer_inter_dim, 1)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding)

    def get_dependency_graph(self):
        # L2 normalizasyon ile embedding benzerliği hesapla
        norm = self.embedding.norm(p=2, dim=1, keepdim=True)
        # Bölme hatasını önlemek için epsilon ekle
        normalized_embedding = self.embedding.div(norm + 1e-8)
        similarity = torch.mm(normalized_embedding, normalized_embedding.t())
        
        topk_values, topk_indices = torch.topk(similarity, k=self.topk, dim=1)
        
        device = self.embedding.device
        node_indices = torch.arange(self.node_num).unsqueeze(1).expand(self.node_num, self.topk).flatten().to(device)
        topk_indices = topk_indices.flatten()
        
        edge_index = torch.stack([topk_indices, node_indices])
        return edge_index

    def forward(self, x):
        edge_index = self.get_dependency_graph()
        
        gcn_out = self.gnn_layer(x, edge_index, embedding=self.embedding)
        
        # Güvenlik Kilidi
        if gcn_out.shape[0] == self.node_num and gcn_out.shape[1] == x.shape[0]:
            gcn_out = gcn_out.permute(1, 0, 2)
        
        batch_num = x.shape[0]
        embedding_expanded = self.embedding.unsqueeze(0).expand(batch_num, -1, -1)
        
        out = torch.mul(gcn_out, embedding_expanded)
        out = self.out_layer(out)
        
        return out.squeeze(-1)