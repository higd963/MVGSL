import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        agg = torch.matmul(adj, x)
        out = self.linear(agg)
        return F.relu(out)

class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes, input_dim, topk=5, use_topk=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, num_nodes)
        self.topk = topk
        self.use_topk = use_topk

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        adj_soft = torch.clamp(F.gumbel_softmax(logits, tau=1, hard=False, dim=-1), min=1e-8)

        if self.use_topk:
            topk_vals, topk_indices = logits.topk(self.topk, dim=-1)
            mask = torch.zeros_like(logits)
            B, N = logits.shape[:2]
            batch_idx = torch.arange(B, device=logits.device).view(-1, 1, 1).expand(-1, N, self.topk)
            node_idx = torch.arange(N, device=logits.device).view(1, -1, 1).expand(B, -1, self.topk)
            mask[batch_idx, node_idx, topk_indices] = 1.0
            adj = adj_soft * mask
        else:
            adj = adj_soft
        return adj, adj_soft

class SimilarityAdjacency(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        return torch.matmul(x_norm, x_norm.transpose(-1, -2))

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn_micro_to_macro = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_macro_to_micro = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.skip_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, h_micro, h_macro):
        B, N, T, H = h_micro.shape

        h_micro_ = h_micro.permute(0, 2, 1, 3).contiguous().view(B * T, N, H)
        h_macro_ = h_macro.permute(0, 2, 1, 3).contiguous().view(B * T, N, H)

        out1, _ = self.attn_micro_to_macro(h_micro_, h_macro_, h_macro_)
        out2, _ = self.attn_macro_to_micro(h_macro_, h_micro_, h_micro_)

        skip_input = torch.cat([out1, out2], dim=-1)  # [B*T, N, 2H]
        skip_input_proj = self.skip_proj(skip_input)  # [B*T, N, H]

        gate_input = torch.cat([skip_input_proj, out1 + out2], dim=-1)  # [B*T, N, 2H]
        alpha = self.gate(gate_input)  # [B*T, N, H]

        fused = alpha * skip_input_proj + (1 - alpha) * (out1 + out2)  # [B*T, N, H]

        return fused.view(B, T, N, H).permute(0, 2, 1, 3).contiguous()


class TemporalModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        B, N, T, D = x.shape
        x = x.view(B * N, T, D)
        out, _ = self.gru(x)
        return out.view(B, N, T, -1)

class ContrastiveHybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_nodes = config['num_nodes']
        self.hidden_dim = config['h_dim']
        self.output_dim = config['output_dim']
        self.seq_len = config['seq_len']
        self.num_tasks = 3
        self.use_modality_alignment = config.get('use_modality_alignment', False)

        self.weather_encoder = TemporalModalityEncoder(5, self.hidden_dim)
        self.date_encoder = TemporalModalityEncoder(3, self.hidden_dim)
        self.disease_encoder = TemporalModalityEncoder(3, self.hidden_dim)

        self.macro_adj = LearnableAdjacency(self.num_nodes, self.hidden_dim * 3) if config.get('macro_adj', 'gumbel') == 'gumbel' else SimilarityAdjacency()
        self.micro_adj = SimilarityAdjacency() if config.get('micro_adj', 'similarity') == 'similarity' else LearnableAdjacency(self.num_nodes, self.hidden_dim * 3)

        self.micro_gnn = GCNLayer(self.hidden_dim * 3, self.hidden_dim)
        self.macro_gnn = GCNLayer(self.hidden_dim * 3, self.hidden_dim)

        
        self.micro_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.macro_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.task_cross_attn = nn.ModuleList([
            CrossAttentionFusion(self.hidden_dim) for _ in range(self.num_tasks)
        ])

        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.gru_norm = nn.LayerNorm(self.hidden_dim)

        self.task_projections = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.output_dim)
            for _ in range(self.num_tasks)
        ])

        self.disease_projector = nn.Linear(1, self.hidden_dim)

    def modality_alignment_loss(self, h_weather, h_disease, temperature=0.1):
        B, N, T, H = h_weather.shape
        losses = []
        for var_idx in range(h_disease.size(-1)):
            z1 = F.normalize(h_weather.mean(dim=[1, 2]), dim=-1)
            disease_var = h_disease[..., var_idx].unsqueeze(-1)
            disease_proj = self.disease_projector(disease_var.mean(dim=[1, 2]))
            z2 = F.normalize(disease_proj, dim=-1)
            logits = torch.matmul(z1, z2.T) / temperature
            labels = torch.arange(B, device=logits.device)
            losses.append(F.cross_entropy(logits, labels))
        return sum(losses) / len(losses)

    def forward(self, x):
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()

        weather = x[..., :5]
        date = x[..., 5:8]
        disease = x[..., 8:]

        h_w = self.weather_encoder(weather)
        h_d = self.date_encoder(date)
        h_s = self.disease_encoder(disease)
        h = torch.cat([h_w, h_d, h_s], dim=-1)

        h_avg = h.mean(dim=2)
        adj_macro, _ = self.macro_adj(h_avg) if isinstance(self.macro_adj, LearnableAdjacency) else (self.macro_adj(h_avg), None)

        h_micro_seq, h_macro_seq = [], []
        for t in range(T):
            h_t = h[:, :, t, :]
            adj_t, _ = self.micro_adj(h_t) if isinstance(self.micro_adj, LearnableAdjacency) else (self.micro_adj(h_t), None)
            
            h_micro_t = self.micro_proj(self.micro_gnn(h_t, adj_t))
            h_macro_t = self.macro_proj(self.macro_gnn(h_t, adj_macro))

            h_micro_seq.append(h_micro_t)
            h_macro_seq.append(h_macro_t)

        h_micro_seq = torch.stack(h_micro_seq, dim=2)
        h_macro_seq = torch.stack(h_macro_seq, dim=2)

        task_outputs = []
        for task_idx in range(self.num_tasks):
            h_fused = self.task_cross_attn[task_idx](h_micro_seq, h_macro_seq)
            h_fused = h_fused.view(B * N, T, -1)
            gru_out, _ = self.gru(h_fused)
            gru_out = self.gru_norm(gru_out)

            out = self.task_projections[task_idx](gru_out)
            out = out.view(B, N, T, self.output_dim).permute(0, 2, 1, 3)
            task_outputs.append(out)

        outputs = torch.stack(task_outputs, dim=-2)

        loss_modal = self.modality_alignment_loss(h_w, h_s) if self.use_modality_alignment else torch.tensor(0.0, device=x.device, requires_grad=False)

        return outputs, torch.tensor(0.0, device=x.device), loss_modal
