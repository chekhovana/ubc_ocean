import torch
from torch import nn
import timm
import yaml
import hydra
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


class NoisyOrPooling(nn.Module):
    def forward(self, x):
        x = torch.prod(1 - x, dim=0)
        x = 1 - x
        return x


class MaxPooling(nn.Module):
    def forward(self, x):
        return x.max(dim=0).values


class MultipleInstanceModel(nn.Module):
    def __init__(self, backbone, pooling, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.backbone = backbone
        self.pooling = pooling
        pass

    # def load_checkpoint(self, checkpoint_path):
    #     map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     checkpoint = torch.load(checkpoint_path, map_location=map_location)
    #     checkpoints = {f'heads.{i}.': dict() for i in range(len(self.heads))}
    #     checkpoints['backbone.'] = dict()
    #     for key in checkpoint.keys():
    #         for cpk in checkpoints.keys():
    #             if key.startswith(cpk):
    #                 checkpoints[cpk][key.replace(cpk, '')] = checkpoint[key]
    #     self.backbone.load_state_dict(checkpoints['backbone.'])
    #     for i, head in enumerate(self.heads):
    #         head.load_state_dict(checkpoints[f'heads.{i}.'])

    def forward(self, x):
        from tqdm import tqdm
        ds = TensorDataset(x)
        loader = DataLoader(ds, batch_size=self.batch_size)
        result = []
        for batch in loader:
            batch = batch[0]
            batch = self.backbone(batch)
            batch = torch.sigmoid(batch)
            result.append(batch)
        result = torch.cat(result)
        result = self.pooling(result)
        return result


class MilAttentionModel(nn.Module):
    def __init__(self, backbone, batch_size, num_classes=5):
        super().__init__()
        self.batch_size = batch_size
        self.backbone = backbone
        self.L = backbone.num_features
        self.D = 128
        self.K = 1
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(self.L * self.K, num_classes),
            nn.Softmax()
        )
        pass

    def forward(self, x):
        H = self.backbone(x)  # [N, L]
        A = self.attention(H)  # [N, K]
        A = torch.transpose(A, 0, 1)  # [K, N]
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # [K, L]
        result = self.classifier(M)
        # result = self.attention(result)
        return result


class Attention(nn.Module):
    def __init__(self, L, D, K):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, K)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(L * K, 1),
            nn.Sigmoid()
        )

    def forward(self, H):
        A = self.attention(H)  # [N, K]
        A = torch.transpose(A, 0, 1)  # [K, N]
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # [K, L]
        result = self.classifier(M)
        return result

    pass


class GatedAttention(nn.Module):
    def __init__(self, L, D, K):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            # nn.Sigmoid()
        )

    def forward(self, H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        return self.classifier(M)


class MilAttentionMulticlassModel(nn.Module):
    def __init__(self, backbone, batch_size, num_classes=5):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.backbone = backbone
        self.L = backbone.num_features
        self.D = 128
        self.K = 1
        self.attentions = nn.ModuleList()
        for i in range(num_classes):
            a = GatedAttention(self.L, self.D, self.K)
            self.attentions.append(a)
        pass

    def forward(self, x):
        H = self.backbone(x)  # [N, L]
        outputs = []
        for i in range(self.num_classes):
            outputs.append(self.attentions[i](H))
            pass  # [N, K]
        # result = self.attention(result)
        result = torch.stack(outputs)
        result = F.softmax(result, dim=0).squeeze()
        return result


# def forward(self, x):
#     ds = TensorDataset(x)
#     loader = DataLoader(ds, batch_size=self.batch_size)
#     embeddings, attentions = [], []
#     for batch in loader:
#         batch = batch[0]
#         H = self.backbone(batch)  # [N, L]
#         embeddings.append(H)
#         A = self.attention(H)  # [N, K]
#         attentions.append(A)
#     H = torch.cat(embeddings)
#     A = torch.cat(attentions)
#     A = torch.transpose(A, 0, 1)
#     A = F.softmax(A, dim=1)  # softmax over N
#     M = torch.mm(A, H)  # [K, L]
#     result = self.classifier(M)
#     # result = self.attention(result)
#     return result


# class MilLogitAttentionModel(nn.Module):
#     def __init__(self, backbone, batch_size):
#         super().__init__()
#         self.batch_size = batch_size
#         self.backbone = backbone
#         self.L = backbone.num_features
#         self.D = 128
#         self.K = 1
#         self.attention = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh(),
#             nn.Linear(self.D, self.K)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Flatten(start_dim=0),
#             nn.Linear(self.L * self.K, self.K),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         ds = TensorDataset(x)
#         loader = DataLoader(ds, batch_size=self.batch_size)
#         embeddings, attentions = [], []
#         for batch in loader:
#             batch = batch[0]
#             H = self.backbone(batch)  # [N, L]
#             embeddings.append(H)
#             A = self.attention(H)  # [N, K]
#             attentions.append(A)
#             torch.cuda.empty_cache()
#         H = torch.cat(embeddings)
#         A = torch.cat(attentions)
#         A = torch.transpose(A, 0, 1)
#         A = F.softmax(A, dim=1)  # softmax over N
#         M = torch.mm(A, H)  # [K, L]
#         result = self.classifier(M)
#         return result

# size_224_overlap_10 11940374
# size_256_overlap_10_threshold_50 47802
if __name__ == '__main__':
    t = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch.float32)
    model = nn.Flatten(start_dim=0)
    y = model(t)
    print(F.softmax(t, dim=0))
    print(F.softmax(t, dim=1))
    pass
