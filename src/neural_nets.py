from torch import nn
from torchvision.models import efficientnet_b0
import torch
from torch import nn
import math
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

EFFICIENTNET_PATH = 'google/efficientnet-b0'
VIT_PATH = 'jayanta/vit-base-patch16-224-in21k-face-recognition'
EMBED_SIZE = 512

def param_count(model):
    return sum([p.numel() for name, p in model.named_parameters() if p.requires_grad])

class EmbedderNet(nn.Module):
    def __init__(self, embedding_size, backbone_name):
        super(EmbedderNet, self).__init__()
        
        if backbone_name == 'eff':
            self.backbone = AutoModel.from_pretrained(EFFICIENTNET_PATH)
            fc_in_f = 1280
        elif backbone_name == 'vit':
            self.backbone = AutoModel.from_pretrained(VIT_PATH)
            fc_in_f = 768
        else:
            raise KeyError
        
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(in_features=fc_in_f,
                      out_features=embedding_size)
        )

    def forward(self, x):
        hidden_st = self.backbone(x)
        out = self.classifier(hidden_st.pooler_output)
        return out
    
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device='cpu'):
        super(ArcMarginProduct, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=True).to(device)
        #self.weight.retain_grad()
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label, train=True):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if train:
            one_hot = torch.zeros(cosine.size(), device=self.device)
            one_hot = torch.scatter(one_hot, 1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        else:
            output = cosine
        output *= self.s

        #output.retain_grad()
        return output
    
class MetricNet(nn.Module):
    def __init__(self, head_name,  in_features, out_features, s=30.0, m=0.50, easy_margin=False, device='cpu'):
        super(MetricNet, self).__init__()

        self.h_name = head_name
        if self.h_name == 'arc':
            self.head = ArcMarginProduct(in_features, out_features, s, m, easy_margin, device)
        elif self.h_name == 'lin':
            self.head_ = nn.Linear(in_features, out_features).to(device)
            self.head = lambda x, y: self.head_(x)

    def forward(self, x, y):
        return self.head(x, y)