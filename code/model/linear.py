import torch.nn as nn
from omegaconf import DictConfig

class Linear(nn.Module):
    def __init__(self, config: DictConfig, num_classes: int):
        super().__init__()

        input_dim = config.data.input_dim
        dropout = config.model.dropout

        # 特征提取层 (Backbone) ===
        # 这一部分负责生成 256维 的 Embedding
        self.backbone = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # === 2. 分类层 (Classifier) ===
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        logits = self.classifier(x)
        return logits