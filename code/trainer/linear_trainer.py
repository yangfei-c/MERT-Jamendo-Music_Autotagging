import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from torchmetrics import AUROC, F1Score
from model.linear import Linear

# === Asymmetric Loss ===
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x_sigmoid.clamp(min=self.eps, max=1 - self.eps)
        pt = x_sigmoid * y + (1 - x_sigmoid) * (1 - y)
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.clip > 0:
            one_sided_w = one_sided_w * (1 - y) * (x_sigmoid > self.clip).float() + one_sided_w * y
        loss = - one_sided_w * torch.log(pt)
        return loss.mean()


class MTGJamendoTrainer(pl.LightningModule):
    def __init__(self, config, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        self.model = Linear(config, num_classes)
        self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)

        self.val_auroc = AUROC(task="multilabel", num_labels=num_classes, average='macro')
        self.val_f1 = F1Score(task="multilabel", num_labels=num_classes, average='macro')
        self.test_auroc = AUROC(task="multilabel", num_labels=num_classes, average='macro')

        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # 进行验证
        self.val_auroc(logits, y.long())
        self.val_f1(logits, y.long())

        # 指标记录情况
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        self.test_step_outputs.append({
            'predictions': probs.cpu(),
            'targets': y.cpu()
        })

        # 进行测试以及指标记录
        self.test_auroc(logits, y.long())
        self.log('test_auroc', self.test_auroc)

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            all_preds = torch.cat([x['predictions'] for x in self.test_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.test_step_outputs])

            # 构造保存路径
            save_dir = Path(self.cfg.paths.output_dir) / self.cfg.project.name
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / "results.pt"

            torch.save({
                'predictions': all_preds,  # 预测概率
                'targets': all_targets,  # 真实标签
                'auroc': self.test_auroc.compute().item()
            }, save_path)

            print(f"\n[Result] Test results saved to: {save_path}")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.cfg.training.scheduler.factor,
            patience=self.cfg.training.scheduler.patience,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }