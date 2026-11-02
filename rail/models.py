"""Main RAIL encoders and objective; computations preserved from the research code."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T


class InfoNCELoss(nn.Module):
    def __init__(self, init_logit_scale=1 / 0.05):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_logit_scale)))

    @property
    def scale(self):
        return self.logit_scale.exp().clamp(max=50.0)

    def forward(self, fA, fB):
        fA = F.normalize(fA, dim=-1)
        fB = F.normalize(fB, dim=-1)
        logits = self.scale * (fA @ fB.T)
        labels = torch.arange(fA.size(0), device=fA.device)
        loss_x2y = F.cross_entropy(logits, labels)
        loss_y2x = F.cross_entropy(logits.T, labels)
        loss = (loss_x2y + loss_y2x) / 2
        with torch.no_grad():
            preds_x2y = logits.argmax(dim=1)
            preds_y2x = logits.T.argmax(dim=1)
            contrastive_acc = (
                (preds_x2y == labels).float().mean()
                + (preds_y2x == labels).float().mean()
            ) / 2
        return (loss, contrastive_acc)

    def get_loss_with_duplicates(self, A, B, fA, fB):
        logits = self.scale * (fA @ fB.T)
        sim_matrix = (
            (A[:, None, :] == A[None, :, :]).all(-1)
            | (B[:, None, :] == B[None, :, :]).all(-1)
        ).float()
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum(dim=1, keepdim=True)
        prob_pos = exp_logits * sim_matrix / denom
        loss_x2y = -torch.log(prob_pos.sum(dim=1).clamp(min=1e-08)).mean()
        exp_logits_T = torch.exp(logits.T)
        denom_T = exp_logits_T.sum(dim=1, keepdim=True)
        prob_pos_T = exp_logits_T * sim_matrix.T / denom_T
        loss_y2x = -torch.log(prob_pos_T.sum(dim=1).clamp(min=1e-08)).mean()
        loss = (loss_x2y + loss_y2x) / 2
        with torch.no_grad():
            preds_x2y = logits.argmax(dim=1)
            preds_y2x = logits.argmax(dim=0)
            acc = (
                (
                    sim_matrix[torch.arange(len(A)), preds_x2y]
                    + sim_matrix[torch.arange(len(A)), preds_y2x]
                )
                / 2
            ).mean()
        return (loss, acc)


class ImageEncoder(nn.Module):
    def __init__(self, obs_s, emb_d):
        super().__init__()
        n_input_channels = obs_s.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_obs = torch.zeros(1, *obs_s.shape)
            n_flatten = self.cnn(sample_obs).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.LeakyReLU())

    def forward(self, x) -> torch.Tensor:
        x = self.cnn(x)
        x = self.linear(x)
        return x


class LinearEncoder(nn.Module):
    def __init__(self, obs_s, emb_d):
        super().__init__()
        obs_d = obs_s.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(obs_d, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class CommonNet(nn.Module):
    def __init__(self, emb_d):
        super(CommonNet, self).__init__()
        self.embs = nn.Embedding(2, 4)
        self.mlp = nn.Sequential(
            nn.Linear(512 + 4, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, emb_d),
        )

    def forward(self, x, flag):
        flag = torch.as_tensor(flag, device=x.device, dtype=torch.long)
        e = self.embs(flag).expand(x.size(0), -1)
        x = torch.cat([x, e], dim=-1)
        x = self.mlp(x)
        return x


class CombinedExtractors(nn.Module):
    def __init__(
        self, obs_s_dict, emb_d, normalize=True, detach_forward=False, augment=False
    ):
        super().__init__()
        self.features_dim = emb_d
        self.normalize = normalize
        self.detach_forward = detach_forward
        self.augment = augment
        self.update_noise(0.6)
        obs_space_s = obs_s_dict["student_obs"]
        obs_space_t = obs_s_dict["teacher_obs"]
        self.img_obs_s = len(obs_space_s.shape) == 3
        img_obs_t = len(obs_space_t.shape) == 3
        encoder_class_s = ImageEncoder if self.img_obs_s else LinearEncoder
        encoder_class_t = ImageEncoder if img_obs_t else LinearEncoder
        self.student_encoder = encoder_class_s(obs_space_s, emb_d)
        self.teacher_encoder = encoder_class_t(obs_space_t, emb_d)
        self.common_net = CommonNet(emb_d)
        self.aug = T.Compose(
            [T.RandomCrop((84, 84), padding=4), T.RandomRotation(degrees=10)]
        )
        self.forward_key = "teacher"

    def forward(self, x):
        if self.forward_key == "teacher":
            x = self.forward_teacher(x, dropout=False)
        elif self.forward_key == "student":
            x = self.forward_student(x, dropout=False)
        x = self._add_noise(x)
        if self.detach_forward:
            return x.detach()
        return x

    def forward_teacher(self, x, dropout=True):
        x = self.teacher_encoder(x["teacher_obs"])
        common_net = self.common_net
        x = common_net(x, 0)
        if self.normalize:
            x = self._normalize(x)
        return x

    def forward_student(self, x, dropout=True, augment=False):
        if self.augment and augment:
            x = self.aug(x)
        x = self.student_encoder(x["student_obs"])
        common_net = self.common_net
        x = common_net(x, 1)
        if self.normalize:
            x = self._normalize(x)
        return x

    def _normalize(self, x):
        return F.normalize(x, dim=-1)

    def _add_noise(self, x):
        n = 0.01 * torch.randn_like(x)
        return self._normalize(x + n)

    def update_noise(self, cos_sim):
        cs = max(0.4, cos_sim)
        cs_c = (
            cs + ((2 - cs**2) * (1 - cs**2) ** 2 / (2 * (self.features_dim - 1))) ** 0.5
        )
        var = (1.0 / cs_c**2 - 1.0) / (self.features_dim - 1.0)
        self.noise_std = var**0.5
