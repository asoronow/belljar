import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def reinhard_normalization(img, target_means, target_stds):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_l, img_a, img_b = cv2.split(img_lab)

    img_means, img_stds = [], []
    for channel in [img_l, img_a, img_b]:
        img_means.append(np.mean(channel))
        img_stds.append(np.std(channel))

    img_l = ((img_l - img_means[0]) / img_stds[0]) * target_stds[0] + target_means[0]
    img_a = ((img_a - img_means[1]) / img_stds[1]) * target_stds[1] + target_means[1]
    img_b = ((img_b - img_means[2]) / img_stds[2]) * target_stds[2] + target_means[2]

    img_normalized = cv2.merge((img_l, img_a, img_b))
    img_normalized = cv2.cvtColor(img_normalized.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return img_normalized


# Target means and stds for normalization
TARGET_MEANS = [128, 128, 128]
TARGET_STDS = [50, 50, 50]


class StainNormalization(nn.Module):
    def __init__(self):
        super(StainNormalization, self).__init__()

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1, branch3, branch5, branch_pool]
        return torch.cat(outputs, 1)


class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, expert_output_dim):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                self._create_expert(input_dim, expert_output_dim)
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(input_dim, num_experts)

    def _create_expert(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        gate_weights = F.softmax(self.gate(x.view(x.size(0), -1)), dim=1)
        expert_outputs = [expert(x) for expert in self.experts]
        weighted_sum = sum(
            [
                gate_weights[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                * expert_outputs[i]
                for i in range(self.num_experts)
            ]
        )
        return weighted_sum


class BrainSliceAnglePredictor(nn.Module):
    def __init__(self):
        super(BrainSliceAnglePredictor, self).__init__()
        self.stain_norm = StainNormalization()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(128, 128) for _ in range(12)])
        self.inception_blocks = nn.Sequential(*[InceptionModule(128) for _ in range(4)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.moe = MoE(input_dim=256, num_experts=8, expert_output_dim=256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_angle_x = nn.Linear(256, 1)
        self.fc_angle_y = nn.Linear(256, 1)
        self.fc_depth = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stain_norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.res_blocks(x)
        x = self.inception_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.moe(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output_x = self.fc_angle_x(x)
        output_y = self.fc_angle_y(x)
        output_depth = self.fc_depth(x)
        return output_x, output_y, output_depth


# Instantiate the model with input size 256x256
model = BrainSliceAnglePredictor()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation loops will go here
