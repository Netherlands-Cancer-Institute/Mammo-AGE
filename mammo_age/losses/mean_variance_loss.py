# MV loss from

from torch import nn
import math
import torch
import torch.nn.functional as F


class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1=0.2, lambda_2=0.05, cumpet_ce_loss=True, start_age=0):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cumpet_ce_loss = cumpet_ce_loss
        self.start_age = start_age

    def forward(self, input, target):
        # target = torch.round(target * 100).long()
        # N = input.size()[0]
        class_dim = input.shape[-1]
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
        # target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # mean loss
        a = torch.arange(class_dim, dtype=torch.float32).cuda()
        # mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1) / 100
        mean = torch.squeeze((p * (a + self.start_age)).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        # variance loss
        # b = (a[None, :] / 100 - mean[:, None])**2
        b = (a[None, :] - mean[:, None])**2
        variance_loss = (p * b).sum(1, keepdim=True).mean()

        if self.cumpet_ce_loss:
            # CE loss
            # target = torch.round(target * 100).long()
            target = torch.round(target).long()
            ce_loss = F.cross_entropy(input.view(-1, class_dim), target - self.start_age)
            loss = ce_loss + (self.lambda_1 * mean_loss) + (self.lambda_2 * variance_loss)
        else:
            loss = (self.lambda_1 * mean_loss) + (self.lambda_2 * variance_loss)
        return loss
