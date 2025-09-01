# POE method are adjusted from https://github.com/Li-Wanhua/POEs
# Global and local transformer are adjusted from https://github.com/shengfly/global-local-transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from itertools import chain
from torch import Tensor
from mammo_age.models.transformerModel import Instance_bag_transformer
GLOBAL_DROP_RATE = 0.5
# GLOBAL_DROP_RATE = False

class Mammo_AGE(nn.Module):
    """
    Mammo_AGE model class.

    Args:
        arch (str): Model architecture. Default is 'resnet18'.
        num_output (int): Number of output neurons. Default is 100.
        nblock (int): Number of transformer blocks. Default is 10.
        hidden_size (int): Hidden size for the transformer. Default is 128.
        second_stage (bool): Whether to use the second stage. Default is False.
        first_stage_model (str): Path to the first stage model. Default is None.
    """
    def __init__(self, arch: str = 'resnet18', num_output: int = 100, nblock: int = 10,
                 hidden_size: int = 128, second_stage: bool = False, first_stage_model: str = None):

        super(Mammo_AGE, self).__init__()

        self.backbone_model = POE_Baselie_Model(arch, num_output, backbone_model=True)
        in_dim = self.backbone_model.num_feat

        self.Instance_bag_trans = Instance_bag_transformer(
            in_dim=in_dim, hidden_dim=hidden_size, num_blocks=nblock,
            drop_rate=GLOBAL_DROP_RATE if GLOBAL_DROP_RATE else 0.3)

        self.avg = nn.AdaptiveAvgPool2d(1)

        out_hidden_size = hidden_size + in_dim
        self.emd = nn.Sequential(nn.Linear(out_hidden_size + 4, out_hidden_size), nn.ReLU(True), )
        self.var = nn.Sequential(nn.Linear(out_hidden_size + 4, out_hidden_size),
                                 nn.BatchNorm1d(out_hidden_size, eps=0.001, affine=False), )

        self.locout = nn.Sequential(
            nn.Dropout(GLOBAL_DROP_RATE if GLOBAL_DROP_RATE else 0.1), # 0.1
            nn.Linear(out_hidden_size, num_output), )

        self.second_stage = second_stage
        if self.second_stage:
            self._load_weights(first_stage_model)

    def forward(self, x: Tensor, max_t: int = 50, use_sto: bool = False):
        """
        Forward pass of the Mammo_AGE model.

        Args:
            x: Input tensor.
            max_t (int): Maximum number of stochastic samples. Default is 50.
            use_sto (bool): Whether to use stochastic sampling. Default is False.

        Returns:
            Tuple containing global pathway, local pathway, global embedding, local embedding, global log variance, local log variance, and density output.
        """
        local_feat_list, xglo, glo, glo_emb, glo_log_var, density_out = self.backbone_model(x, max_t, use_sto)
        emblist, log_varlist, outlist = [glo_emb], [glo_log_var], [glo]

        loc_list = self.Instance_bag_trans(local_feat_list, xglo)

        for indx_loc in range(len(loc_list)):
            xloc = loc_list[indx_loc]
            xloc = torch.flatten(self.avg(xloc), 1)
            xloc = torch.cat([xloc, density_out], dim=1)
            loc_emb = self.emd(xloc)
            emblist.append(loc_emb)
            loc_log_var = self.var(xloc)
            log_varlist.append(loc_log_var)
            if use_sto:
                loc_sqrt_var = torch.exp(loc_log_var * 0.5)
                loc_rep_emb = loc_emb[None].expand(max_t, *loc_emb.shape)
                loc_rep_sqrt_var = loc_sqrt_var[None].expand(max_t, *loc_sqrt_var.shape)
                loc_norm_v = torch.randn_like(loc_rep_emb, device=loc_rep_emb.device)
                loc_sto_emb = loc_rep_emb + loc_rep_sqrt_var * loc_norm_v
                out = self.locout(loc_sto_emb)
            else:
                out = self.locout(loc_emb)
            outlist.append(out)

        global_pathway = outlist[0]
        local_pathway = outlist[1:]

        global_emb = emblist[0]
        local_emb = emblist[1:]

        global_log_var = log_varlist[0]
        local_log_var = log_varlist[1:]

        return global_pathway, local_pathway, global_emb, local_emb, global_log_var, local_log_var, density_out

    def _load_weights(self, model_path: str):
        """
        Load weights from a checkpoint.

        Args:
            model_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(model_path)
        self.backbone_model.load_state_dict(checkpoint['state_dict'], strict=True)


class POE_Baselie_Model(nn.Module):
    """
    POE_Baselie_Model class.

    Args:
        args: Arguments for the model.
        backbone_model (bool): Whether to use the model as a backbone. Default is False.
    """
    def __init__(self, arch: str, num_output: int, backbone_model: bool = False):
        super(POE_Baselie_Model, self).__init__()
        self.backbone_model = backbone_model
        self.num_feat = self._initialize_model(arch)

        # Simplified layers
        self.latent = nn.Sequential(
            nn.Dropout(p=GLOBAL_DROP_RATE if GLOBAL_DROP_RATE else 0.2),
            nn.Linear(self.num_feat * 4, self.num_feat), nn.ReLU(True))
        self.emd = nn.Sequential(nn.Linear(self.num_feat + 4, self.num_feat), nn.ReLU(True))
        self.var = nn.Sequential(nn.Linear(self.num_feat + 4, self.num_feat),
                                 nn.BatchNorm1d(self.num_feat, eps=0.001, affine=False),)
        self.drop = nn.Dropout(p=GLOBAL_DROP_RATE if GLOBAL_DROP_RATE else 0.1)
        self.final = nn.Linear(self.num_feat, num_output)

        # Multi-task (breast density prediction) branches
        self.predict_density = nn.Sequential(
            nn.Dropout(p=GLOBAL_DROP_RATE if GLOBAL_DROP_RATE else 0.2),
            nn.Linear(self.num_feat * 4, self.num_feat),
            nn.Dropout(p=GLOBAL_DROP_RATE if GLOBAL_DROP_RATE else 0.1),
            nn.ReLU(True),
            nn.Linear(self.num_feat, 4)
        )

        self.avg = nn.AdaptiveAvgPool2d(1)
        self._initialize_weights()

    def _initialize_model(self, arch: str):
        """
        Initialize the model based on the architecture.

        Args:
            arch (str): Model architecture.

        Returns:
            int: Number of features.
        """
        print(f"=> creating model '{arch}'")
        # model = models.__dict__[arch](pretrained=True)
        model = models.__dict__[arch](weights='IMAGENET1K_V1')

        # Get feature dimension
        if 'densenet' in arch:
            num_feat = model.classifier.in_features
        elif 'resnet' in arch:
            num_feat = model.fc.in_features
        elif 'vgg' in arch or 'convnext' in arch or 'efficientnet' in arch:
            num_feat = model.classifier[-1].in_features
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Remove unnecessary layers
        self.model = nn.Sequential(*[
            module for name, module in model.named_children()
            if name not in ['avgpool', 'classifier', 'fc']
        ])
        return num_feat

    def forward(self, x: Tensor, max_t: int = 50, use_sto: bool = False):
        """
        Forward pass of the POE_Baselie_Model.

        Args:
            x: Input tensor.
            max_t (int): Maximum number of stochastic samples. Default is 50.
            use_sto (bool): Whether to use stochastic sampling. Default is False.

        Returns:
            Tuple containing logits, embeddings, log variance, and density predictions.
        """
        assert x.dim() == 5, f"Input must be 5D tensor, got {x.shape}"
        assert x.size(2) == 1, f"Expected single channel input, got {x.size(2)} channels"
        x = x.repeat(1, 1, 3, 1, 1)  # [B, 4, 3, H, W]

        # Process each view
        local_feats = []
        for i in range(4):
            feat = self.model(x[:, i])  # [B, C, h, w]
            local_feats.append(feat)

        # Feature aggregation
        flattened_feats = [torch.flatten(self.avg(feat), 1) for feat in local_feats]
        out = torch.cat(flattened_feats, dim=1)  # [B, num_feat*4]

        # Density prediction (simplified)
        density_x = self.predict_density(out)  # Directly use aggregated features

        # Main pathway
        out_latent = self.latent(out)

        # Concatenate density prediction
        out_latent = torch.cat([out_latent, density_x], dim=1)

        emb = self.emd(out_latent)
        log_var = self.var(out_latent)
        sqrt_var = torch.exp(log_var * 0.5)
        if use_sto:
            rep_emb = emb[None].expand(max_t, *emb.shape)
            rep_sqrt_var = sqrt_var[None].expand(max_t, *sqrt_var.shape)
            norm_v = torch.randn_like(rep_emb, device=emb.device)
            sto_emb = rep_emb + rep_sqrt_var * norm_v
            sto_emb = self.drop(sto_emb)
            logit = self.final(sto_emb)
        else:
            drop_emb = self.drop(emb)
            logit = self.final(drop_emb)

        return (logit, emb, log_var, density_x) if not self.backbone_model else \
            (local_feats, torch.cat(local_feats, dim=1), logit, emb, log_var, density_x)

    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in chain(self.emd.children(), self.var.children(), self.predict_density.children(),
                       self.latent.children(), self.final.children()):
            if isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# if __name__ == '__main__':
#     import argparse
#     import sys
#     sys.path.append('./Mammo-AGE/mammo_age')
#     parser = argparse.ArgumentParser(description='Mammo_AGE model')
#     parser.add_argument('--arch', default='resnet18', type=str, help='Model architecture')
#     parser.add_argument('--num_output_neurons', default=100, type=int, help='Number of output neurons')
#     parser.add_argument('--second_stage', default=False, type=bool, help='Whether to use second stage')
#     parser.add_argument('--first_stage_model', default=None, type=str, help='First stage model')
#     args = parser.parse_args()
#
#     model = Mammo_AGE(arch=args.arch, num_output=args.num_output_neurons, second_stage=args.second_stage,
#                       first_stage_model=args.first_stage_model)
#     B = 3
#     C = 1
#     W = 512
#     H = 512
#
#     input_a = torch.rand(B, 4, C, H, W)
#     output = model(input_a, max_t=50, use_sto=False)
#
#     print()