# adjust from Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models
# abs: https://arxiv.org/pdf/2306.17203v1
# code: https://github.com/luosiallen/Diff-Foley

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributed as dist

try:
    import torch.distributed.nn
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a compatible PyTorch version.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)

            if not local_loss:
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            
            if not local_loss:
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss_Temporal_Semantic(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
            temporal_mix_weight=0.5,
            init_logit_scale=1 / 0.07
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temporal_mix_weight = temporal_mix_weight
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_logit_scale)))

        self.prev_num_logits_semantic = 0
        self.prev_num_logits_temporal = 0
        self.labels_semantic = {}
        self.labels_temporal = {}

    def forward(self, video_temporal_features, video_mean_features, spec_temporal_features, spec_mean_features, output_dict=False):

        device = video_mean_features.device

        if self.world_size > 1:
            all_video_mean_features, all_spec_mean_features = gather_features(
                video_mean_features, spec_mean_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )
            if self.local_loss:
                logits_per_video_semantic = self.logit_scale * video_mean_features @ all_spec_mean_features.T
                logits_per_spec_semantic = self.logit_scale * spec_mean_features @ all_video_mean_features.T
            else:
                logits_per_video_semantic = self.logit_scale * all_video_mean_features @ all_spec_mean_features.T
                logits_per_spec_semantic = logits_per_video_semantic.T
        else:
            logits_per_video_semantic = self.logit_scale * video_mean_features @ spec_mean_features.T
            logits_per_spec_semantic = self.logit_scale * spec_mean_features @ video_mean_features.T

        # Temporal Contrastive Loss
        if self.world_size > 1:
            all_video_temporal_features, all_spec_temporal_features = gather_features(
                video_temporal_features, spec_temporal_features, 
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )

            if self.local_loss:
                logits_per_video_temporal = self.logit_scale * video_temporal_features @ all_spec_temporal_features.permute(0, 2, 1)
                logits_per_spec_temporal = self.logit_scale * spec_temporal_features @ all_video_temporal_features.permute(0, 2, 1)
            else:
                logits_per_video_temporal = self.logit_scale * all_video_temporal_features @ all_spec_temporal_features.permute(0, 2, 1)
                logits_per_spec_temporal = logits_per_video_temporal.permute(0, 2, 1)
        else:
            logits_per_video_temporal = self.logit_scale * video_temporal_features @ spec_temporal_features.permute(0, 2, 1)
            logits_per_spec_temporal = self.logit_scale * spec_temporal_features @ video_temporal_features.permute(0, 2, 1)

        # Semantic Contrast Loss Calculation
        num_logits_semantic = logits_per_video_semantic.shape[0]
        if self.prev_num_logits_semantic != num_logits_semantic or device not in self.labels_semantic:
            labels_semantic = torch.arange(num_logits_semantic, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels_semantic = labels_semantic + num_logits_semantic * self.rank
            if self.cache_labels:
                self.labels_semantic[device] = labels_semantic
                self.prev_num_logits_semantic = num_logits_semantic
        else:
            labels_semantic = self.labels_semantic[device]

        semantic_contrast_loss = (F.cross_entropy(logits_per_video_semantic, labels_semantic) + 
                                  F.cross_entropy(logits_per_spec_semantic, labels_semantic)) / 2

        # Temporal Contrast Loss Calculation
        bs, num_logits_temporal , _ = logits_per_video_temporal.shape
        if self.prev_num_logits_temporal != num_logits_temporal or device not in self.labels_temporal:
            labels_temporal = torch.arange(num_logits_temporal, device=device, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
            if self.cache_labels:
                self.labels_temporal[device] = labels_temporal
                self.prev_num_logits_temporal = num_logits_temporal
        else:
            labels_temporal = self.labels_temporal[device]

        logits_per_video_temporal = logits_per_video_temporal.reshape(bs * num_logits_temporal, num_logits_temporal)
        logits_per_spec_temporal = logits_per_spec_temporal.reshape(bs * num_logits_temporal, num_logits_temporal)
        labels_temporal = labels_temporal.reshape(bs * num_logits_temporal)

        temporal_contrast_loss = (F.cross_entropy(logits_per_video_temporal, labels_temporal) + 
                                  F.cross_entropy(logits_per_spec_temporal, labels_temporal)) / 2

        total_loss = self.temporal_mix_weight * temporal_contrast_loss + semantic_contrast_loss

        return {
                "semantic_contrast_loss": semantic_contrast_loss, 
                "temporal_contrast_loss": temporal_contrast_loss, 
                "total_loss": total_loss,
                "temp_mix_weight": torch.tensor(self.temporal_mix_weight), 
                "logit_scale": self.logit_scale
                    }
