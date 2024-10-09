import warnings
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.finished_warmup = False
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr, last_epoch=-1)
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            self.cosine_scheduler.last_epoch = self.last_epoch - self.warmup_steps
            self.finished_warmup = True
            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is not None:
            warnings.warn("This scheduler is step-based, the `epoch` argument is ignored.", UserWarning)
        self.last_epoch += 1
        if self.finished_warmup:
            self.cosine_scheduler.step()
        else:
            super(WarmupCosineScheduler, self).step()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
