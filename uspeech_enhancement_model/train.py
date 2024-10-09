import os
import time
import shutil
import yaml
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from dataset import unetDataset
from model import unet_model
from modules.bottleneck_transformer import *
from loss import UnetLoss

with open('/path/to/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

writer = SummaryWriter(log_dir=config['en_tb_dir'])
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpus']

def show_learning_rate(optimizer):
    lr = [f'{param_group["lr"]:.6f}' for param_group in optimizer.param_groups]
    return ','.join(lr)

def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    missed_params = [k for k in model_dict if k not in pretrained_dict]

    logger.info(f'Loaded params/total params: {len(pretrained_dict)}/{len(model_dict)}')
    logger.info(f'Missed params: {missed_params}')
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model

def create_dataloader(dataset, batch_size, num_workers, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True
    )

def visulize_result(clean_speech, noisy_speech, enhanced_speech, ultrasound_syn, ultrasound_phy, writer, step):

    idx = np.random.randint(0, ultrasound_phy.shape[0])
    ultrasound_syn_np = ultrasound_syn[idx].squeeze().detach().cpu().numpy()
    ultrasound_phy_np = ultrasound_phy[idx].squeeze().detach().cpu().numpy()
    clean_speech_np = clean_speech[idx].squeeze().detach().cpu().numpy()
    noisy_speech_np = noisy_speech[idx].squeeze().detach().cpu().numpy()
    enhanced_speech_np = enhanced_speech[idx].squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(5, 1, figsize=(10, 10))

    img_ultrasound_syn = librosa.display.specshow(ultrasound_syn_np.T, ax=axes[0], x_axis='time', y_axis='linear')
    axes[0].set_title('ultrasound_syn')
    fig.colorbar(img_ultrasound_syn, ax=axes[0], format="%+2.0f dB")

    img_ultrasound_phy_np = librosa.display.specshow(ultrasound_phy_np.T, ax=axes[1], x_axis='time', y_axis='linear')
    axes[1].set_title('ultrasound_phy')
    fig.colorbar(img_ultrasound_phy_np, ax=axes[1], format="%+2.0f dB")

    img_clean_speech = librosa.display.specshow(clean_speech_np.T, ax=axes[2], x_axis='time', y_axis='linear')
    axes[2].set_title('Clean Speech')
    fig.colorbar(img_clean_speech, ax=axes[2], format="%+2.0f dB")
    
    img_noisy_speech = librosa.display.specshow(noisy_speech_np.T, ax=axes[3], x_axis='time', y_axis='linear')
    axes[3].set_title('Noisy Speech')
    fig.colorbar(img_noisy_speech, ax=axes[3], format="%+2.0f dB")
    
    img_enhanced_speech = librosa.display.specshow(enhanced_speech_np.T, ax=axes[4], x_axis='time', y_axis='linear')
    axes[4].set_title('Enhanced Speech')
    fig.colorbar(img_enhanced_speech, ax=axes[4], format="%+2.0f dB")

    writer.add_figure('Visualization', fig, global_step=step)

    plt.tight_layout()
    plt.close(fig)


def train():
    logger.info(f"Using GPUs Index: {config['en_gpus']}")
    best_loss = torch.tensor(float('inf')).cuda()

    en_model = unet_model(config).cuda()
    load_pretrained_transformer(config, en_model.speech_unet.transformer)

    unet_loss_instance = UnetLoss().cuda()

    lr = config['en_learning_rate']

    optimizer = getattr(optim, config['en_optimizer'])(en_model.parameters(), lr=lr, weight_decay=config['en_weight_decay'])
    scheduler = StepLR(optimizer, step_size=config['en_step_size'], gamma=config['en_gamma'])

    if config['en_num_gpus'] > 1:
        logger.info("Parallelizing model...")
        en_model = nn.DataParallel(en_model)

    scalar = torch.cuda.amp.GradScaler()
    start_time = time.time()

    current_step = 0
    dataset_train = unetDataset(phases=['train'], config=config, p=1)
    dataset_val = unetDataset(phases=['test'], config=config)

    while current_step < config['en_total_steps']:
        for phase in ['train', 'test']:
            if phase == 'train':
                dataset = dataset_train
            else:
                dataset = dataset_val

            logger.info(f'Start {phase} phase, dataset size: {len(dataset)}')
            loader = create_dataloader(dataset, config['en_batch_size'], config['en_num_workers'], shuffle=(phase == 'train'))

            steps_per_epoch = len(dataset) // config['en_batch_size']
            logger.info(f'Steps per epoch: {steps_per_epoch}')

            if phase == 'train':
                en_model.train()
            else:
                en_model.eval()
                accumulated_test_loss = 0.0
                num_test_steps = 0

            for i_iter, input_data in enumerate(loader):
                if current_step >= config['en_total_steps']:
                    break

                clean_speech = input_data.get('batch_clean_speech').cuda(non_blocking=True)
                noisy_speech = input_data.get('batch_noisy_speech').cuda(non_blocking=True)
                ultrasound_syn = input_data.get('batch_ultrasound_syn').cuda(non_blocking=True)
                ultrasound_phy = input_data.get('batch_ultrasound_phy').cuda(non_blocking=True)

                if phase == 'train':
                    with torch.cuda.amp.autocast():
                        result = en_model(noisy_speech, ultrasound_syn)
                        enhanced_speech = result['enhancement_speech']
                        unet_loss = unet_loss_instance(enhanced_speech, clean_speech)
                        total_loss = unet_loss['loss']
                else:
                    with torch.no_grad():
                        result = en_model(noisy_speech, ultrasound_syn)
                        enhanced_speech = result['enhancement_speech']
                        unet_loss = unet_loss_instance(enhanced_speech, clean_speech)
                        total_loss = unet_loss['loss']

                if phase == 'train':
                    optimizer.zero_grad()
                    scalar.scale(total_loss).backward()
                    
                    scalar.step(optimizer)
                    scalar.update()

                    end_time = time.time()
                    eta = (end_time - start_time) * (len(loader) - i_iter)
                    start_time = time.time()

                    logger.info(f"Train step={current_step}, ETA={eta:.1f}s, Train loss={total_loss:.5f}, \
                                LR={show_learning_rate(optimizer)}, Best test loss={best_loss:.2f}")
                    current_step += 1

                    if i_iter % config['en_visualization_strides'] == 0:
                        visulize_result(clean_speech, noisy_speech, enhanced_speech, ultrasound_syn, ultrasound_phy, writer, current_step)
                    
                    writer.add_scalar('Loss/train', total_loss.item(), current_step)
                   
                    
                    current_lr = optimizer.param_groups[0]['lr']

                    scheduler.step()
                    writer.add_scalar('Learning_rate', current_lr, current_step)
                    if current_step % steps_per_epoch == 0:
                        break
                else:
                    test_loss = total_loss.item()
                    accumulated_test_loss += test_loss
                    num_test_steps += 1
            if phase =='test':
                average_test_loss = accumulated_test_loss / num_test_steps
                writer.add_scalar('Loss/test', average_test_loss, current_step)

                if not os.path.exists(config['en_saved_prefix']):
                    os.makedirs(config['en_saved_prefix'])
                
                savename = os.path.join(config['en_saved_prefix'], 'last.pt')
                logger.info(f"Saving model at {savename}")

                if config['num_gpus'] > 1:
                    torch.save({'en_model': en_model.module.state_dict()}, savename)
                else:
                    torch.save({'en_model': en_model.state_dict()}, savename)
                
                if average_test_loss < best_loss or best_loss == torch.tensor(float('inf')):
                    best_loss = average_test_loss
                    shutil.copy(savename, savename.replace('last.pt', 'best.pt'))
                    logger.info(f'Best test loss updated to {best_loss:.5f}')

if __name__ == "__main__":
    train()
