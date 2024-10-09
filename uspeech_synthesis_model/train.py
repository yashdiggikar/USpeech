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
from torch.cuda.amp import autocast, GradScaler
from loguru import logger

from modules.ultrasoundAudioModel import ultrasoundAudioModel
from train_dataset import collectedDataset
from train_loss import SpectrumLoss
from scheduler import WarmupCosineScheduler

with open('/path/to/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

writer = SummaryWriter(log_dir=config['ua_tb_dir'])
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

def visualize_result(ultrasound_syn, ultrasound_phy, audio, writer, step):
    idx = np.random.randint(0, ultrasound_syn.size(0))
    ultrasound_syn_np = ultrasound_syn[idx].squeeze().detach().cpu().numpy()
    ultrasound_phy_np = ultrasound_phy[idx].squeeze().detach().cpu().numpy()
    audio_np = audio[idx].squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_pred = librosa.display.specshow(ultrasound_syn_np.T, ax=axes[0], x_axis='time', y_axis='linear')
    axes[0].set_title('Predicted Ultrasound')
    fig.colorbar(img_pred, ax=axes[0], format='%+2.0f dB')

    img_gt = librosa.display.specshow(ultrasound_phy_np.T, ax=axes[1], x_axis='time', y_axis='linear')
    axes[1].set_title('Ground Truth Ultrasound')
    fig.colorbar(img_gt, ax=axes[1], format='%+2.0f dB')

    img_audio = librosa.display.specshow(audio_np.T, ax=axes[2], x_axis='time', y_axis='linear')
    axes[2].set_title('Audio Spectrogram')
    fig.colorbar(img_audio, ax=axes[2], format='%+2.0f dB')

    writer.add_figure('Visualization', fig, global_step=step)
    plt.close(fig)



def train():
    logger.info(f"Using GPUs Index: {config['gpus']}")
    best_loss = torch.tensor(float('inf')).cuda()
    ua_model = ultrasoundAudioModel(config).cuda()

    spectrum_loss_instance = SpectrumLoss().cuda()

    if config['ua_previous_ckpt'] is not None:
        logger.info(f"Loading previous checkpoint: {config['ua_previous_ckpt']}")
        weight = torch.load(config['ua_previous_ckpt'], map_location=torch.device('cpu'))
        ua_model.load_state_dict(weight['model'])
    
    if config['num_gpus'] > 1:
        logger.info("Parallelizing model...")
        ua_model = nn.DataParallel(ua_model)
    
    lr = config['learning_rate']
    optimizer = getattr(optim, config['optimizer'])(ua_model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=config['warmup_steps'], total_steps=config['total_steps'], min_lr=config['min_lr'])
    scalar = GradScaler()
    start_time = time.time()

    current_step = 0
    dataset_train = collectedDataset(phases=['train'], config=config, p=0.75)
    dataset_test = collectedDataset(phases=['test'], config=config)
    
    while current_step < config['total_steps']:
        for phase in ['train', 'test']:
            if phase == 'train':
                dataset = dataset_train
            else:
                dataset = dataset_test

            logger.info(f'Start {phase} phase, dataset size: {len(dataset)}')
            loader = create_dataloader(dataset, config['batch_size'], config['num_workers'], shuffle=(phase == 'train'))

            steps_per_epoch = len(dataset) // config['batch_size']
            logger.info(f'Steps per epoch: {steps_per_epoch}')

            if phase == 'train':
                ua_model.train()
            else:
                ua_model.eval()
                accumulated_test_loss = 0.0
                num_test_steps = 0
            
            for i_iter, input_data in enumerate(loader):
                if current_step >= config['total_steps']:
                    break

                audio = input_data.get('batch_audio').cuda(non_blocking=True)
                ultrasound_phy = input_data.get('batch_ultrasound').cuda(non_blocking=True)
                
                if phase == 'train':
                    with autocast():
                        result = ua_model(audio)
                else:
                    with torch.no_grad():
                        result = ua_model(audio)

                ultrasound_syn = result['ultrasound_syn']
                ultrasound_phy = ultrasound_phy.to(ultrasound_syn.dtype)

                generative_loss = spectrum_loss_instance(ultrasound_syn, ultrasound_phy)
                mse_loss = generative_loss['mse']
                temporal_loss = generative_loss['temporal']
                total_loss = generative_loss['total_loss']  

                if phase == 'train':
                    optimizer.zero_grad()
                    scalar.scale(total_loss).backward()
                    scalar.step(optimizer)
                    scalar.update()
                    end_time = time.time()
                    eta = (end_time - start_time) * (len(loader) - i_iter)
                    start_time = time.time()

                    logger.info(f"Train step={current_step}, ETA={eta:.1f}s, Train loss={total_loss:.5f}, \
                                Train mse loss={mse_loss:.5f}, Train temporal loss={temporal_loss:.5f}, \
                                LR={show_learning_rate(optimizer)}, Best test loss={best_loss:.2f}")
                    
                    current_step += 1
                    if i_iter % config['visualization_strides'] == 0:
                        visualize_result(ultrasound_syn, ultrasound_phy, audio, writer, current_step)

                    writer.add_scalar('Loss/train', total_loss.item(), current_step)
                    writer.add_scalar('Loss/mse_loss', mse_loss.item(), current_step)
                    writer.add_scalar('Loss/temporal_loss', temporal_loss.item(), current_step)
                    
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

                if not os.path.exists(config['ua_saved_prefix']):
                    os.makedirs(config['ua_saved_prefix'])
                
                savename = os.path.join(config['ua_saved_prefix'], 'last.pt')
                logger.info(f"Saving model at {savename}")

                if config['num_gpus'] > 1:
                    torch.save({'ua_model': ua_model.module.state_dict()}, savename)
                else:
                    torch.save({'ua_model': ua_model.state_dict()}, savename)
                
                if average_test_loss < best_loss or best_loss == torch.tensor(float('inf')):
                    best_loss = average_test_loss
                    shutil.copy(savename, savename.replace('last.pt', 'best.pt'))
                    logger.info(f'Best testation loss updated to {best_loss:.5f}')
                
if __name__ == "__main__":
    train()
