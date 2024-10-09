import os
import time
import shutil
import io
import yaml
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import ToTensor

from modules.videoAudioModel import videoAudioModel, load_dict
from pretrain_dataset import videoAudioDataset
from pretrain_loss import ClipLoss_Temporal_Semantic
from scheduler import WarmupCosineScheduler

with open('/path/to/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

torch.backends.cudnn.benchmark = True
writer = SummaryWriter(log_dir=config['av_tb_dir'])
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

def log_embeddings_and_logits(writer, embeddings_v, embeddings_a, logits, label, global_step):
    
    def process_embeddings(embeddings):
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        embeddings_reshaped = embeddings_norm.view(-1, 1, 16, 32)
        return embeddings_reshaped

    embeddings_v_processed = process_embeddings(embeddings_v).detach().cpu()
    embeddings_a_processed = process_embeddings(embeddings_a).detach().cpu()

    logits_normalized = (logits - logits.min()) / (logits.max() - logits.min())
    logits_normalized = logits_normalized.detach().cpu()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(embeddings_v_processed[0].squeeze(), cmap='hot')
    axs[0].set_title('Embeddings V')
    axs[1].imshow(embeddings_a_processed[0].squeeze(), cmap='hot')
    axs[1].set_title('Embeddings A')
    cax = axs[2].matshow(logits_normalized.numpy(), cmap='viridis')
    fig.colorbar(cax, ax=axs[2])
    axs[2].set_title('Logits')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = ToTensor()(image)
    writer.add_image(f'{label}', image_tensor, global_step)
    buf.close()
    plt.close(fig)

def pretrain():
    logger.info(f"Using GPUs Index: {config['gpus']}")
    best_loss = torch.tensor(float('inf')).cuda()

    videoaudio_model = videoAudioModel(config).cuda()
    contrastive_loss_instance = ClipLoss_Temporal_Semantic().cuda()

    if config['av_previous_ckpt'] is not None:
        logger.info(f"Loading weights from {config['av_previous_ckpt']}")
        weight = torch.load(config['av_previous_ckpt'], map_location=torch.device('cpu'))
        videoaudio_model = load_missing(videoaudio_model, weight.get('videoaudio_model'))

    if config['pretrained']:
        videoaudio_model = load_dict(videoaudio_model, config, check=True)
        logger.info('Loaded pretrained weights')

    if config['num_gpus'] > 1:
        logger.info("Parallelizing model...")
        videoaudio_model = nn.DataParallel(videoaudio_model)

    lr = config['learning_rate']
    optimizer = getattr(optim, config['optimizer'])(videoaudio_model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=config['warmup_steps'], total_steps=config['total_steps'], min_lr=config['min_lr'])
    scaler = GradScaler()
    start_time = time.time()

    current_step = 0

    while current_step < config['total_steps']:

        for phase in ['train', 'test']:
            dataset = videoAudioDataset([phase], config)

            logger.info(f'Start {phase} phase, dataset size: {len(dataset)}')
            loader = create_dataloader(dataset, config['batch_size'], config['num_workers'], shuffle=(phase == 'train'))

            steps_per_epoch = len(dataset) // config['batch_size']
            logger.info(f'Steps per epoch: {steps_per_epoch}')

            if phase == 'train':
                videoaudio_model.train()
            else:
                videoaudio_model.eval()
                accumulated_test_loss = 0.0
                num_test_steps = 0

            for i_iter, input_data in enumerate(loader):
                if current_step >= config['total_steps']:
                    break

                videos = input_data.get('batch_video').cuda(non_blocking=True)
                audios = input_data.get('batch_audio').cuda(non_blocking=True)

                if phase == 'train':
                    with autocast():
                        result = videoaudio_model(videos, audios)
                else:
                    with torch.no_grad():
                        result = videoaudio_model(videos, audios)

                video_temp_emb = result['video_temp_emb']
                audio_temp_emb = result['audio_temp_emb']
                video_emb = result['video_emb']
                audio_emb = result['audio_emb']

                contrastive_loss = contrastive_loss_instance(video_temp_emb, video_emb, audio_temp_emb, audio_emb, config['logit_scale'])

                temporal_loss = contrastive_loss['temporal_contrast_loss']
                semantic_loss = contrastive_loss['semantic_contrast_loss']
                loss = contrastive_loss['total_loss']
                logit_scale = contrastive_loss['logit_scale']

                logits = (F.normalize(video_emb, p=2, dim=-1) @ F.normalize(audio_emb, p=2, dim=-1).T) * logit_scale.exp().item()

                if phase == 'train':
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    end_time = time.time()
                    eta = (end_time - start_time) * (len(loader) - i_iter)
                    start_time = time.time()
                    logger.info(f"Train step={current_step}, ETA={eta:.1f}s, Train loss={loss:.5f}, \
                                Train temporal loss={temporal_loss:.5f}, Train semantic loss={semantic_loss:.5f}, \
                                Logit scale={logit_scale.exp().item():.5f}, LR={show_learning_rate(optimizer)}, Best test loss={best_loss:.2f}")

                    current_step += 1
                    if i_iter % config['visualization_strides'] == 0:
                        log_embeddings_and_logits(writer, video_emb, audio_emb, logits, 'Embeddings_Logits', current_step)

                    writer.add_scalar('Loss/train', loss.item(), current_step)
                    writer.add_scalar('Loss/temporal_loss', temporal_loss.item(), current_step)
                    writer.add_scalar('Loss/semantic_loss', semantic_loss.item(), current_step)
                    writer.add_scalar('Logit Scale', logit_scale.exp().item(), current_step)

                    current_lr = optimizer.param_groups[0]['lr']
                    scheduler.step()
                    writer.add_scalar('Learning Rate', current_lr, current_step)

                    if current_step % steps_per_epoch == 0:
                        break
                else:
                    test_loss = loss.item()
                    accumulated_test_loss += test_loss
                    num_test_steps += 1

            if phase == 'test':
                average_test_loss = accumulated_test_loss / num_test_steps
                writer.add_scalar('Loss/test', average_test_loss, current_step)

                if not os.path.exists(config['av_saved_prefix']):
                    os.makedirs(config['av_saved_prefix'])

                savename = os.path.join(config['av_saved_prefix'], 'last.pt')
                logger.info(f"Saving model at {savename}")

                if config['num_gpus'] > 1:
                    torch.save({'videoaudio_model': videoaudio_model.module.state_dict()}, savename)
                else:
                    torch.save({'videoaudio_model': videoaudio_model.state_dict()}, savename)

                if average_test_loss < best_loss or best_loss == torch.tensor(float('inf')):
                    best_loss = average_test_loss
                    shutil.copy(savename, savename.replace('last.pt', 'best.pt'))
                    logger.info(f'Best testidation loss updated to {best_loss:.5f}')

    writer.close()

if __name__ == '__main__':
    pretrain()
