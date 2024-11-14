# USpeech
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Official implementation of USpeech: Ultrasound-Enhanced Speech with Minimal Human Effort via Cross-Modal Synthesis

## Abstract
> Speech enhancement is pivotal in human-computer interaction, especially in ubiquitous devices. Ultrasound-based speech enhancement has emerged as an attractive choice without the need for extra hardware. Existing solutions, however, rely on labor-intensive and time-consuming data collection under various settings, limiting the full potential of ultrasound-based speech enhancement. To address the challenge, we propose USpeech, a cross-modal ultrasound synthesis system for speech enhancement with minimal human effort. At the core of USpeech is a two-stage framework that establishes a correspondence between visual and ultrasonic modalities by leveraging the audible audio as the bridge, overcoming challenges caused by the lack of paired video-ultrasound datasets and the inherent heterogeneity between video and ultrasound. Our framework incorporates contrastive video-audio pre-training to project multiple modalities into a shared semantic space and employs an audio-ultrasound encoder-decoder for ultrasound synthesis. Based on this, we present a speech enhancement network to enhance the speech in the time-frequency domain and further recover the clean speech waveform via the neural vocoder. Comprehensive experiments show that USpeech demonstrates remarkable performance using synthetic ultrasound data, comparable to that using physical data, while significantly outperforming state-of-the-art baselines.
##
<p align="center"> <img src='fig/overall.png' align="center"> </p>

> **USpeech: Ultrasound-Enhanced Speech with Minimal Human Effort via Cross-Modal Synthesis**               

---
## Setup Instructions
To get started with USpeech, follow these steps:
1. Clone the GitHub repository:
``` bash
$ git clone https://github.com/aiot-lab/USpeech.git
$ cd USpeech
$ pip install -r requirements.txt

# Install ParallelWaveGAN under the USpeech folder (https://github.com/kan-bayashi/ParallelWaveGAN)
$ git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
$ cd ParallelWaveGAN
$ pip install -e .
```
2. Configure dataset paths, preprocessing parameters, and training settings in the ```config.yaml``` file.

3. Run the code in synthesis and enhancement models.

---

## Required Files
- Datasets
    - [LRW Dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
    - [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
    - [TIMIT Dataset](https://catalog.ldc.upenn.edu/LDC93S1)
    - [VCTK Dataset](https://datashare.ed.ac.uk/handle/10283/3443)
    - [Nonspeech7k Dataset](https://zenodo.org/records/6967442)

- Pretrained Models
    - [Slowonly Model](https://github.com/open-mmlab/mmaction2/tree/main/configs/recognition/slowonly)
    - [PANN Model](https://github.com/qiuqiangkong/audioset_tagging_cnn)

---

## Repository Structure
```
.
├── config.yaml
├── evaluation.py
├── preprocessing
│   ├── LRW_dataset_preprocessing.py
│   ├── collected_dataset_preprocessing.py
│   └── noisy_collected_dataset_preprocessing.py
├── synthesis.py
├── uspeech_enhancement_model
│   ├── __init__.py
│   ├── dataset.py
│   ├── loss.py
│   ├── model.py
│   ├── modules
│   │   ├── UNet.py
│   │   ├── UNet_parts.py
│   │   ├── __init__.py
│   │   └── bottleneck_transformer.py
│   └── train.py
└── uspeech_synthesis_model
    ├── __init__.py
    ├── modules
    │   ├── __init__.py
    │   ├── pann.py
    │   ├── slowonly.py
    │   ├── ultrasoundAudioModel.py
    │   ├── ultrasoundDecoder.py
    │   └── videoAudioModel.py
    ├── pretrain.py
    ├── pretrain_dataset.py
    ├── pretrain_loss.py
    ├── scheduler.py
    ├── train.py
    ├── train_dataset.py
    └── train_loss.py
```

---

## Citation
```
@article{yu2024uspeech,
  title={USpeech: Ultrasound-Enhanced Speech with Minimal Human Effort via Cross-Modal Synthesis},
  author={Yu, Luca Jiang-Tao and Zhao, Running and Ji, Sijie and Ngai, Edith CH and Wu, Chenshu},
  journal={arXiv preprint arXiv:2410.22076},
  year={2024}
}
```

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.



