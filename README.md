# Anomaly detection using Score based generative model

Authors: [Jinhwan Suk](https://github.com/vrvrv), [Jihyeong Jung](https://github.com/JhngJng)

[Video](https://youtu.be/AxmfQoNdIso), [Slides](https://www.overleaf.com/read/zjwnhhmwnqdt)

## Get Started

```bash
git clone https://github.com/vrvrv/Anomaly-detection-using-SBM.git
cd Anomaly-detection-using-SBM

pip install -r requirements.txt
```

### Download `MvTec` data

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection.
It contains over 5000 high-resolution images divided into fifteen different object and texture categories.

```bash
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar Jxvf mvtec_anomaly_detection.tar.xz
```

## Train
You can find configuration files at [configs/experiment/](configs/experiment).
Also, we provide pretrained weights from [here](https://www.dropbox.com/sh/dut2fypgx3igpq2/AABY6y66eVZTIb4XbekjVV1Ja?dl=0).
Please save the checkpoints at [checkpoints/](checkpoints/) directory.

In default
Before running code, sign up [wandb](https://wandb.ai/) which is experiment tracking platform.

### Training score SDE
```bash
python train.py experiment=capsule_64
```
The above code starts training from the prescribed checkpoints. If you want to train the model from scratch,
comment out the line `resume_from_checkpoint: ...` in `YAML` configuration file that you selected.

## Test
```bash
python test.py experiment=capsule_64
```

This computes the likelihood and *within-image* conditional likelihood of test dataset.


### References
Our codes are based on the following references.

- [Score based generative modeling through SDE, official code](https://github.com/yang-song/score_sde_pytorch)
- [Pytorch implementation of DDPM](https://github.com/w86763777/pytorch-ddpm.git)