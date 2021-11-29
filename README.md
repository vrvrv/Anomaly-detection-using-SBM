# DGM_BO_Final

## Get Started
### Download `MvTec` data
```bash
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1
tar Jxvf mvtec_anomaly_detection.tar.xz
```

## Train
You can find configuration files at [here](configs/experiment).

### Score SDE
```bash
python train.py experiment=mvtec_capsule_ddpm_subvp
```

## Test
```bash
python test.py experiment=mvtec_capsule_ddpm_subvp trainer.gpus=1
```
