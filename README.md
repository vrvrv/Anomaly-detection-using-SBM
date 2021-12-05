# Anomaly detection using Score based generative model

[Presentation slides](https://www.overleaf.com/read/zjwnhhmwnqdt)

## Get Started

```bash
git clone https://github.com/vrvrv/Anomaly-detection-using-SBM.git
cd Anomaly-detection-using-SBM

pip install -r requirements.txt
```

### Download `MvTec` data
```bash
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1
tar Jxvf mvtec_anomaly_detection.tar.xz
```

## Train
You can find configuration files at [here](configs/experiment).
Also, we provide pretrained weights from [here](https://www.dropbox.com/sh/dut2fypgx3igpq2/AABY6y66eVZTIb4XbekjVV1Ja?dl=0).
Please save the checkpoints at [checkpoints](checkpoints/) directory.

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