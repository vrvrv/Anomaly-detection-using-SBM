# DGM_BO_Final


## Train

```python
data_name in ['MVTec_AD', 'KDD14']
```

### VAE
```bash
python train.py experiment=<data_name>_vae
```

### Score SDE
```bash
python train.py experiment=<data_name>_score_sde
```