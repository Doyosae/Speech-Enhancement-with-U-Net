# Speech Enhancement (진행 중)
## 0. Introduction  
Deep learning models for Speech Enhancement were made using U Net.  
Assume that the model will deduce a good mask.  
Speech Enhancement performed STFT and trained on multiple scales.  
Ratio mask makes proportion mask and targets model.  
Speech Enhancement is performed in and out of SNR -5 dB. 
## 1. Paper review  
- [Supervised Speech Separation Based on Deep Learning: An Overview](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/01.md)  
- [Scaling Speech Enhancement in Unseen Environments with Noise Embeddings](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/02.md)  
- [End-to-End Model for Speech Enhancement by Consistent Spectrogram Masking](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/03.md)
- [Towards Generalized Speech Enhancement with Generative Adversarial Networks](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/04.md)
- [Deep Speech Enhancement for Reverberated and Noisy Signals using Wide Residual Networks](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/05.md)
- Coooooming Soooon   
## 2. Requirement
- Python 3.7.4
- numpy 1.18.1
- librosa 0.7.2
- Tensorflow 2.1.0
## 3. Datasets  
- [MS-SNND](https://github.com/microsoft/MS-SNSD)
- Data from lab (data is private)
## 4. Folder
```
├─Datasets
│  │  README.md
│  ├─MS_SNDN
│  │      stft_test_clean.npy
│  │      stft_test_noisy.npy
│  │      stft_train_clean.npy
│  │      stft_train_noisy.npy
│  │      test_clean.npy
│  │      test_noisy.npy
│  │      train_clean.npy
│  │      train_noisy.npy
```
## 5. Sample
### Noisy wave, Clean wave
---
![Test Wave](https://github.com/Doyosae/Speech_Enhancement/blob/master/sample/test_wave.png)
  
### Noisy spectogram, Ratio mask, Clean spectogram
---
![Test Spectogram and ratio mask](https://github.com/Doyosae/Speech_Enhancement/blob/master/sample/test_spectogram.png)
  
### Noisy spectogram, Ratio mask, Clean spectogram
---
![Test Spectogram and ratio mask](https://github.com/Doyosae/Speech_Enhancement/blob/master/sample/test_spectogram.png)
  
### Noisy spectogram, Predict ratio mask, Enhancement spectogram
---
![Test Spectogram and ratio mask](https://github.com/Doyosae/Speech_Enhancement/blob/master/sample/enhancement_specogram.png)
  
