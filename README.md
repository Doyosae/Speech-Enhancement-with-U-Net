# Speech Enhancement (Coming soon)
## Datasets  
- [noizeus](https://ecs.utdallas.edu/loizou/speech/noizeus/)  
- [MS-SNND](https://github.com/microsoft/MS-SNSD)
## Paper review  
- [Supervised Speech Separation Based on Deep Learning: An Overview](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/01.md)  
- [Scaling Speech Enhancement in Unseen Environments with Noise Embeddings](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/02.md)  
- Coming soon!
## 01, First Speech Enhancement  
### 세팅
- 데이터셋은 noizeus, 0dB, 5dB, 10dB, 15dB의 SNR로 구성
- 총 930개의 노이즈 음성 데이터 중에서 0dB ~ 10dB 720개는 훈련 데이터, 15dB 210개는 테스트 데이터로 사용
- 데이터 전처리는 Short-time Fourier transform (librosa의 메서드 이용)
- 모델 구조는 Autoencoder, 손실 함수는 MSE, 평가 메트릭은 RMSE
#### 깨끗한 음성
![clean](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Denoising_Prototype/images/clean.png)
### 가정
- 낮은 dB의 데이터로 학습하면 비교적 선명한 높은 dB의 새로운 데이터에 대해서도 잘 개선하지 않을까?
### 학습
- 전처리한 데이터의 출력 크기가 (257, 265)여서, 모델의 편의를 위해 (256, 256) 사이즈로 잘라내었다.
- batch size = 60, epochs = 300
- Adam (lr = 0.00015, beta_1 = 0.5)
#### 훈련 데이터
![0db](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Denoising_Prototype/images/0dB.png)
![5db](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Denoising_Prototype/images/5dB.png)
![10db](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Denoising_Prototype/images/10dB.png)
### 결과
#### 훈련 데이터
![train](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Denoising_Prototype/images/trainSound.png)
![test](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Denoising_Prototype/images/testSound.png)
#
평가 데이터보다 낮은 dB의 훈련 데이터 720개로 학습한 모델은, 평가 데이터에 대해 더 나은 enhancement를 보이지 못했다.
