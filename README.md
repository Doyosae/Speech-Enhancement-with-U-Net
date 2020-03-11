# Speech Enhancement (Coming soon)
## Requirement
- Python 3.7.4
- numpy 1.18.1
- librosa 0.7.2
- Tensorflow 2.1.0
## Datasets  
- [noizeus](https://ecs.utdallas.edu/loizou/speech/noizeus/)  
- [MS-SNND](https://github.com/microsoft/MS-SNSD)
## Paper review  
- [Supervised Speech Separation Based on Deep Learning: An Overview](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/01.md)  
- [Scaling Speech Enhancement in Unseen Environments with Noise Embeddings](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/02.md)  
- [End-to-End Model for Speech Enhancement by Consistent Spectrogram Masking](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/03.md)
- Coming Soon !
## 01 ~ 03 폴더 구조
```
01_Speech_Enhancement
├── result/
├── save/
├── clean/
|        sp01.wav
|        sp01.wav
|        sp01.wav
|        ...
|        ...
|        sp30.wav
├── test/
|        sp01_airport_sn15.wav
|        sp01_babble_sn15.wav
|        sp01_airport_sn15.wav
|        ...
|        ...
|        sp01_street_sn15.wav
├── train/
|        sp01_airport_sn0.wav
|        sp01_airport_sn5.wav
|        sp01_airport_sn10.wav
|        ...
|        ...
|        sp01_train_sn10.wav
├── dataload.py
|        class dataLoader
|                def __init__
|                def createLabel
|                def createTrain
|                def createTest
|                def callData
|        class fixedLength(dataLoader)
|                def slicingData
|                def callData
├── processing.py
|        class dataProcessing
|                def __init__
|                def stft 
|                def melspectrogram
|                def mfcc
├── train.py
|        class enhancement
|                def __init__
|                def get_data 
|                def model
|                def train
|                        def rmse
├── test.py
|        arc = enhancement()
|        model = arc.model()
|        model.load_weights("speech_enhancement.h5")
|        trainLabel, testLabel, trainSound, testSound = arc.get_data()
|        def test
```
## 01 Speech Enhancement  
### 세팅
- 데이터셋은 noizeus, 0dB, 5dB, 10dB, 15dB의 SNR로 구성
- 총 930개의 노이즈 음성 데이터 중에서 0dB ~ 10dB 720개는 훈련 데이터, 15dB 210개는 테스트 데이터로 사용
- 데이터 전처리는 Short-time Fourier transform (librosa의 메서드 이용)
- 모델 구조는 Autoencoder, 손실 함수는 MSE, 평가 메트릭은 RMSE
### 예상
- 낮은 dB의 데이터로 학습하면 비교적 선명한 높은 dB의 새로운 데이터에 대해서도 잘 개선하지 않을까?
### 학습
- 전처리한 데이터의 출력 크기가 (257, 265)여서, 모델의 편의를 위해 (256, 256) 사이즈로 잘라내었다.
- batch size = 60, epochs = 300
- Adam (lr = 0.00015, beta_1 = 0.5)
### Spectogram dof train datasets
![train](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Enhancement_Example/images/train.png)
### Model test
![test](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Enhancement_Example/images/result.png)
#
Autoencoder 모델에서 낮은 SNR로 학습한 모델은 높은 SNR에 대해서 일반화를 하지 못하였다.
#
#
## 02 Speech Enhancement
### 세팅
- 기본적으로 01번이랑 모델 구조는 동일
- 오버피팅을 막기 위해 Fully Connected Layer를 0.3 드롭아웃
- 모델 자체가 노이즈를 커버할 수 있는 Binary Mask를 학습, 그래서 결과물은 estimated ideal binary mask
- train data와 clean data로 계산한 SNR와 criteria로 label of ideal binary mask를 생성
- 그래서 입력한 데이터에 대해 모델이 산출한 esimated ideal binary mask와 대응하는 label of ideal binary mask를 비교
- 이 둘의 Cross Entropy를 손실함수로 사용
- 결과 비교는 테스트 샘플의 estimated IBM와 테스트 샘플에 대응하는 IBM을 hadamard product를 수행하여 비교
### 예상
- 모델이 소음이 낀 음성을 입력으로 받으면, 최적의 바이너리 마스크를 추정할 수 있을까?
### Ideal Binary Mask
![M1](https://github.com/Doyosae/Speech_Enhancement/blob/master/02_Enhancement_Example/images/IBMtrain.png)
![M2](https://github.com/Doyosae/Speech_Enhancement/blob/master/02_Enhancement_Example/images/IBMTest.png)
- 훈련에서는 train IBM을 target으로 사용하고, 학습하는 동안 모델의 validation 체크로 test IBM을 target으로 사용
### 훈련 데이터
![train](https://github.com/Doyosae/Speech_Enhancement/blob/master/02_Enhancement_Example/images/train.png)
### 모델의 훈련 결과
![test](https://github.com/Doyosae/Speech_Enhancement/blob/master/02_Enhancement_Example/images/result.png)
