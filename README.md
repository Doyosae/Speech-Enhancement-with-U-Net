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
## 01 Speech Enhancement  
### 폴더 구조
```
01_Speech_Enhancement
├── result/
├── save/
├── clean/
        sp01.wav
        sp01.wav
        sp01.wav
        ...
        ...
        sp30.wav
├── test/
        sp01_airport_sn15.wav
        sp01_babble_sn15.wav
        sp01_airport_sn15.wav
        ...
        ...
        sp01_street_sn15.wav
├── train/
        sp01_airport_sn0.wav
        sp01_airport_sn5.wav
        sp01_airport_sn10.wav
        ...
        ...
        sp01_train_sn10.wav
├── dataload.py
        class dataLoader
                def __init__
                def createLabel
                def createTrain
                def createTest
                def callData
        class fixedLength(dataLoader)
                def slicingData
                def callData
├── processing.py
        class dataProcessing
                def __init__
                def stft 
                def melspectrogram
                def mfcc
├── train.py
        class enhancement
                def __init__
                def get_data 
                def model
                def train
                        def rmse
├── test.py
        arc = enhancement()
        model = arc.model()
        model.load_weights("speech_enhancement.h5")
        trainLabel, testLabel, trainSound, testSound = arc.get_data()
        def test
```
### 세팅
- 데이터셋은 noizeus, 0dB, 5dB, 10dB, 15dB의 SNR로 구성
- 총 930개의 노이즈 음성 데이터 중에서 0dB ~ 10dB 720개는 훈련 데이터, 15dB 210개는 테스트 데이터로 사용
- 데이터 전처리는 Short-time Fourier transform (librosa의 메서드 이용)
- 모델 구조는 Autoencoder, 손실 함수는 MSE, 평가 메트릭은 RMSE
### 가정
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
