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
- [Towards Generalized Speech Enhancement with Generative Adversarial Networks](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/04.md)
- [Deep Speech Enhancement for Reverberated and Noisy Signals using Wide Residual Networks](https://github.com/Doyosae/Speech_Enhancement/blob/master/paper/05.md)
- Coooooming Soooon
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
## 해야할 것
sampling_rate = 7000, n_fft = 256, hop_length = 128에서 모델의 기본 옵션  
train data 660, test data 60의 11:1 split  
autoencoder, U Net (only real value parts, binary mask, ratio mask, phase conserving)
## 목적
- 동일한 모델에서 깨끗한 음성의 스펙트럼을 표현하는 매핑 기반 모델과,  
  SNR 값을 기반으로 한 마스킹 기반 모델의 Speech Enhancement 결과를 비교
- 여러가지 마스크를 활용한 마스킹 기반의 모델의 Speech Enhancement 방법론 이해
- 01 ~ 03 모든 파일 공유 https://drive.google.com/file/d/11x9cVIdkQ4R5FGkjXvrvjb0yo5np-zDV/view?usp=sharing
## 01 Speech Enhancement  
### 세팅
- 데이터셋은 noizeus, 0dB, 5dB, 10dB, 15dB의 SNR로 구성
- 총 930개의 노이즈 음성 데이터 중에서 0dB, 5dB, 10dB 720개는 훈련 데이터, 15dB 210개는 테스트 데이터로 사용
- 데이터 전처리는 Short-Time Fourier transform을 적용하고 TF 도메인 기반 (librosa의 메서드 이용)
- 모델 구조는 Autoencoder, 손실 함수는 MSE, 평가 메트릭은 RMSE (평가 메트릭을 변경할 필요 있음)
### 학습
- 전처리한 데이터의 출력 크기가 (257, 265)여서, 모델의 편의를 위해 (256, 256) 사이즈로 잘라내었다.
- batch size = 60, epochs = 60
- Adam (lr = 0.00015, beta_1 = 0.5)
### Spectogram of train datasets
![train](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Enhancement_Example/images/train.png)
### 모델의 훈련 결과
![test](https://github.com/Doyosae/Speech_Enhancement/blob/master/01_Enhancement_Example/images/result_2.png)
#
#
## 02 ~ 03 Speech Enhancement
### 세팅
- 01번이랑 모델 구조는 동일, 오버피팅을 막기 위해 Fully Connected Layer를 0.3 드롭아웃
- 모델 자체가 binary mask를 산출하도록 학습, 그래서 결과물은 estimated ideal binary mask
- train data와 clean data로 계산한 SNR을 이용하여 train data의 train target ideal binary mask를 생성
- 입력한 데이터에 대하여 모델이 산출한 esimated ideal binary mask와 그에 대응하는 target ideal binary mask를 비교
- 이 둘의 손실함수는 MSE와 Cross Entropy에 대하여 모델 테스트 (논문에서는 주로 Cross Entropy를 사용한다고 언급)
- 결과 비교는 테스트 샘플의 estimated IBM와 테스트 샘플에 대응하는 label IBM을 hadamard product를 수행하여 비교
- 02는 Leaky ReLU와 MSE, 03은 ReLU와 MSE 그리고 ReLU와 Binary Cross Entropy 저장 파일이 있다.
### 예상
- 모델이 소음이 낀 음성을 입력으로 받으면, 최적의 바이너리 마스크를 추정할 수 있을까?
- 전처리한 데이터의 출력 크기가 (257, 265)여서, 모델의 편의를 위해 (256, 256) 사이즈로 잘라내었다.
- batch size = 60, epochs = 50, Adam (lr = 0.00015, beta_1 = 0.5)
### Ideal Binary Mask
![M1](https://github.com/Doyosae/Speech_Enhancement/blob/master/02_Enhancement_Example/images/IBMtrain.png)
![M2](https://github.com/Doyosae/Speech_Enhancement/blob/master/02_Enhancement_Example/images/IBMTest.png)
- 훈련에서는 train IBM을 target으로 사용하고, 학습하는 동안 모델의 validation 검증으로 test IBM을 target으로 사용
### Spectogram of train datasets
![train](https://github.com/Doyosae/Speech_Enhancement/blob/master/02_Enhancement_Example/images/train.png)
### 모델의 훈련 결과
![MSE](https://github.com/Doyosae/Speech_Enhancement/blob/master/02_Enhancement_Example/images/result.png)
![BCE](https://github.com/Doyosae/Speech_Enhancement/blob/master/03_Enhancement_Example/images/result.png)
### 결론
- 낮은 dB에서는 매핑 기반의 Autoencoder 모델이 마스킹 기반보다 더 나은 성능을 보임
- Cross Entropy가 더 나은 성능을 보일 것으로 예상했으나 MSE하고 사실상 차이를 보이지 않음
- 그 어느 모델에서든 음성이 갈라지는 결과를 보임 (무슨 이유인지 생각하고 앞으로 개선해나가야 함)
- 모델의 출력을 binary mask가 아니라, 먼저 speech에 binary mask를 씌워서 estimated speech의 손실값  
  자체를 계산하는 것도 시도를 하였으나, 이 테스트의 결과는 매우 나빠서 앞으로의 고려사항에 넣지 않았다.
- Overview 논문에서 이야기하는 그럼에도 불구하고 마스킹 기반이 매핑 기반보다 우월하다는 말이 무슨 뜻일까?
