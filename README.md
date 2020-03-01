# Paper
#
## Supervised Speech Separation Based on Deep Learning: An Overview (DeLiang Wang, Fellow, IEEE, and Jitong Chen)  
### 요약  
최근의 접근 방법은 speech separation를 supervised learning problem으로 공식화한다.  
이렇게 지도 학습으로 푸는 것은 훈련 데이터로부터 음성, 화자, 배경 노이즈로부터 구별 가능한 패턴을 학습하는 것이다.  
음원 분리의 목표는 방해 background interference로부터 target speech를 분리하는 것이다.  
사람은 여러 음원 소스가 섞여 있어도, 듣고 싶은 것을 잘 듣는다. 그래서 음원 분리는 곧 칵테일 문제와 같다.  
### 개요  
어떻게 음원 분리를 잘 해낼 수 있을까?  
그 중 한가지 방법은 50% intelligibility score를 위해 필요한 SNR 수준에서 음성 수신 임계값을 측정하는 것이다.  
서로 다른 interference로부터 SRT 값의 차이가 매우 크게 난다. (SRT는 50% 점수를 위한 SNR 값)  
지난 수 십년간 음성 분리는 신호 처리 영역에서 연구되었는데, 센서와 마이크로 폰의 갯수에 의존하는 방법으로는 아래와 같다.  
- Monaural (single microphone)  
- Array based (multi microphone)  
#
Monaural seperation을 위한 두 전통적인 방법은 Speech enhancement와 CASA이다.
Speech enhancement는 노이즈의 일반적인 통계를 분석하고, 추정한 소음을 포함하는 음원에서 깨끗한 speech를 걸러낸다.
이러한 알고리즘으로 spectrum substraction이 유명하다. 이것은 추정한 노이즈의 강력한 스펙트럼이 노이즈가 낀 speech에서 빠진다.
이때 가정을 하나 한다. 배경 노이즈가 정적이라는 것이다. 하다못해 최소한 speech보다 정적이여야 한다.
CASA (computational auditory scene analysis)는 청각 장면 분석의 지각 원리에 기초한다.
그리고 pitch와 onset과 같은 그룹화된 큐들을 적극 이용한다.
#
두 개 이상의 마이크가 있는 경우는 speech seperation을 위한 다른 원리를 사용한다.
Beamforming 또는 공간 필터링이라는 방법은 적절한 배열 구성을 통해, 특정 방향에 도달하는 신호를 가속시킨다. (증폭?) 
그렇게 함으로써 다른 방향에서 오는 간섭을 감소시킨다. 
이러한 노이즈 감쇠의 양은 일반적으로 어레이의 간격, 크기 및 구성에 따라 달라진다.
#
보다 최근의 방법은 speech seperation을 지도 학습 문제로 다루는 것이다. 
지도 음성 분리의 원래 방식은 CASA에서 time frequency masking의 개념에서 영감을 받았다. 
분리 수단으로, TF 마스킹은 목표하는 소스를 분리하기 위해, 섞인 소스의 TF 표현에다가 2차원 마스크를 (가중치) 적용한다. 
CASA의 주요 목표는 이상적인 바이너리 마스크이다. 이 마스크는 섞인 신호의 TF 표현에서 목표하는 신호가 TF unit을 지배하는지 여부를 나타낸다. 
위에서 말한 이상적인 바이너리 마스크가 목표라면, 음성 분리는 지도 학습으로 수행되는 이진 분류 문제가 된다.  
#
음성 분리를 분류 문제로 보기 시작하면서, 데이터 기반 접근법은 음성 처리 커뮤니티에서 넓게 연구되어 왔다. 
지난 10년 동안 지도 음성 분리는 최첨단 성능을 크게 향상시켰다. 
지도 학습 분리는 특히 딥 러닝의 성장으로부터 큰 혜택을 보았다. 앞으로 볼테지만 말이다. 
지도 학습 기반 음성 분리 알고리즘은 크게 학습하는 기계, 교육하는 대상, 음향학적 특징으로 나눌 수 있다. 
이 문서에서는 먼저 세 가지 구성 요소를 검토하고, 별도로 monural 및 array 기반 대표적인 알고리즘을 설명한다.  
### Section II, 분류기와 학습하는 기계  
심층 신경망은 지난 10년간 많은 지도 학습 작업의 성능을 크게 높였다. 
지도 학습 기반 Speech seperation을 위한 심층 신경망은 큰 이점을 가지고 왔는데, 이 논문에서는 MLP, CNN, RNN, GAN 모델들을 검토한다. 
... ... 라고 하지만 신경망 종류와 기초적인 내용들이라 이미 다 알고 있으므로 자세히는 패스... 
DNN에서 크로스 엔트로피는 보편적인 손실함수이고, 회귀 분석에서의 손실함수는 MSE를 쓴다. 
파라미터의 초기화 문제와 그래디언트 배니싱 문제가 있다. 
레이어가 깊어지면서 미분값을을 많이 곱하여 오차를 줄이다보니, 그 값이 0에 수렴해서 파라미터 업데이트가 안되는 현상이다. 
비선형 활성함수를 도입해서 이러한 문제를 해결할 수 있다. 이를테면 ReLU 같은 것들이다. 
그 다음 문단에서는 합성곱 신경망에 대하여 설명한다. 이 신경망은 지금도 많이 사용하므로 설명을 패스한다. 
GAN도 현재 계속 공부하고 있으므로 자세한 내용은 생략한다. 
참고로 오리지널 GAN은 KL Divergence의 한계가 있어서, 나중의 GAN에서는 전혀 다른 거리 함수를 정의하여 모델이 나왔다. 
#
RNN은 그리 많이 사용해보지 않았다. 그래서 이 부분은 짚고 넘어간다. 
RNN은 숨겨진 단위 사이에서 연결되는 재귀 신경망이다. 피드포워드 신경망과는 달리 RNN은 입력 샘플을 Sequence로 처리하고, 시간에 따라 변하는 모델이다. 
사실 음성 신호라는 것이 시간 도메인을 기반으로 한 구조이고, 현재 프레임에 있는 신호는 이전 프레임의 신호에 영향을 받는다. 
그러므로 음성의 시간에 의존하는 성질 때문에 RNN을 선택하는 것은 매우 자연스럽다. 
우리는 반복적인 연결을 통해 RNN이 유연하고 무한히 확장 가능한 시간 차원을 도입한다는 것에 주목한다. 
이 특징은 DNN와는 달리 아무리 깊더라도 문제가 되지 않는다. 재귀 신경망 역시 시간을 통한 backpropagation으로 학습한다. 
그러나 RNN 역시 그래디언트 배니싱 또는 폭발 문제로부터 자유롭지 못하다. 
이를 개선한 모델이 LSTM이다. LSTM은 시간이 오래 지난 정보는 잊어버려서 정보 흐름이 원활하다. 
이 LSTM의 메모리 셀에는 입력 게이트, 망각 게이트, 출력 게이트 세 가지로 구성되어 있다. 
망각 게이트는 이전 정보를 얼마나 보존해야 하는지 제어한다. 그리고 입력 게이트는 얼마나 현재 정보가 메모리 셀에 추가되어야 하는지 제어한다. 
이러한 게이트 기능으로 LSTM은 맥락과 연관된 정보를 메모리 셀에 유지하여 RNN 성능을 향상시킨다.  
#
여하튼 우리는 이제 히든 레이어를 가진 DNN을 Speech seperation을 위한 기계로 논의한다. 
서포트 벡터 머신이나, 믹스처 가우시안 모델 등과 같이 전통적인 머신러닝 모델들은 앞으로 논의 하지 않는다.  
### III. 훈련 대상  
지도 학습 기반 음성 분리에서 학습과 일반화를 위해 적절한 훈련 대상을 찾는 것이 중요하다. 
훈련 대상은 주로 마스킹 기반 대상과 매핑 기반 대상으로 나눈다. 
마스킹 기반 대상은 배경 간섭으로부터 깨끗한 음성의 시간 - 주파수 관계를 설명한다. 
반면에 매핑 기반 대상은 깨끗한 음성의 스펙트럼 표현에 해당한다. 
훈련 대상을 설명하기에 앞서, 먼저 음성 분리에서 흔히 사용하는 평가 메트릭에 대해 이야기한다. 
메트릭들은 두 가지로 구분이 가능하다. 하나는 신호 레벨이고 다른 하나는 인식 레벨이다. 
신호 레벨에서 메트릭은 신호 향상 또는 간섭 감소의 정도를 정량화하는 것을 목표한다.
대표적인 평가 메트릭 집합은 신호 대 왜곡, 신호 대 간섭, 신호 대 아티팩트 비율로 이루어진다. 
# 
객관적인 메트릭들이 음성의 명료성과 음성의 품질의 평가를 분리하기 위하여 개발되었다. 
HIT는 정확하게 분류된 IBM에서 음성을 지배하는 시간 - 주파수 유닛의 백분율을 보여준다. 
FA은 (False Alarm) 잘못 분류된 노이즈를 지배하는 유닛의 백분율을 보여준다. 
그래서 HIT - FA는 음성의 명료성에서 잘 작동한다. 왜? 각 유닛에서 노이즈가 지배적인지 음성이 지배적의 지표이므로. 
#  
최근에 흔히 사용하는 명료성 평가 메트릭은 shot time objective intelligibility이다. 이것은 추론한 발성과 분리된 발성의 short time temporal envelopes 간의 상관관계를 측정한다. 
음성 품질에 대해서는 ITU가 권장하는 PESQ가 표준 평가 지표이다. 
PESQ는 음량 스펙트럼을 생성하기 위해 audiotory transform을 적용한다. 
그리고 깨끗한 추론 신호와 분리된 신호의 음량 스펙트럼을 비교하여 -0.5에서 4.5 사이의 점수를 산출한다.  
#### A. Ideal Binary Mask (IBM)  
IBM은 청각 장면 해석의 독점 할당 원리와 오디션에서의 청각적 마스킹 현상에 영감을 얻었다. 
IBM은 노이즈가 낀 신호의 시간 - 주파수 표현에서 정의된다. (예를 들면 스펙토그램) 
IBM은 시간 - 주파수 단위 안에서 SNR이 로컬 기준 내지 임계값보다 크면, 그 단위에 1을 할당한다.  
IBM은 청각 장애를 위해 음성의 명료성을 획기적으로 향상시킨다. (1 아니면 0이므로, 바이너리) 
IBM은 모든 시간 - 주파수 유닛에 target-dominant이든 interference-dominannt이든 이름을 붙인다. 
결과적으로 IBM의 추정은 지도 학습으로 다룰 수 있다. (타겟이냐? 간섭이냐? 문제) 
IBM의 추정을 위해 사용되는 비용 함수는 보통 cross entorpy이다.  
#### B. Target Binary Mask  
IBM처럼, TBM은 모든 시간 - 주파수 단위를 이진 레이블로 분류한다. 
차이점이라면 TBM은 각각의 시간 - 주파수 단위에서 목표 음성 에너지와 고정 간섭을 비교하여 레이블을 얻는다. 
이 고정 간섭은 노이즈 형태의 음성인데, 모든 음성 신호의 평균과 일치하는 정지 신호이다. 
(앞에서 간섭이 정지 신호여야 한다는 가정과 일치함, 모든 신호의 평균은 사실상 정지 신호?) 
TBM 역시 음성의 명료성을 획기적으로 향상시킨다.  
#### C. Ideal Ratio Mask  
각 TF 단위의 하드한 라벨링 대신, IRM은 IBM의 부드러운 버전이다. 
이것은 TF 단위 내의 음성 에너지와 노이즈 에너지를 이용한다. 튜닝 가능한 파라미터는 리스케일링 하여 0.5로 선택한다. 
S와 N이 상관없다는 가정 하에 IRM의 제곱근은 TF 유닛의 음성 에너지를 보존한다. 
이 가정은 겹쳐진 노이즈에는 적합하지만, 실내 반향같이 난향성 노이즈에는 적합하지 않다. 
제곱근이 없다면 IRM은 세기 스펙트럼에서 대상 음성의 최적의 추정기인 Winer filter와 유사하다. 
MSE가 일반적으로 IRM 추정의 비용 함수로 쓰인다. 
#### D. 스펙트럼 크기 마스크
스펙트럼 크기 마스크 SMM은 깨끗한 음성과 노이즈 음성의 진폭을 STFT 하는 것에서 정의된다. 
수식 표현에서 S(t, f)와 Y(t, f)는 각각 깨끗한 음성의 스펙트럼 크기와 노이즈 음성 스펙트럼의 크기이다. 
IRM과는 달리 SMM은 상한값이 1로 제한되지 않는다.
분리된 음성을 얻기 위하여, 우리는 SMM 또는 이것의 추정값을 노이즈 음성의 스펙트럼 크기에 적용한다. 
그리고 노이즈 음성의 페이즈와 분리된 음성을 재합성한다.
#### E. 위상에 민감한 마스크  
위상에 민감한 마스크, PSM은 위상의 측정을 포함해서 SMM을 확장한다. 
여기세 세타는 TF 단위의 깨끗한 음성과 노이즈 음성의 위상 차이를 나타낸다. 
PSM에 위상 차이를 포함하는 것은 신호 대 잡음 비를 더 높인다. 그리고 SMM 보다 더 나은 깨끗한 음성의 추정치를 산출한다. 
#### F. 복소수 기반 IRM  
복소수 기반 IRM은 복소수 도메인에서의 이상적인 마스크이다. 
앞서 이야기한 마스크와는 달리, 이것은 완벽하게 노이즈 음성으로부터 깨끗한 음성을 재구성할 수 있다. (복소수라서) 
S, Y는 각각 깨끗한 음성과 노이즈 음성의 STFT 결과를 나타낸다. 그리고 *는 복소수 곱연산을 말한다. 
cIRM의 정의식을 보면 real Y와 imag Y가 있다. 이는 각각 노이즈 음성의 실수 성분과 허수 성분을 가리킨다. 
마찬가지로 깨끗한 음성인 S에도 실수 성분과 허수 성분이 있다. 
허수 파트는 “I” 라는 기호로 표시된다. 그리고 cIRM은 실수 성분과 허수 성분 모두 가진다. 
또 실수 도메인에서 별도로 값을 추정할 수도 있다. 복소수 연산이기 때문에 마스크의 값은 경계가 없다. (제한이 없다.) 
그래서 어떤 압측된 형태를 마스크 값에 적용해야 한다. 예를 들면 탄젠트 함수나 시그모이드 같이 말이다.
#### G. Target Magnitude Spectrum
깨끗한 음성의 TSM, 또는 S(t, f)는 훈련 데이터에 기반한 매핑이다. 
이러한 지도 학습 목표는 노이즈 음성으로 부터 깨끗한 음성의 크기를 추정한다. 
Power spectrum 또는 Mel spectrum 같은 형태가 Magnitude spectrum 대신에 사용될 수 있다.
일반적으로 훈련을 쉽게 하고, 역동적인 범위를 압축시키기 위해 로그 연산이 쓰인다.
잘 알려진 TMS 형태는 평균이 0으로 정규화된 Log power - spectrum이다. 
계산된 음성의 magnitude는  잡음의 위상과 조합되어 분리한 음성의 파형을 생성한다. 
비용 함수로는 MSE가 TMS 계산에 종종 쓰인다. 
또는 결과의 상관관계를 명시적으로 모델링하는 TMS 계산기를 위해서는 최대우도 방법이 쓰인다.
#### H. Gammatone Frequency Target Power Spectrum
GF-TPS 이라는 방법도 있다. 이 역시 mapping - based target에 기반한다. spectrogram에서 정의된 TMS와는 달리,
이것의 목표는 gammatone filterbank에 기반하는 cochleagram에서 정의된다.
구체적으로는 깨끗한 음성을 위한 cochleagram 반응의 세기에 달려있다. 
cochleagram의 역계산을 통하여 GF - TPS의 추정치는 분리한 음성 파형으로 쉽게 변환된다. 
#### I. Signal Approximation  
신호 근사의 생각은 깨끗한 음성의 스펙트럼 세기와 추정된 음성과의 차이를 최소화하는 ratio maks estimator을 학습하는 것이다.
SA(t, f)의 정의식에서 RM(t, f)는 SMM의 추정값을 말한다.
그래서 SA는 SNR 최소화를 추구하는 ratio masking과 spectral mapping의 조합된 목표로 해석할 수 있다. 
이와 관련된 이전의 사례는 IBM 추정치의 맥락에서 SNR를 최대화하는 목표로 한다. 
SA 목표는 다음의 두 단계로 더 나은 성능을 보여준다. 첫 번째는 SMM을 목표로 학습하는 것이다. 
두 번째는 손실 함수의 최소화로 학습을 미세 조정한다. 
#### Training targets에 대한 결론
다양한 훈련 대상을 고정된 deep neural network를 사용하여 동일한 입력 데이터들로 비교하였다. 
다양한 훈련 대상을 사용하여 분리된 음성은 STOI와 PESQ 조건에서 예측된 음성 명료성과 음성 품질을 평가한다. 
게다가, 대표적인 음성 향상 알고리즘과 지도 기반 음수가 없는 matrix factorization 알고리즘이 벤치마크로 사용된다. 
- 최근 연구는 masking이 오직 높은 SNR에서만 유리하고 낮은 SNR에서는 mapping이 더 큰 장점을 가지는 것을 보여주었지만, 여전히 masking based targets이 mapping based targets을 능가한다.
- 음성 품질 관점에서는 ratio masking이 binary masking보다 더 성능이 우수하다.
- SMM은 간섭 신호와 SNR에 민감하다. TMS는 간섭 신호와 SNR에 민감하지 않다.
- TMS의 디테일한 매핑은 SMM의 그것보다 추정이 더 어렵다. 그리고 스펙트럼 크기의 한계가 없는 것은 추정의 오류를 확대시킨다. 
- IRM과 SMM이 선호된다. 심층 신경망 기반 ratio masking이 지도 기반 NMF와 비지도 음성 향상보다 더 우수한 성능을 발휘한다.
