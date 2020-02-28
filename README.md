# Paper
#
## Supervised Speech Separation Based on Deep Learning: An Overview (DeLiang Wang, Fellow, IEEE, and Jitong Chen)  
### 요약  
최근의 접근 방법은 speech separation를 supervised learning problem으로 공식화한다.  
이렇게 지도 학습으로 푸는 것은 훈련 데이터로부터 음성, 화자, 배경 노이즈로부터 구별 가능한 패턴을 학습하는 것이다.  
음원 분리의 목표는 방해 background interference로부터 target speech를 분리하는 것이다.  
사람은 여러 음원 소스가 섞여 있어도, 듣고 싶은 것을 잘 듣는다. 그래서 음원 분리는 곧 칵테일 문제와 같다.  
### 
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
