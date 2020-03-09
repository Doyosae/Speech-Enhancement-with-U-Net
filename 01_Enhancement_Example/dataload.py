import os, librosa
import numpy as np


class dataLoader():
    
    def __init__ (self, sr = 16000, sliceCut1 = 744, sliceCut2 = 930):

        self.sr = sr
        self.sliceCut1 = sliceCut1
        self.sliceCut2 = sliceCut2
        
    def createLabel (self):
        """
        라벨에 사용할 깨끗한 음성을 불러와서 라벨 데이터셋을 만드는 코드
        ./clean/ 폴더에 보면 서로 다른 30개의 깨끗한 음성이 있다.
        """
        labelList, trainLabel, testLabel = [], [], []
        
        # 깨끗한 음성 파일을 모두 풀러오면 labelList은 30개의 경로를 담고 
        for root, dirs, files in os.walk('./clean'):
            for fname in files:
                full_fname = os.path.join(root, fname)
                labelList.append(full_fname)
        
        """
        라벨 데이터 만들기
        훈련 라벨, 검증 라벨 리스트를 만든다.
        1. 첫 번째 사람의 목소리를 24번 append한다.
        2. 두 번째 사람의 목소리를 24번 appned한다.
        3. 세 번째 사람의 목소리를 24번 appned한다.
        ... ...
        이런 방법으로 30명의 사람을 모두 append하면
        총 720개의 훈련용 라벨이 만들어진다.
        
        마찬가지로 이번에는 7번씩 반복하여
        총 210개의 훈련용 라벨이 만들어진다.
        111...222...333...444...
        == 17273747...307
        """
        # len(LabelList) == 30
        for index in range (len(labelList)):
            rawSound, sr = librosa.load(labelList[index], sr = self.sr)

            for repeat in range (24):
                trainLabel.append(rawSound)
            
            for repeat in range (7):
                testLabel.append(rawSound)
                
        return trainLabel, testLabel


    def createTrain (self):
        """
        노이즈의 종류는 8가지이고 0dB, 5dB, 10dB 데이터 구성됨
        따라서 1개의 음성에 가능한 노이즈 조합의 경우의 수는 24개
        총 30명의 사람이므로 720개의 mixture 훈련 세트를 준비
        """
        dataList, trainSound = [], []
        
        for root, dirs, files in os.walk('./train'):
            for fname in files:
                full_fname = os.path.join(root, fname)
                dataList.append(full_fname)
                
        # len(dataList) == 720
        for index in range (len(dataList)):
            noiseSound, sr = librosa.load(dataList[index], sr = self.sr)
            trainSound.append(noiseSound)
            
        return trainSound
    
    
    def createTest (self):
        """
        ./test 폴더에서 15dB의 검증 데이터가 210개 있다.
        15dB 파일은 노이즈 1개에 대해서 존재하지 않아 노이즈 7개로 구성
        30명의 사람이므로 총 210개. 이것을 모두 불러와서 testSound를 만든다.
        1. 첫 번째 사람은 7가지의 노이즈로 구성
        2. 두 번째 사람은 7가지의 노이즈로 구성
        3. 세 번째 사람은 7가지의 노이즈로 구성
        ... ... ...
        """
        dataList, testSound = [], []
        
        for root, dirs, files in os.walk('./test'):
            for fname in files:
                full_fname = os.path.join(root, fname)
                dataList.append(full_fname)
        
        # len(dataList) == 210
        for index in range (len(dataList)):
            noiseSound, sr = librosa.load(dataList[index], sr = self.sr)
            testSound.append(noiseSound)
            
        return testSound
        

    def callData (self):
        """
        훈련용 라벨, 테스트용 라벨, 훈련용 데이터, 테스트용 데이터의 크기는
        (720,) (210,) (720,) (210,)
        """
        trainLabel , testLabel = [], []
        trainData, testData = [], []
        
        # (훈련 및 검증) 라벨, 훈련 데이터, 검증 데이터를 call
        trainLabel, testLabel = self.createLabel()
        trainData = self.createTrain()
        testData = self.createTest()

        # size is (720,) (210,) (720,) (210,), repectively
        print("훈련 라벨의 크기   : ", np.shape(trainLabel))
        print("검증 라벨의 크기   : ", np.shape(testLabel))
        print("훈련 데이터의 크기 : ", np.shape(trainData))
        print("검증 데이터의 크기 : ", np.shape(testData))
        
        print("\n")
        print("각 데이터마다의 시퀀스 길이 (음성 길이) \n")
        print("Length of trainLabel sequence : ", np.shape(trainLabel[0]))
        print("Length of testLabel sequence  : ", np.shape(testLabel[0]))
        print("Length of trainData sequence  : ", np.shape(trainData[0]))
        print("Length of testData sequence   : ", np.shape(testData[0]))
        
        return trainLabel, testLabel, trainData, testData
    

class fixedLength (dataLoader):
    """
    이 클래스를 만든 목적.
    1. 위의 dataLoader 클래스는 소리의 시퀀스가 가변적이다.
    2. 가변적인 시퀀스는 RNN, LSTM에 대응하면 되지만, CNN 등에 대응하기는 힘듬
    3. 따라서 고정 길이의 시퀀스를 만들기 위하여 이 클래스가 필요할 것.
    - 모든 음성 중에서 가장 짧은 길이의 음성을 선택
    - 그 음성을 기준 길이로 하여, 모든 음성을 다 잘라서 나머지는 버림
    """
    
    def slicingData (self):
        
        """
        super().callData()로 데이터 불러오기
        1. 각 데이터 모음마다, 음성의 시퀀스 길이를 받아서 append한다.
        2. 길이 리스트 중 min()으로 최솟값 추출
        3. 이 길이를 토대로 모든 데이터들을 slice
        """
        trainLabel, testLabel, trainData, testData = super().callData()
        
        sliceLength = []
        
        # Find minimus array element size , for slicing all array
        for index in range (len(trainLabel)):
            sliceLength.append(len(trainLabel[index]))
            
        minLenght = min(sliceLength)
        
        for index in range (len(trainLabel)):
            trainLabel[index] = trainLabel[index][:minLenght]
            trainData[index] = trainData[index][:minLenght]
            
        for index in range (len(testLabel)):
            testLabel[index] = testLabel[index][:minLenght]
            testData[index] = testData[index][:minLenght]
            
        return trainLabel, trainData, testLabel, testData
    
    
    def callData (self):
        """
        훈련용 라벨, 테스트용 라벨, 훈련용 데이터, 테스트용 데이터의 크기는
        (720,) (210,) (720,) (210,)
        """
        trainLabel , testLabel = [], []
        trainData, testData = [], []
        
        trainLabel, trainData, testLabel, testData = self.slicingData()

        # size is (720,) (210,) (720,) (210,) repectively
        print("분리한 훈련 라벨의 크기   : ", np.shape(trainLabel))
        print("분리한 훈련 데이터의 크기   : ", np.shape(trainData))
        print("분리한 검증 라벨의 크기 : ", np.shape(testLabel))
        print("분리한 검증 데이터의 크기 : ", np.shape(testData))

        return trainLabel, testLabel, trainData, testData
    
    
if __name__ == '__main__':
    data = fixedLength()
    data.callData()