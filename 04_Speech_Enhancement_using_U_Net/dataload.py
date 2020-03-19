import os, librosa
import numpy as np


class dataLoader():
    
    def __init__ (self, sr = 16000, train_label_size = 22, test_label_size = 2):

        self.sr = sr
        self.train_label_size = train_label_size
        self.test_label_size = test_label_size
        
    def createLabel (self):
        """
        라벨에 사용할 음성을 불러와서 라벨 데이터셋을 만드는 코드
        노이즈 음성이 한 음성 당 31개의 노이즈여서, 해당 라벨도 각각 31개씩 맞춤
        1 : label[0] ~ label[30], 31개
        2 : label[31] ~ label[61], 31개
        3 : label[62] ~ label[92], 31개
        ... ... 해서 30번 음성까지 있음 (index range is from 0 to 929
        """
        labelList, trainLabel, testLabel = [], [], []
        
        # 깨끗한 음성 파일을 모두 풀러오면 labelList은 30개의 경로를 담고 
        for root, dirs, files in os.walk('./clean'):
            for fname in files:
                if fname == "desktop.ini":
                    continue 
                full_fname = os.path.join(root, fname)
                labelList.append(full_fname)
        
        # len(LabelList) == 30
        for index in range (len(labelList)):
            rawSound, sr = librosa.load(labelList[index], sr = self.sr)

            for repeat in range (self.train_label_size):
                trainLabel.append(rawSound)
            
            for repeat in range (self.test_label_size):
                testLabel.append(rawSound)
                
        return trainLabel, testLabel


    def createTrain (self):
        """
        훈련에 사용할 mixture 데이터 720개
        1번 음성에 대하여 24가지 노이즈가 껴있음 (0dB, 5dB, 10dB 마다)
        2번 음성도 마찬가지로 31가지 노이즈가 껴있음
        이렇게 30번까지 음원을 고려하면 720개의 훈련셋
        """
        dataList, trainSound = [], []
        
        for root, dirs, files in os.walk('./train'):
            for fname in files:
                if fname == "desktop.ini":
                    continue 
                full_fname = os.path.join(root, fname)
                dataList.append(full_fname)
                
        # len(dataList) == 720
        for index in range (len(dataList)):
            noiseSound, sr = librosa.load(dataList[index], sr = self.sr)
            trainSound.append(noiseSound)
            
        return trainSound
    
    
    def createTest (self):
        """
        테스트에 사용할 데이터 210개
        """
        dataList, testSound = [], []
        
        for root, dirs, files in os.walk('./test'):
            for fname in files:
                if fname == "desktop.ini":
                    continue 
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
        (930,) (930,) (45058,) (45058,)
        """
        trainLabel , testLabel = [], []
        trainData, testData = [], []
        
        trainLabel, testLabel = self.createLabel()
        trainData = self.createTrain()
        testData = self.createTest()

        # size is (744,) (186,) (744,) (186,), repectively
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
    """
    
    def slicingData (self):
        
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
        (930,),  (930,), (45058,) (45058,)
        """
        trainLabel , testLabel = [], []
        trainData, testData = [], []
        
        trainLabel, trainData, testLabel, testData = self.slicingData()

        # size is (744,) (186,) (744,) (186,), repectively
        print("분리한 훈련 라벨의 크기   : ", np.shape(trainLabel))
        print("분리한 훈련 데이터의 크기   : ", np.shape(trainData))
        print("분리한 검증 라벨의 크기 : ", np.shape(testLabel))
        print("분리한 검증 데이터의 크기 : ", np.shape(testData))

        return trainLabel, testLabel, trainData, testData
    
    
if __name__ == '__main__':
    data = fixedLength()
    data.callData()