import os, librosa
import numpy as np

import dataload
from dataload import dataLoader
from dataload import fixedLength


class dataProcessing (fixedLength):
    
    
    def __init__ (self, sr = 16000, n_fft = 512, n_mfcc = 40, hop_length = 512):
        
        super().__init__ (sr = 16000, train_label_size = 22, test_label_size = 2)
        """
        부모 클래스에서 초기값들을 받아옴
        dataLoader의 __init__ 값들을 fixedLength에 상속하였고 다시 fixedLength를 이 클래스에 상속
        따라서 dataProcessing 클래스는 dataLoader의 __init__ 값들을 상속 받은 것
        """
        
        self.sr = sr
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        
        self.trainLabel = []
        self.testLabel = []
        self.trainData = []
        self.testData = []
        
        """
        super().callData()에서 반환값이 총 네 개.
        return trainLabel, testLabel, trainData, testData
        """
        self.loadedCallDataList = super().callData()
        print("\n")
        print("Check size of loadedCallDataList", np.shape(self.loadedCallDataList))
        print("Check the element size of loadedCallDataList, respectively")
        print("trainLabel :", np.shape(self.loadedCallDataList[0]))
        print("testLabel  :", np.shape(self.loadedCallDataList[1]))
        print("trainData  :", np.shape(self.loadedCallDataList[2]))
        print("testData   :", np.shape(self.loadedCallDataList[3]))
        print("\n")
        
        
    def stft (self):
        """
        librosa.core.stft 특징
        타입은 numpy.complex64, n_fft 값에 따라 반환 값의 행, 열 사이즈가 달라짐
        입력 사이즈 33856에서
        n_fft = 2048 -> (1025, 89)
        n_fft = 1024 -> (513, 177)
        n_fft = 512 -> (257, 353) // n_fft == win_length
        
        STFT의 수학적 특징
        1. 시간 구간 별로 잘라서, 해당 윈도우의 주파수를 보는 것
        2. 윈도우의 길이가 크면, 주파수 해상도가 낮아짐 (주파수를 정밀하게 볼 수 없음)
        3. 윈도우의 길이가 작으면, 주파수 해상도가 커짐 (주파수를 정밀하게 볼 수 있음)
        librosa.core.stft(y:np.ndarray [shape=(n,)], real-valued, 
                        n_fft:int > 0 [scalar], 
                        hop_length:int > 0 [scalar] 
                        If unspecified, defaults to win_length / 4 (see below)., 
                        
                        win_length:int <= n_fft [scalar], 
                        window:string, tuple, number, function, or np.ndarray [shape=(n_fft,)], 
                        center=True)
        """
        dataSTFT = [[], [], [], []]
        
        for index1 in range (len(self.loadedCallDataList)):
            
            for index2 in range (len(self.loadedCallDataList[index1])):
                
                STFT = librosa.core.stft(y = self.loadedCallDataList[index1][index2], 
                                         n_fft = self.n_fft)
                dataSTFT[index1].append(STFT)

        output1 = dataSTFT[0]
        output2 = dataSTFT[1]
        output3 = dataSTFT[2]
        output4 = dataSTFT[3]
        
        print("\n")
        print("훈련용 라벨, 테스트 라벨")
        print(np.shape(output1), np.shape(output2))
        print("훈련용 데이터, 테스트 데이터")
        print(np.shape(output3), np.shape(output4))
        
        output = [output1, output2, output3, output4]

        return output

    
    def melspectrogram (self):
        """
        librosa.feature.melspectrogram 특징
        1. 타입은 복소수가 없는 실수
        2. n_fft 값에는 사이즈가 의존하지 않음
        3. hop_length 값에 따라
            반환 값의 행, 열 사이즈가 달라짐
            입력 사이즈 45058에서
            hop_lenth = 512 -> (128, 89)
            hop_lenth = 256 -> (128, 177)
            hop_lenth = 128 -> (128, 353)
        """
        dataMelSpecto = [[], [], [], []]
        
        for index1 in range (len(self.loadedCallDataList)):
    
            for index2 in range (len(self.loadedCallDataList[index1])):
                
                melSpec = librosa.feature.melspectrogram(y = self.loadedCallDataList[index1][index2], 
                                                        sr = self.sr, 
                                                        S = None, 
                                                        n_fft = self.n_fft, 
                                                        hop_length = self.hop_length)
                dataMelSpecto[index1].append(melSpec)

        output1 = dataMelSpecto[0]
        output2 = dataMelSpecto[1]
        output3 = dataMelSpecto[2]
        output4 = dataMelSpecto[3]
        
        print("\n")
        print("훈련용 라벨, 테스트 라벨")
        print(np.shape(output1), np.shape(output2))
        print("훈련용 데이터, 테스트 데이터")
        print(np.shape(output3), np.shape(output4))
        
        output = [output1, output2, output3, output4]

        return output

    
    def mfcc (self):
        """
        librosa.feature.mfcc 특징
        1. 타입은 복소수가 없는 실수
        2. n_mfcc 값에 따라 반환 값의 행, 열 사이즈가 달라짐
            입력 사이즈 45058에서
            n_mfcc = 20 -> (20, 89)
            n_mfcc = 40 -> (40, 89) (기본값)
            n_mfcc = 60 -> (60, 89)
            
        """
        dataMFCC = [[], [], [], []]
        
        for index1 in range (len(self.loadedCallDataList)):
    
            for index2 in range (len(self.loadedCallDataList[index1])):
            
                mfcc = librosa.feature.mfcc(y = self.loadedCallDataList[index1][index2], 
                                             sr = self.sr, 
                                             n_mfcc = self.n_mfcc)
                dataMFCC[index1].append(mfcc)

        output1 = dataMFCC[0]
        output2 = dataMFCC[1]
        output3 = dataMFCC[2]
        output4 = dataMFCC[3]
        
        print("\n")
        print("훈련용 라벨, 테스트 라벨")
        print(np.shape(output1), np.shape(output2))
        print("훈련용 데이터, 테스트 데이터")
        print(np.shape(output3), np.shape(output4))
        
        output = [output1, output2, output3, output4]
        
        return output


if __name__ == "__main__":
    data = dataProcessing()
    out = data.mfcc()