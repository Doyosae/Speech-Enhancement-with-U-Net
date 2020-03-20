import numpy as np
import librosa
from librosa.display import *
import matplotlib.pyplot as plt

"""
STFT 등의 결과물은 복소수 기반.
1. 실수부를 선택할 것이지 복소수를 선택할 것인지 중요
2. 그리고 U Net 등의 모델에 잘 넣으려면 slice하여 reshape 중요
"""
def get_realData (data, data_length, wanted_size1 = 256, wanted_size2 = 256):
    
    for index in range (data_length):
        data[index] = data[index][:wanted_size1, :wanted_size2]
    
    data = np.reshape(data, (data_length, wanted_size1, wanted_size2, 1))
    data = data.real
    
    print("real data slice and reshape  : ", np.shape(data))
    
    return data


def get_imagData (data, data_length, wanted_size1 = 256, wanted_size2 = 256):
    
    for index in range (data_length):
        data[index] = data[index][:wanted_size1, :wanted_size2]
    
    data = np.reshape(data, (data_length, wanted_size1, wanted_size2, 1))
    data = data.imag
    
    print("imagnary data slice and reshape : ", np.shape(data))
    
    return data
#
#
#
#
"""
trainLabel, trainSound
testLabel, testSound
Ideal Binary Mask를 만드는 함수
"""
def createIdealBinaryMask (clean_data, 
                           mixture_data, 
                           data_length, 
                           image_size1 = 256,
                           image_size2 = 256,
                           alpha = 1.0,
                           criteria = 0.5):

    alpha = alpha
    criteria = criteria
    eps = np.finfo(np.float).eps

    idealBinaryMask = np.divide(np.abs(clean_data)**alpha, (eps + np.abs(mixture_data))**alpha)
    idealBinaryMask[np.where(idealBinaryMask >= criteria)] = 1.
    idealBinaryMask[np.where(idealBinaryMask <= criteria)] = 0.
    
    output = idealBinaryMask
    testImage = np.reshape(idealBinaryMask, (data_length, image_size1, image_size2))
    
    return output


"""
trainLabel, trainSound
testLabel, testSound
Ideal Ratio Mask를 만드는 함수
"""
def createIdealRatioMask (clean_data, 
                          mixture_data, 
                          data_length, 
                          image_size1 = 256,
                          image_size2 = 256,
                          beta = 0.5):
    
    parts1 = np.abs(clean_data)**2
    parts2 = np.abs(clean_data)**2 + np.abs(mixture_data)**2
    
    beta = beta
    eps = np.finfo(np.float).eps
    
    idealRatioMask = np.divide(parts1, (eps+parts2)) ** beta
    
    output = idealRatioMask
    testImage = np.reshape(idealRatioMask, (data_length, image_size1, image_size2))
    
    return output
#
#
#
#
"""
IBM, IRM 등 여러 마스크를 보여주는 함수
"""
def showMask (mask, figure_count = 5, 
              image_size1 = 256, 
              image_size2 = 256,
              cmap = "gray", 
              title = "title"):

    fig, ax = plt.subplots(1, figure_count, figsize = (20, 20))
    for index in range(figure_count):

        ax[index].imshow(np.reshape(mask[5*index], (image_size1, image_size2)), cmap = cmap)

    plt.title(title)
    plt.show ()

"""
librosa.specshow 메서드로 전처리한 데이터 image를 띄우는 함수
"""
def showSpectogram (data, title, 
                    image_size1 = 256, 
                    image_size2 = 256, 
                    fig_size = (10, 7)):
    
    output = np.reshape(data, [image_size1, image_size2])
    librosa.display.specshow(librosa.amplitude_to_db(output, ref = np.max),
                             y_axis = 'log', 
                             x_axis = 'time')
    plt.rcParams["figure.figsize"] = fig_size
    plt.title(title)
    plt.colorbar(format = '%+2.0f dB')
    plt.show()