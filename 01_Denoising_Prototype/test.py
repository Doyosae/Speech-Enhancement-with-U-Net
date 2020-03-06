#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from dataload import *
from processing import *
from train import *
from librosa.display import *

arc = enhancement()
model = arc.model()
model.load_weights("speech_enhancement.h5")
trainLabel, testLabel, trainSound, testSound = arc.get_data()

def test(noise_value = 1.):
    """
    테스트로 예측한 것이 훈련으로 개선된 것과 얼마나 차이가 나는지 비교
    """
    def shapeSound (inputs):
        output = np.reshape(inputs, [256, 256])

        return output
    def spectoshow (inputs, title):
        plt.rcParams["figure.figsize"] = (10, 7)
        librosa.display.specshow(librosa.amplitude_to_db(inputs, ref=np.max),
                                y_axis='log', x_axis='time')
        plt.title(title)
        plt.colorbar(format = '%+2.0f dB')
        plt.show()

    noise = np.random.normal(-noise_value, noise_value, size = (210, 256, 256, 1))
    trainResult = model.predict(x = trainSound, batch_size = 60 , verbose = 1)
    testResult  = model.predict(x = testSound+noise, batch_size = 60, verbose = 1)

    view = shapeSound (trainSound[0])
    spectoshow(view, "train Sound")
    view = shapeSound (trainResult[0])
    spectoshow(view, "train sound enhancement")
    view = shapeSound (testSound[0])
    spectoshow(view, "test sound")
    view = shapeSound (testResult[0])
    spectoshow(view, "test sound enhancement + normal noise")
    
if __name__ == '__main__':
    test()

