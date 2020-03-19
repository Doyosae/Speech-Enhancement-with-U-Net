#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K

from train import *
from utility import *
from dataload import *
from processing import *
import segmentation_models as sm

class Test(TrainUnet):
    
    def __init__ (self):
        
        super().__init__()
        model = self.model((self.image_width, self.image_height, 1))
        model.load_weights("./save/Unet_mse_loss.h5")
        
        self.valid_mask = model.predict(x = self.train_data, 
                                        batch_size = self.batch_size + 30 , 
                                        verbose = 1)
        self.test_mask = model.predict(x = self.test_data, 
                                       batch_size = self.batch_size + 30, 
                                       verbose = 1)
        
        """
        Basic Idea
        1. STFT mixture data
        2. Only get real parts (not abs) and discard imagnary parts
        3. Using SNR, Make ideal binary mask for train data, test data, respectively
        Good.
        4. Unet network is Ideal Binary Mask Estimator
        5. So, Network is trained inputs : mixture data, outputs : estimated mask, label : clean mask
        6. Loss function is MSE or BCE and optimizer is Adam
        
        Test Idea
        1. Network output is estimated mask
        2. So, Need to multiply speech * mask,
        Why? https://www.encyclopediaofmath.org/index.php/Matrix_multiplication
        """
        
        # Calculate estimated speech
        self.estimated_valid_speech = np.multiply(self.train_data, self.valid_mask)
        self.estimated_test_speech = np.multiply(self.test_data, self.test_mask)
        
    def show (self):

        # show Spectogram about train sound -> estimated train speech and
        # test sound (15db) -> estimated test speech
        """
        index1 = 1
        index2 = 21
        index1 = 3
        index2 = 21+22
        index1 = 5
        index2 = 21+22+22
        index1 = 7
        index2 = 21+22+22+22
        index1 = 9
        index2 = 21+22+22+22+22
        """
        index1 = 9
        index2 = 21+22+22+22+22
        showSpectogram(self.train_data[index2], "train_speech",
                       image_size1 = self.image_width, image_size2 = self.image_height)
        showSpectogram(self.estimated_valid_speech[index2], "estimated_valid_speech",
                      image_size1 = self.image_width, image_size2 = self.image_height)
        
        showSpectogram(self.test_data[index1], "test_speech",
                       image_size1 = self.image_width, image_size2 = self.image_height)
        showSpectogram(self.estimated_test_speech[index1], "estimated_test_speech",
                       image_size1 = self.image_width, image_size2 = self.image_height)
        
        showSpectogram(self.test_label[index1], "celar_speech",
                       image_size1 = self.image_width, image_size2 = self.image_height)
        
    def iSTFT (self):
        
        # 음원 파일로 저장하기 위하여 채널 1을 없애고, isftft 처리한 후, 저장
        estimated_train_Speech = np.reshape(self.estimated_valid_speech, 
                                            (self.train_length, self.image_width, self.image_height))
        estimated_test_Speech  = np.reshape(self.estimated_test_speech, 
                                            (self.test_length, self.image_width, self.image_height))

        for index in range (self.test_length):
            test  = librosa.core.istft(estimated_test_Speech[index])
            librosa.output.write_wav("./result/unet_result_" + str(index+1) + ".wav", 
                                     test, sr = 16000, norm = False)
            
if __name__ == '__main__':
    test = Test()
    show = test.show()
    save = test.iSTFT()

