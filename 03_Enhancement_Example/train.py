#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from utility import *
from dataload import *
from processing import *
from librosa.display import *

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K


class enhancement():
    
    def __init__ (self, train_length = 720, test_length = 210, input_size = 256):
        
        callData = dataProcessing()
        self.trainLabel, self.testLabel, self.trainSound, self.testSound = callData.stft()
        
        self.train_IdealBinaryMask = []
        self.test_IdealBinaryMask = []

        self.train_length = train_length #720
        self.test_length  = test_length  #210
        self.input_size   = input_size   #256
        
        self.filter1 = 32
        self.filter2 = 64
        self.filter3 = 128
        self.filter4 = 256
        self.filter5 = 512
        self.filter6 = 1024
    
        self.dropout_rate = 0.3
        self.kernel_size = (3, 3)
        self.strides = (2, 2)
        
        self.padding_valid = "valid"
        self.padding_same = "same"
        
        self.batch_size = 60
        self.epochs = 50
        
        self.learning_rate = 0.00015
        self.beta_1 = 0.5
        
        # trainLabel, testLabel, trainSound, testSound
        # call getRealData utility
        self.trainLabel = get_realData(self.trainLabel, 720, 256)
        self.testLabel  = get_realData(self.testLabel, 210, 256)
        self.trainSound = get_realData(self.trainSound, 720, 256)
        self.testSound  = get_realData(self.testSound, 210, 256)
        
        
        # 깨끗한 음성은 훈련 라벨, 더러운 음성은 훈련 데이터, 그 길이는 720
        self.train_IdealBinaryMask = createIdealBinaryMask (clean_data = self.trainLabel, 
                                                            mixture_data = self.trainSound, 
                                                            data_length = self.train_length, 
                                                            image_size = 256,
                                                            alpha = 1.0,
                                                            criteria = 0.5)
        # 깨끗한 음성은 검증 라벨, 더러운 음성은 검증 데이터, 그 길이는 210
        self.test_IdealBinaryMask = createIdealBinaryMask (clean_data = self.testLabel, 
                                                            mixture_data = self.testSound, 
                                                            data_length = self.test_length, 
                                                            image_size = 256,
                                                            alpha = 1.0,
                                                            criteria = 0.5)
            
        """
        We get [trainSound, trainLabel], [testSound, testLabel] &&
        Ideal Binary Mask for training datasets, Ideal Binary Mask for testing datasets
        """
        
        print("\n\n 이 모델을 위해 전처리한 훈련셋과 IBM 셋의 크기들")
        print("훈련 데이터 크기   : ", np.shape(self.trainSound), np.shape(self.trainLabel))
        print("검증 데이터 크기   : ", np.shape(self.testSound), np.shape(self.testLabel))
        print("마스크 데이터 크기 : ", np.shape(self.train_IdealBinaryMask), 
                                      np.shape(self.train_IdealBinaryMask))
        
    def get_data(self):
        
        out1 = self.trainLabel
        out2 = self.testLabel
        out3 = self.trainSound
        out4 = self.testSound
        
        print ("\n\n")
        print ("returning seqeunce is trianLabel, testLabel, trainSound, testSound")
        print ("\n\n")
        
        return out1, out2, out3, out4
    
    def model(self):
        """
        self.filter1 = 32
        self.filter2 = 64
        self.filter3 = 128
        self.filter4 = 256
        self.filter5 = 512
        self.filter6 = 1024
        self.kernel_size = (3, 3)
        self.strides = (2, 2)
        self.padding_valid = "valid"
        self.padding_same = "same"
        """
        inputs = Input(shape = (self.input_size, self.input_size, 1))
        output = Conv2D(self.filter1, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.padding_same,
                     activation = None)(inputs)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2D(self.filter2, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.padding_same,
                     activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2D(self.filter3, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.padding_same,
                     activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2D(self.filter4, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.padding_same,
                     activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2D(self.filter5, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.padding_same,
                     activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2D(self.filter6, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.padding_same,
                     activation = None)(output)
        output = tf.nn.relu(output)

        output = Flatten()(output)
        output = Dropout(rate = self.dropout_rate)(output)
        output = Dense(2048, activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Dropout(rate = self.dropout_rate)(output)
        output = Dense(512, activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Dropout(rate = self.dropout_rate)(output)
        output = Dense(2048, activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Dropout(rate = self.dropout_rate)(output)
        output = Dense(16384, activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, (-1, 4, 4, 1024))

        output = Conv2DTranspose(self.filter5, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.padding_same,
                     activation=None)(output)
        output = tf.nn.relu(output)
        output = Conv2DTranspose(self.filter4, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.padding_same,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2DTranspose(self.filter3, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.padding_same,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2DTranspose(self.filter2, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.padding_same,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2DTranspose(self.filter1, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.padding_same,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.relu(output)
        output = Conv2DTranspose(1, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.padding_same,
                     activation=None)(output)
        output = tf.nn.sigmoid(output)

        model = Model (inputs = inputs, outputs = output)
        model.summary()

        """
        self.batch_size = 60
        self.epochs = 50
        
        self.learning_rate = 0.00015
        self.beta_1 = 0.5
        """
        return model
    
    def train(self):
        
        def rmse(y_true, y_pred):
            output = tf.sqrt (2 * tf.nn.l2_loss(y_true - y_pred)) / self.batch_size
            return output
    
        model = self.model()
        
        mean_squared_error = "mean_squared_error"
        binary_cross_entopy = "binary_crossentropy"
        model.compile(optimizer = Adam(lr = self.learning_rate, 
                                    beta_1 = self.beta_1, 
                                    beta_2 = 0.999, 
                                    epsilon = None, 
                                    decay = 0.0, 
                                    amsgrad = False),
                                loss = binary_cross_entopy, metrics = [rmse])
        
        """
        self.train_IdealBinaryMask
        self.test_IdealBinaryMask
        
        훈련 데이터와 훈련 라벨을 이용해서, 훈련 마스크를 만듬. --> self.train_IdealBinaryMask
        1. 훈련 데이터를 이용해서 신경망에 넣고 output을 만듬 sigmoid(outputs) == estimated mask
        2. estimated mask와 IBM Label로 cross entorpy 학습
        3. 테스트 데이터를 모델에 넣엇 estimated mask를 뽑아내고, 
        """
        model.fit(x = self.trainSound,
                y = self.train_IdealBinaryMask,
                batch_size = self.batch_size,
                epochs = self.epochs,
                verbose = 1,
                callbacks = None,
                shuffle = True,
                validation_data = (self.testSound, self.test_IdealBinaryMask))
        
        return model
        
if __name__ == '__main__':
    test = enhancement()
    test.train()

