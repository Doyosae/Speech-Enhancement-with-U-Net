import numpy as np
import matplotlib.pyplot as plt

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
    
    def __init__ (self, trainSize = 720, testSize = 210, inputSize = 256):
        
        callData = dataProcessing()
        self.trainLabel, self.testLabel, self.trainSound, self.testSound = callData.stft()

        self.trainSize = trainSize
        self.testSize = testSize
        self.inputSize = inputSize
        
        self.filter1 = 32
        self.filter2 = 64
        self.filter3 = 128
        self.filter4 = 256
        self.filter5 = 512
        self.filter6 = 1024
        
        self.kernel_size = (3, 3)
        self.strides = (2, 2)
        
        self.paddingValid = "valid"
        self.paddingSame = "same"
        
        self.batch_size = 60
        self.epochs = 300
        
        self.learning_rate = 0.00015
        self.beta_1 = 0.5
        

        for index in range (self.trainSize):
            self.trainLabel[index] = self.trainLabel[index][:self.inputSize, :self.inputSize]
            self.trainSound[index] = self.trainSound[index][:self.inputSize, :self.inputSize]
        for index in range (self.testSize):
            self.testLabel[index] = self.testLabel[index][:self.inputSize, :self.inputSize]
            self.testSound[index] = self.testSound[index][:self.inputSize, :self.inputSize]
            
        self.trainLabel = np.reshape(self.trainLabel, (self.trainSize, self.inputSize, self.inputSize, 1))
        self.testLabel  = np.reshape(self.testLabel,  (self.testSize, self.inputSize, self.inputSize, 1))
        self.trainSound = np.reshape(self.trainSound, (self.trainSize, self.inputSize, self.inputSize, 1))
        self.testSound  = np.reshape(self.testSound,  (self.testSize, self.inputSize, self.inputSize, 1))

        self.trainLabel = self.trainLabel.real
        self.testLabel  = self.testLabel.real
        self.trainSound = self.trainSound.real
        self.testSound  = self.testSound.real

        print("\n")
        print("trainLabel resize  : ", np.shape(self.trainLabel))
        print("testLabel resize   : ", np.shape(self.testLabel))
        print("trainSound resize  : ", np.shape(self.trainSound))
        print("testSound resize   : ", np.shape(self.testSound))

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
        self.paddingValid = "valid"
        self.paddingSame = "same"
        """
        inputs = Input(shape = (self.inputSize, self.inputSize, 1))
        output = Conv2D(self.filter1, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.paddingSame,
                     activation = None)(inputs)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2D(self.filter2, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.paddingSame,
                     activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2D(self.filter3, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.paddingSame,
                     activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2D(self.filter4, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.paddingSame,
                     activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2D(self.filter5, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.paddingSame,
                     activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2D(self.filter6, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides,
                     padding = self.paddingSame,
                     activation = None)(output)
        output = tf.nn.leaky_relu(output)

        output = Flatten()(output)
        output = Dense(2048, activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Dense(512, activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Dense(2048, activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Dense(16384, activation = None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = tf.reshape(output, (-1, 4, 4, 1024))

        output = Conv2DTranspose(self.filter5, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.paddingSame,
                     activation=None)(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2DTranspose(self.filter4, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.paddingSame,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2DTranspose(self.filter3, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.paddingSame,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2DTranspose(self.filter2, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.paddingSame,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2DTranspose(self.filter1, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.paddingSame,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)
        output = Conv2DTranspose(1, 
                     kernel_size = self.kernel_size, 
                     strides = self.strides, 
                     padding = self.paddingSame,
                     activation=None)(output)
        output = BatchNormalization()(output)
        output = tf.nn.leaky_relu(output)

        model = Model (inputs = inputs, outputs = output)
        model.summary()

        """
        self.batch_size = 60
        self.epochs = 300
        self.learning_rate = 0.00015
        self.beta_1 = 0.5
        """
        return model
    
    def train(self):
        
        def rmse(y_true, y_pred):
            output = tf.sqrt (2 * tf.nn.l2_loss(y_true - y_pred)) / self.batch_size
            return output
    
        model = self.model()
        model.compile(optimizer = Adam(lr = self.learning_rate, 
                                            beta_1 = self.beta_1, 
                                            beta_2 = 0.999, 
                                            epsilon = None, 
                                            decay = 0.0, 
                                            amsgrad = False),
                                            loss = 'mean_squared_error', metrics = [rmse])
        model.fit(x = self.trainSound,
                    y = self.trainLabel,
                    batch_size = self.batch_size,
                    epochs = self.epochs,
                    verbose = 1,
                    callbacks = None,
                    shuffle = True,
                    validation_data = (self.testSound, self.testLabel))
        
        return model
    
if __name__ == '__main__':
    M = enhancement()
    M.train()
