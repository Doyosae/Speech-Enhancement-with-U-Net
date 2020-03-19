import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K

from utility import *
from dataload import *
from processing import *
import segmentation_models as sm

class TrainUnet ():
    
    def __init__ (self):
        
        # Define constant
        # Data Information
        self.train_length = 660
        self.test_length = 60
        
        # Ideal Binary Mask constant
        self.criteria = 0.8
        self.alpha = 1.0
        
        # Hyper parameter
        self.batch_size = 30
        self.epochs = 50
        self.learning_rate = 0.0001
        self.decay_prop = 0.9
        
        self.filter_size4 = 4
        self.filter_size8 = 8
        self.filter_size16 = 16
        self.filter_size32 = 32
        self.filter_size64 = 64
        self.filter_size128 = 128
        self.filter_size256 = 256
        self.filter_size512 = 512
        
        # Call data
        self.train_data, self.train_label, self.test_data, self.test_label = self.return_data()
        
        self.image_width = np.shape(self.train_data)[1]
        self.image_height = np.shape(self.train_data)[2]
        
        # returned Ideal Binary Mask
        self.train_IBM, self.test_IBM = self.return_IBM()
        
    def return_data (self):
        
        data = dataProcessing(n_fft = 512, n_mfcc = 40, hop_length = 0.5 * 512)
        train_label, test_label, train_data, test_data = data.stft()

        image_width = np.shape(train_label)[1]-1
        image_height = np.shape(train_label)[2]-9

        # 순서 변경
        # train label, test label, train data, test data
        # train data, train label, test data, test label
        
        train_data = get_realData(train_data, self.train_length, image_width, image_height)
        train_label = get_realData(train_label, self.train_length, image_width, image_height)
        test_data  = get_realData(test_data, self.test_length, image_width, image_height)
        test_label  = get_realData(test_label, self.test_length, image_width, image_height)
        
        print("")
        
        # Return type is turple
        return train_data, train_label, test_data, test_label
    
    def return_IBM (self):
        
        # 깨끗한 음성은 훈련 라벨, 더러운 음성은 훈련 데이터, 그 길이는 660
        train_IdealBinaryMask = createIdealBinaryMask (self.train_label, 
                           self.train_data, 
                           data_length = self.train_length, 
                           image_size1 =  self.image_width,
                           image_size2 = self.image_height,
                           alpha = self.alpha,
                           criteria = self.criteria)
        # 깨끗한 음성은 검증 라벨, 더러운 음성은 검증 데이터, 그 길이는 60
        test_IdealBinaryMask = createIdealBinaryMask (self.test_label, 
                           self.test_data, 
                           data_length = self.test_length, 
                           image_size1 =  self.image_width,
                           image_size2 = self.image_height,
                           alpha = self.alpha,
                           criteria = self.criteria)
        print(np.shape(train_IdealBinaryMask), np.shape(test_IdealBinaryMask))
                       
        # Return IBM
        return train_IdealBinaryMask, test_IdealBinaryMask
    
    def model(self, input_shape):
        
        inputs = Input(input_shape)

        c1 = Conv2D(self.filter_size32, (3, 3), activation='relu', padding='same') (inputs)
        c1 = Conv2D(self.filter_size32, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2), padding='same') (c1)

        c2 = Conv2D(self.filter_size64, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(self.filter_size64, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2), padding='same') (c2)

        c3 = Conv2D(self.filter_size128, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(self.filter_size128, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2), padding='same') (c3)

        c4 = Conv2D(self.filter_size256, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(self.filter_size256, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D((2, 2), padding='same') (c4)

        c5 = Conv2D(self.filter_size512, (3, 3), activation='relu', padding='same') (p4)
        c5 = Conv2D(self.filter_size512, (3, 3), activation='relu', padding='same') (c5)
        p5 = MaxPooling2D((2, 2), padding='same') (c5)

        c55 = Conv2D(self.filter_size256, (3, 3), activation='relu', padding='same') (p5)
        c55 = Conv2D(self.filter_size256, (3, 3), activation='relu', padding='same') (c55)

        u6 = Conv2DTranspose(self.filter_size256, (2, 2), strides=(2, 2), padding='same') (c55)
        u6 = concatenate([u6, c5])
        c6 = Conv2D(self.filter_size256, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(self.filter_size256, (3, 3), activation='relu', padding='same') (c6)

        u71 = Conv2DTranspose(self.filter_size128, (2, 2), strides=(2, 2), padding='same') (c6)
        u71 = concatenate([u71, c4])
        c71 = Conv2D(self.filter_size128, (3, 3), activation='relu', padding='same') (u71)
        c61 = Conv2D(self.filter_size128, (3, 3), activation='relu', padding='same') (c71)

        u7 = Conv2DTranspose(self.filter_size128, (2, 2), strides=(2, 2), padding='same') (c61)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(self.filter_size128, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(self.filter_size128, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(self.filter_size64, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(self.filter_size64, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(self.filter_size64, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(self.filter_size32, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(self.filter_size32, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(self.filter_size32, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(self.filter_size16, (1, 1), activation='relu') (c9)
        outputs = Conv2D(self.filter_size4, (1, 1), activation='relu') (outputs)
        outputs = Conv2D(1, (1, 1), activation='sigmoid') (outputs)

        model = Model(inputs = [inputs], outputs = [outputs])

        return model
    
    def train(self):
        
        def dice_loss(y_true, y_pred):
            smooth = 1.
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = y_true_f * y_pred_f
            score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
            return 1. - score

        def bce_dice_loss(y_true, y_pred):
            return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
        
        model = self.model((super().self.image_width, self.image_height, 1))
        model.compile(optimizer = Adam(lr = self.learning_rate, 
                        beta_1 = self.decay_prop, 
                        beta_2 = 0.999, 
                        epsilon = None, 
                        decay = 0.0, 
                        amsgrad = False),
                        loss = "mean_squared_error")
        model.summary()
        
        """
        self.train_data, self.train_label, self.test_data, self.test_label
        self.train_IBM, self.test_IBM
        """
        model.fit(x = self.train_data,
                y = self.train_IBM,
                batch_size = self.batch_size,
                epochs = self.epochs,
                verbose = 1,
                callbacks = None,
                shuffle = True,
                validation_data = (self.test_data, self.test_IBM))
        
        model.save("Unet_mse_loss.h5")
        
if __name__ == '__main__':
    test = TrainUnet().train()