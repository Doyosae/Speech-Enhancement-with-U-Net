from train import *
from utility import *
from dataload import *
from processing import *

test = enhancement()
model = test.model()
model.load_weights("./save/IBM_ghost.h5")

if __name__ == '__main__':

    # trianLabel, testLabel, trainSound, testSound
    trainLabel, testLabel, trainSound, testSound = test.get_data()

    # Exploit estimated mask
    estimated_train_mask = model.predict(x = trainSound, batch_size = 60 , verbose = 1)
    estimated_test_mask  = model.predict(x = testSound, batch_size = 60, verbose = 1)

    # Calculate estimated speech
    estimated_train_Speech = np.multiply(trainSound, estimated_train_mask)
    estimated_test_Speech  = np.multiply(testSound, estimated_test_mask)
    
    # show Spectogram about train sound -> estimated train speech and
    # test sound (15db) -> estimated test speech
    showSpectogram(trainSound[0], "train sound")
    showSpectogram(estimated_train_Speech[0], "estimated train speech")
    showSpectogram(testSound[0], "test sound")
    showSpectogram(estimated_test_Speech[0], "estimated test speech")
    
    
    # 음원 파일로 저장하기 위하여 채널 1을 없애고, isftft 처리한 후, 저장
    estimated_train_Speech = np.reshape(estimated_train_Speech, (720, 256, 256))
    estimated_test_Speech  = np.reshape(estimated_test_Speech, (210, 256, 256))
    train = librosa.core.istft(estimated_train_Speech[0])
    test  = librosa.core.istft(estimated_test_Speech[0])
    librosa.output.write_wav("./result/estimated_train.wav", train, sr = 16000, norm=False)
    librosa.output.write_wav("./result/estimated_test.wav", test, sr = 16000, norm=False)