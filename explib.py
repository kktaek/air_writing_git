
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


import random
def calAccuracies_acc2letters(cm):
    nChars = cm.shape[0]
    prec = np.zeros(nChars,)
    rec = np.zeros(nChars, )
    f1 = np.zeros(nChars,)
    for i in range(nChars):
        nTP = cm[i,i]
        nFP = np.sum(cm[:,i]) - nTP
        nFN = np.sum(cm[i,:]) - nTP
        prec[i] = nTP/ (nTP + nFP)
        rec[i] = nTP / (nTP + nFN)
        f1[i] = 2 * nTP / (2*nTP+ nFP + nFN)
    return prec, rec, f1


def drawHistory(hist):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='lower left')

    plt.show()

def getTargetData(data, bRemoveSameShapeInfo):
    nSubjects = data.shape[0]
    nTrials = data.shape[1]
    nChars = data.shape[2]

    Y = np.zeros([nSubjects, nTrials, nChars])
    for i in range(nChars):
        if bRemoveSameShapeInfo==0:
            Y[:, :, i] = i
        else:
            if i<18:
                Y[:, :, i] = i
            elif i==18:
                Y[:, :, i] = 1
            elif i<21:
                Y[:, :, i] = i-1
            elif i==21:
                Y[:, :, i] = 4
            elif i<24:
                Y[:, :, i] = i-2
            elif i==24:
                Y[:, :, i] = 0
            elif i<29:
                Y[:, :, i] = i-3
            elif i==29:
                Y[:, :, i] = 7
            else:
                Y[:, :, i] = i-4
    return Y

#학습/validation/ test로 분할
def spilt_data_for_exp(data, Y):
    nSubjects = data.shape[0]
    nArrayLen = data.shape[3]
    nChannels = data.shape[4]

    idx = np.array(range(nSubjects))
    print(f'---------------{idx[-1]}')
    random.shuffle(idx)
    nTrain = 19
    nVal = 0
    nTest = 2

    XTrain = data[idx[:nTrain],:,:,:,:]
    #XVal = data[idx[nTrain:nTrain+nVal],:,:,:,:]
    XTest = data[idx[nTrain + nVal:], :, :, :, :]

    YTrain= Y[idx[:nTrain]]
    #YVal = Y[idx[nTrain:nTrain+nVal]]
    YTest = Y[idx[nTrain + nVal:]]

    XTrain = XTrain.reshape([-1,nArrayLen, nChannels])
    #XVal = XVal.reshape([-1, nArrayLen, nChannels])
    XTest = XTest.reshape([-1, nArrayLen, nChannels])
    YTrain = YTrain.reshape([-1, ])
    #YVal = YVal.reshape([-1, ])
    YTest = YTest.reshape([-1, ])

    YTrain = to_categorical(YTrain)
    #YVal = to_categorical(YVal)
    YTest = to_categorical(YTest)

    #return XTrain, XVal, XTest, YTrain, YVal, YTest
    return XTrain,  XTest, YTrain, YTest

'''#학습/validation/ test로 분할
def spilt_data_for_exp_v2(data, Y):
    nSubjects = data.shape[0]
    nArrayLen = data.shape[3]
    nChannels = data.shape[4]

    #idx = np.array(range(nSubjects))
    #random.shuffle(idx)
    nTrain = 19

    idx = np.zeros([nSubjects,])
    idx[7] = 0

    XTrain = data[idx==1,:,:,:,:]
    XTest = data[7, :, :, :, :]

    YTrain= Y[idx==1,:]
    YTest = Y[7,:]

    XTrain = XTrain.reshape([-1,nArrayLen, nChannels])
    XTest = XTest.reshape([-1, nArrayLen, nChannels])
    YTrain = YTrain.reshape([-1, ])
    YTest = YTest.reshape([-1, ])

    YTrain = to_categorical(YTrain)
    YTest = to_categorical(YTest)

    return XTrain,  XTest, YTrain, YTest'''


#학습/validation/ test로 분할 (Leave one subject out)
def spilt_data_for_exp_Nfold(data, Y, idx_test):
    nSubjects = data.shape[0]
    nArrayLen = data.shape[3]
    nChannels = data.shape[4]

    bTrain = np.ones(nSubjects, )
    bTrain[idx_test] = 0

    idx = np.array(range(nSubjects))

    XTrain = data[bTrain==1,:,:,:,:]
    XTest = data[idx_test, :, :, :, :]

    YTrain= Y[bTrain==1]
    YTest = Y[idx_test]

    XTrain = XTrain.reshape([-1,nArrayLen, nChannels])
    XTest = XTest.reshape([-1, nArrayLen, nChannels])
    YTrain = YTrain.reshape([-1, ])
    YTest = YTest.reshape([-1, ])

    YTrain = to_categorical(YTrain)
    YTest = to_categorical(YTest)

    return XTrain,  XTest, YTrain, YTest
