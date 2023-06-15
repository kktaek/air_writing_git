from tensorflow.keras.layers import *
from tensorflow.keras import Input, Model
from tensorflow.keras import Sequential
#import tensorflow.python.keras.layers.merge as kmerge

def get_model_tree_v3(len,channel, n_class):
    input_xy = Input(shape=(len,channel))
    #model_pres = BatchNormalization() (input_pres)
    model_xy1_0 = Conv1D(4, kernel_size=1, padding='same', activation='relu')(input_xy)
    model_xy1_1 = Conv1D(4, kernel_size=3, padding='same', activation='relu')(input_xy)
    model_xy1_2 = Conv1D(4, kernel_size=5, padding='same', activation='relu')(input_xy)
    model_xy1_3 = Conv1D(4, kernel_size=7, padding='same', activation='relu')(input_xy)
    model_c1 = concatenate([model_xy1_0, model_xy1_1, model_xy1_2,model_xy1_3])
    model_c1 = MaxPooling1D(pool_size=3, strides=3)(model_c1)

    model_xy2_0 = Conv1D(8, kernel_size=1, padding='same', activation='relu')(model_c1)
    model_xy2_1 = Conv1D(8, kernel_size=3, padding='same', activation='relu')(model_c1)
    model_xy2_2 = Conv1D(8, kernel_size=5, padding='same', activation='relu')(model_c1)
    model_xy2_3 = Conv1D(8, kernel_size=7, padding='same', activation='relu')(model_c1)
    model_c2 = concatenate([model_xy2_0, model_xy2_1, model_xy2_2, model_xy2_3])
    model_c2 = MaxPooling1D(pool_size=3, strides=3)(model_c2)

    model_xy3_0 = Conv1D(16, kernel_size=1, padding='same', activation='relu')(model_c2)
    model_xy3_1 = Conv1D(16, kernel_size=3, padding='same', activation='relu')(model_c2)
    model_xy3_2 = Conv1D(16, kernel_size=5, padding='same', activation='relu')(model_c2)
    model_xy3_3 = Conv1D(16, kernel_size=7, padding='same', activation='relu')(model_c2)
    model_c3 = concatenate([model_xy3_0, model_xy3_1, model_xy3_2, model_xy3_3])
    model_c3 = MaxPooling1D(pool_size=2, strides=2)(model_c3)

    model_xy = Flatten()(model_c3)
    model_xy = Dropout(0.3)(model_xy)
    model_xy = Dense(units=10, activation='relu')(model_xy)
    model_xy = Dropout(0.3)(model_xy)
    model_xy = Dense(units=n_class, activation='softmax')(model_xy)
    model_xy = Model(inputs=input_xy, outputs=model_xy)

    return model_xy

def get_model_tree_v2(len,channel, n_class):
    input_xy = Input(shape=(len,channel))
    #model_pres = BatchNormalization() (input_pres)
    model_xy1_0 = Conv1D(4, kernel_size=1, padding='same', activation='relu')(input_xy)
    model_xy1_1 = Conv1D(4, kernel_size=3, padding='same', activation='relu')(input_xy)
    model_xy1_2 = Conv1D(4, kernel_size=5, padding='same', activation='relu')(input_xy)
    model_xy1_3 = Conv1D(4, kernel_size=7, padding='same', activation='relu')(input_xy)
    model_c1 = concatenate([model_xy1_0, model_xy1_1, model_xy1_2,model_xy1_3])
    model_c1 = MaxPooling1D(pool_size=3, strides=3)(model_c1)

    model_xy2_0 = Conv1D(8, kernel_size=1, padding='same', activation='relu')(model_c1)
    model_xy2_1 = Conv1D(8, kernel_size=3, padding='same', activation='relu')(model_c1)
    model_xy2_2 = Conv1D(8, kernel_size=5, padding='same', activation='relu')(model_c1)
    model_xy2_3 = Conv1D(8, kernel_size=7, padding='same', activation='relu')(model_c1)
    model_c2 = concatenate([model_xy2_0, model_xy2_1, model_xy2_2, model_xy2_3])
    model_c2 = MaxPooling1D(pool_size=3, strides=3)(model_c2)

    model_xy3_0 = Conv1D(16, kernel_size=1, padding='same', activation='relu')(model_c2)
    model_xy3_1 = Conv1D(16, kernel_size=3, padding='same', activation='relu')(model_c2)
    model_xy3_2 = Conv1D(16, kernel_size=5, padding='same', activation='relu')(model_c2)
    model_xy3_3 = Conv1D(16, kernel_size=7, padding='same', activation='relu')(model_c2)
    model_c3 = concatenate([model_xy3_0, model_xy3_1, model_xy3_2, model_xy3_3])
    model_c3 = MaxPooling1D(pool_size=3, strides=3)(model_c3)

    model_xy4_0 = Conv1D(32, kernel_size=1, padding='same', activation='relu')(model_c3)
    model_xy4_1 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(model_c3)
    model_xy4_2 = Conv1D(32, kernel_size=5, padding='same', activation='relu')(model_c3)
    model_xy4_3 = Conv1D(32, kernel_size=7, padding='same', activation='relu')(model_c3)
    model_c4 = concatenate([model_xy4_0, model_xy4_1, model_xy4_2, model_xy4_3])
    model_c4 = MaxPooling1D(pool_size=2, strides=2)(model_c4)
    
    model_xy5_0 = Conv1D(64, kernel_size=1, padding='same', activation='relu')(model_c4)
    model_xy5_1 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(model_c4)
    model_xy5_2 = Conv1D(64, kernel_size=5, padding='same', activation='relu')(model_c4)
    model_xy5_3 = Conv1D(64, kernel_size=7, padding='same', activation='relu')(model_c4)
    model_c5 = concatenate([model_xy5_0, model_xy5_1, model_xy5_2, model_xy5_3])
    model_c5 = MaxPooling1D(pool_size=2, strides=2)(model_c5)

    model_xy = Flatten()(model_c5)
    model_xy = Dropout(0.5)(model_xy)
    model_xy = Dense(units=40, activation='relu')(model_xy)
    model_xy = Dropout(0.5)(model_xy)
    model_xy = Dense(units=n_class, activation='softmax')(model_xy)
    model_xy = Model(inputs=input_xy, outputs=model_xy)

    return model_xy

def get_model_tree(len,channel, n_class):
    input_xy = Input(shape=(len,channel))
    #model_pres = BatchNormalization() (input_pres)
    model_xy1_0 = Conv1D(4, kernel_size=1, padding='same', activation='relu')(input_xy)
    model_xy1_1 = Conv1D(4, kernel_size=3, padding='same', activation='relu')(input_xy)
    model_xy1_2 = Conv1D(4, kernel_size=5, padding='same', activation='relu')(input_xy)
    model_xy1_3 = Conv1D(4, kernel_size=7, padding='same', activation='relu')(input_xy)
    model_c1 = concatenate([model_xy1_0, model_xy1_1, model_xy1_2,model_xy1_3])
    model_c1 = MaxPooling1D(pool_size=3, strides=3)(model_c1)

    model_xy2_0 = Conv1D(8, kernel_size=1, padding='same', activation='relu')(model_c1)
    model_xy2_1 = Conv1D(8, kernel_size=3, padding='same', activation='relu')(model_c1)
    model_xy2_2 = Conv1D(8, kernel_size=5, padding='same', activation='relu')(model_c1)
    model_xy2_3 = Conv1D(8, kernel_size=7, padding='same', activation='relu')(model_c1)
    model_c2 = concatenate([model_xy2_0, model_xy2_1, model_xy2_2, model_xy2_3])
    model_c2 = MaxPooling1D(pool_size=3, strides=3)(model_c2)

    model_xy3_0 = Conv1D(16, kernel_size=1, padding='same', activation='relu')(model_c2)
    model_xy3_1 = Conv1D(16, kernel_size=3, padding='same', activation='relu')(model_c2)
    model_xy3_2 = Conv1D(16, kernel_size=5, padding='same', activation='relu')(model_c2)
    model_xy3_3 = Conv1D(16, kernel_size=7, padding='same', activation='relu')(model_c2)
    model_c3 = concatenate([model_xy3_0, model_xy3_1, model_xy3_2, model_xy3_3])
    model_c3 = MaxPooling1D(pool_size=3, strides=3)(model_c3)

#    model_xy4_0 = Conv1D(64, kernel_size=1, padding='same', activation='relu')(model_c3)
#    model_xy4_1 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(model_c3)
#    model_xy4_2 = Conv1D(64, kernel_size=5, padding='same', activation='relu')(model_c3)
#    model_xy4_3 = Conv1D(64, kernel_size=7, padding='same', activation='relu')(model_c3)
#    model_c4 = concatenate([model_xy4_0, model_xy4_1, model_xy4_2, model_xy4_3])
#    model_c4 = MaxPooling1D(pool_size=3, strides=3)(model_c4)

#    model_xy5_0 = Conv1D(32, kernel_size=1, padding='same', activation='relu')(model_c4)
#    model_xy5_1 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(model_c4)
#    model_xy5_2 = Conv1D(32, kernel_size=5, padding='same', activation='relu')(model_c4)
#    model_xy5_3 = Conv1D(32, kernel_size=7, padding='same', activation='relu')(model_c4)
#    model_c5 = concatenate([model_xy5_0, model_xy5_1, model_xy5_2, model_xy5_3])
#    model_c5 = MaxPooling1D(pool_size=2, strides=2)(model_c5)


#    model_xy = Bidirectional(LSTM(128))(model_c3)
    model_xy = Flatten()(model_c3)
    model_xy = Dropout(0.4)(model_xy)
    model_xy = Dense(units=10, activation='relu')(model_xy)
    model_xy = Dropout(0.4)(model_xy)
    model_xy = Dense(units=n_class, activation='softmax')(model_xy)
    model_xy = Model(inputs=input_xy, outputs=model_xy)

    return model_xy

def get_model_seq(n_class):
    model = Sequential()
    model.add(Conv1D(8, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2,padding='same'))

    # model.add(BatchNormalization())
    model.add(Conv1D(8, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2,padding='same'))

    # model.add(BatchNormalization())
    model.add(Conv1D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2,padding='same'))

    # model.add(BatchNormalization())
    model.add(Conv1D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))


    #model.add(Bidirectional(LSTM(256, return_sequences= True)))
    #model.add(Bidirectional(LSTM(512)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units=70, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=n_class, activation='softmax'))

    return model

def get_model_simple(n_class):
    model = Sequential()
    model.add(Conv1D(8, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(8, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))

    model.add(Bidirectional(LSTM(128)))
    model.add(Flatten())
    model.add(Dense(units=n_class, activation='softmax'))

    return model