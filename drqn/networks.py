import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, LSTM, TimeDistributed, Flatten, Lambda, Convolution2D
from keras.optimizers import Adam

from coords import CoordinateChannel2D

class Networks:
    @staticmethod
    def drqn(sequence_length, state_shape, action_size, num_features, learning_rate):

        width, height, num_channels = state_shape

        input = Input(shape=(sequence_length, width, height, num_channels))

        model = TimeDistributed(CoordinateChannel2D())(input)
        model = TimeDistributed(Conv2D(filters=16, kernel_size=7, strides=4, activation='relu', padding='same', name='conv_1'), name='td_conv_1')(model)
        model = TimeDistributed(Conv2D(filters=32, kernel_size=5, strides=4, activation='relu', padding='same', name='conv_2'), name='td_conv_2')(model)
        model = TimeDistributed(Conv2D(filters=num_features, kernel_size=3, strides=1, activation='relu', padding='same', name='conv_3'), name='td_conv_3')(model)
        model = TimeDistributed(Flatten(name='reshape'))(model)

        model = LSTM(units=num_features, activation="tanh", name='lstm_1')(model)

        advantage = Lambda(lambda x: x[:, :num_features // 2], name="advantage_lambda")(model)
        value = Lambda(lambda x: x[:, num_features // 2:], name="value_lambda")(model)

        advantage = Dense(action_size, name="advantage_dense")(advantage)
        value = Dense(action_size, name="value_dense")(value)

        model = Lambda(lambda x: x[0] + (x[1] - K.mean(x[1] - K.mean(x[1], axis=1, keepdims=True))), name="output")([value, advantage])
        model = Model(input, model, name="DRQN")
        model.compile(optimizer=Adam(learning_rate), loss="mse")

        return model

network = Networks.drqn(8,(84,84,3),4,64,0.0001)
network.summary()
