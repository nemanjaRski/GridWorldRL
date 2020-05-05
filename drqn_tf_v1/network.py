import tf_slim as slim
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class Qnetwork:
    def __init__(self, final_layer_size, rnn_cell, scope_name, num_actions, state_shape, learning_rate):
        """Inputs"""
        self.image_in = tf.placeholder(shape=[None, *state_shape], dtype=tf.float32)
        self.action_in = tf.placeholder(shape=[None], dtype=tf.int32)
        self.action_in_one_hot = tf.one_hot(self.action_in, num_actions, dtype=tf.float32)

        """Layers"""
        self.conv1 = slim.convolution2d(
            inputs=self.image_in, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None, scope=scope_name + '_conv1')
        self.conv2 = slim.convolution2d(
            inputs=self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None, scope=scope_name + '_conv2')
        self.conv3 = slim.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None, scope=scope_name + '_conv3')
        self.conv4 = slim.convolution2d(
            inputs=self.conv3, num_outputs=final_layer_size,
            kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None, scope=scope_name + '_conv4')

        self.fc1 = slim.fully_connected(inputs=self.action_in_one_hot, num_outputs=64, biases_initializer=None,
                                        activation_fn=tf.nn.relu)

        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.fc1 = tf.reshape(self.fc1, [self.batch_size, self.train_length, 64])
        self.conv_flat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.train_length, final_layer_size])

        self.rnn_in = tf.concat([self.conv_flat, self.fc1], -1)

        self.rnn_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.rnn_in, cell=rnn_cell, dtype=tf.float32, initial_state=self.rnn_state_in,
            scope=scope_name + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, final_layer_size])
        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.stream_advantage, self.stream_value = tf.split(self.rnn, 2, 1)
        self.advantage_weights = tf.Variable(tf.random_normal([final_layer_size // 2, num_actions]))
        self.value_weights = tf.Variable(tf.random_normal([final_layer_size // 2, 1]))
        self.advantage = tf.matmul(self.stream_advantage, self.advantage_weights)
        self.value = tf.matmul(self.stream_value, self.value_weights)

        # Then combine them together to get our final Q-values.
        self.q_out = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.q_out, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_one_hot), axis=1)

        self.td_error = tf.square(self.target_q - self.q)

        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.mask_a = tf.zeros([self.batch_size, self.train_length // 2])
        self.mask_b = tf.ones([self.batch_size, self.train_length // 2])
        self.mask = tf.concat([self.mask_a, self.mask_b], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_model = self.trainer.minimize(self.loss)
