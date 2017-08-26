import tensorflow as tf
import tensorlayer as tl


class GeneNet(object):
    def __init__(self, input_shape=(15, 4, 3),
                 output_shape=(4,),
                 output_shape1=(4,),
                 kernel_size=(2, 4),
                 kernel_size1=(3, 4),
                 pool_size=(7, 1),
                 pool_size1=(3, 1),
                 filter_num=48,
                 hidden_layer_unit_number=48, learning_rate=0.0005):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_shape1 = output_shape1
        self.output1 = output_shape1
        self.kernel_size = kernel_size
        self.kernel_size1 = kernel_size1
        self.pool_size = pool_size
        self.pool_size1 = pool_size1
        self.filter_num = filter_num
        self.hidden_layer_unit_number = hidden_layer_unit_number
        self.learning_rate = learning_rate
        self.graph = tf.Graph()
        self.create_graph()
        self.session = tf.Session(graph=self.graph)

    def create_graph(self):
        W_init = tf.truncated_normal_initializer(stddev=5e-2)
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)
        with self.graph.as_default():
            x_data = tf.placeholder(tf.float32, shape=[None, self.input_shape[0], self.input_shape[1],
                                                       self.input_shape[2]])
            y = tf.placeholder(tf.float32, shape=[None, self.output_shape[0], self.output_shape1[0]])

            self.x = x_data
            self.y = y

        self.input_layer = tl.layers.InputLayer(x_data, name='data_input')
        self.conv1 = tl.layers.Conv2dLayer(self.input_layer, act=tf.nn.relu, shape=[self.kernel_size, 1,
                                                                                    self.filter_num], padding='SAME',W_init=W_init, b_init=b_init2, name='conv1')
        self.pool1 = tl.layers.MaxPool2d(self.conv1, filter_size=self.pool_size, strides=None,
                                         padding='SAME', name='pool1')
        self.conv2 = tl.layers.Conv2dLayer(self.pool1, act=tf.nn.relu, shape=[self.kernel_size1, 1, self.filter_num],
                                           padding='SAME', W_init=W_init2)
        self.pool2 = tl.layers.MaxPool2d(self.conv2, filter_size=self.pool_size1, strides=None, padding='SAME',
                                         name='pool2')
        self.flat_size = (15 - (self.pool_size[0] - 1) - (self.pool_size1[0] - 1))
        self.flat_size *= (4 - (self.pool_size[1] - 1) - (self.pool_size1[1] - 1))
        self.flat_size *= self.filter_num
        self.conv_flat = tf.reshape(self.pool2, shape=[-1, self.flat_size])
        self.hidden1 = tl.layers.DenseLayer(layer=self.conv_flat, n_units=self.hidden_layer_unit_number, act=tf.nn.elu)
        self.dropout1 = tl.layers.DropoutLayer(layer=self.hidden1, keep=0.50)
        self.hidden2 = tl.layers.DenseLayer(layer=self.dropout1, n_units=self.hidden_layer_unit_number, act=tf.nn.elu)
        self.dropout2 = tl.layers.DropoutLayer(layer=self.hidden2, keep=0.50)
        self.hidden3 = tl.layers.DenseLayer(layer=self.dropout2, n_units=self.hidden_layer_unit_number, act=tf.nn.elu)
        self.dropout3 = tl.layers.DropoutLayer(layer=self.hidden3, keep=0.50)
        y1 = tl.layers.DenseLayer(layer=self.dropout3, n_units=self.output_shape[0], act=tf.nn.sigmoid)
        y2 = tl.layers.DenseLayer(layer=self.dropout3, n_units=self.output_shape1[0], act=tf.nn.elu)
        y3 = tf.nn.sigmoid(y2)
        self.Y1 = y1
        self.Y2 = y3
        cost = tf.reduce_sum(tf.pow(y1 - tf.slice(y, [0, 0], [-1, self.output_shape[0]]), 2)) + \
               tf.reduce_sum(
                   tf.nn.softmax_cross_entropy_with_logits(logits=y2, labels=tf.slice(y, [0, self.output_shape[0]],
                                                                                      [-1, self.output_shape1[0]])))
        self.cost = cost
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_optimizer = self.optimize.minimize(self.cost)
        self.init = tf.global_variables_initializer()

    def initialized(self):
        self.session.run(self.init)

    def close(self):
        self.session.close()

    def train(self, batchX, batchY):
        cost = 0
        X_in = self.x
        Y_out = self.y
        cost, _ = self.session.run((self.cost, self.training_optimizer), feed_dict={X_in: batchY, Y_out: batchY})
        return cost

    def get_lost(self, batchX, batchY):
        cost = 0
        X_in = self.x
        Y_in = self.y
        cost = self.session.run(self.cost, feed_dict={X_in: batchX, Y_in: batchY})
        return cost

    def save_parameters(self, filepath):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.save(self.session, filepath)

    def restore_parameters(self, filepath):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, filepath)

    def predict(self, Xarray):
        with self.graph.as_default():
            base_, type_ = self.session.run((self.Y1, self.Y3), feed_dict={self.x: Xarray})
            return base_, type_
