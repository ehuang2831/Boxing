import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

im = Image.open("/Users/Evan/Dropbox/Project/Pictures/Training_Non_Boxing/n10039946_9597.JPEG")
arr = np.array(im)

def create_training():

    for filename in glob.glob('yourpath/*.JPEG'):  # assuming gif
        im = Image.open(filename)
        image_list.append(im)


def generate_batch_indices(data, batch_size=100):
    x, y = data
    num_batches = int(x.shape[1]/batch_size)
    indices = np.arrange(num_batches*batch_size)
    return indices


class model:
    def __init__(self, num_layer=1, num_neuron=100):
        self.num_layer = num_layer
        self.num_neuron = num_neuron
        self.parameters = {}

    def layer(self, x, num_neuron, layer_index, last_layer=False):
        input_shape = x.get_shape()
        W = tf.get_variable("weights",
                            [input_shape[-1], num_neuron], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias",
                            [1, num_neuron], initializer=tf.truncated_normal_initializer(stddev=0.1))
        y = tf.matmul(x, W) + b
        self.parameters['W' + str(layer_index)] = W
        self.parameters['b' + str(layer_index)] = b
        return tf.nn.relu(y) if not last_layer else tf.nn.sigmoid(y)

    def construct_graph(self, learning_rate):
        self.x = tf.placeholder(tf.float32, [None, 2])
        h = self.x
        for i in range(self.num_layer-1):
            with tf.variable_scope("layer" + str(i)):
                h = self.layer(h, self.num_neuron, parameters = self.parameters, layer_index=i)
        y = self.layer(h, 4, last_layer=True)

        self.prediction = y
        self.y_ = tf.placeholder(tf.float32, [None, 4])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.prediction, labels = self.y))
        self.train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)

    def train(self, sess, data, num_epochs, print_cost = True):
        tf.global_variables_initializer().run()
        costs = []
        batch_size = 1000
        data_x, data_y = data
        num_batches = int(data_x.shape[1]/batch_size)
        for epoch in range(num_epochs):
            indices = generate_batch_indices(data, batch_size)
            epoch_cost = 0
            for index in range(num_batches):
                batch_index = indices[batch_size * index:batch_size * (index + 1)]
                batch_xs = data_x[:,batch_index]
                batch_ys = data_y[:,batch_index]
                loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                epoch_cost += loss / num_batches
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        plt.plot(costs)
        plt.title('training loss')
        plt.show()
        parameters = sess.run(self.parameters)
        return parameters, costs

    def test(self, sess, t):
        return sess.run(self.prediction, feed_dict={self.x: t})

if __name__ == '__main__':

    data = create_training()
    num_layer = 2
    num_neurons = 100
    model_A = model(num_layer=num_layer, num_neuron=num_neurons)
    tf.reset_default_graph()
    model_A.construct_graph()
    with tf.Session() as sess:
        final_parameters, cost_list = model_A.train(sess, data)
