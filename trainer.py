import cv2
import numpy as np
import glob
import random
import os

import tensorflow as tf

class Model:

    def softmax_regression(self, input_dim, output_dim):
        # Model Design
        self.inputs = tf.placeholder(tf.float32, [None, input_dim])
        weights = tf.Variable(tf.zeros([input_dim, output_dim]))
        biases = tf.Variable(tf.zeros([output_dim]))
        outputs = tf.nn.softmax(tf.matmul(self.inputs, weights) + biases)

        # Fitness and Training Design
        self.observations = tf.placeholder(tf.float32, [None, output_dim])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.observations * tf.log(outputs), reduction_indices=[1]))
        return tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    def softmax_testing(self):
        correct_prediction = tf.equal(tf.argmax(self.inputs, 1), tf.argmax(self.observations, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def __init__(self, name, images, labels):

        self.inputs, self.observations = None, None

        # Model creation
        self.model = self.softmax_regression(images[0].shape[0], labels[0].shape[0])
        self.testing_model = self.softmax_testing()

        # Initialization
        init = tf.initialize_all_variables()

        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(init)

        self.images = images
        self.labels = labels
        self.name = name

    def train(self, steps=1000, batch_size=100):
        # Training
        for i in range(steps):
            batch_indices = np.random.randint(0, self.images.shape[0], batch_size)
            batch_xs = self.images[batch_indices]
            batch_ys = self.labels[batch_indices]
            self.session.run(self.model, feed_dict={self.inputs: batch_xs, self.observations: batch_ys})

    def save(self):
        return self.saver.save(self.session, self.name)

    def test(self):
        # Testing
        print(self.session.run(self.testing_model,
                               feed_dict={self.inputs: self.images, self.observations: self.labels}))

print('Loading training data...')

training_directories = os.listdir("training")

models = {}

for directory in training_directories:
    model_names = directory.split('.')
    models[".".join(model_names[0:1])] = []

for model in models.keys():
    models[model] = glob.glob('training/'+model+'.*/*.npz')

# load training data

for model, training_data in models.items():
    random.shuffle(training_data)

    steering_labels = None
    session_labels = None
    acceleration_labels = None

    num_images = len(training_data)

    e0 = cv2.getTickCount()
    for index, single_npz in enumerate(training_data[:num_images]):
        with np.load(single_npz) as data:
            print(float(index)/num_images, data['arr_0'], data['arr_1'])
            if index == 0:
                image_array = data['arr_0'].astype('float')
                steering_labels = data['arr_1'][:2].astype('float')
                acceleration_labels = data['arr_1'][2:4].astype('float')
                session_labels = data['arr_1'][4].astype('float')
            else:
                image_array = np.vstack((image_array, data['arr_0']))
                steering_labels = np.vstack((steering_labels, data['arr_1'][:2]))
                acceleration_labels = np.vstack((acceleration_labels, data['arr_1'][2:4]))
                session_labels = np.vstack((session_labels, data['arr_1'][4]))

    time0 = (cv2.getTickCount() - e0)/ cv2.getTickFrequency()
    print('Loading image duration:', time0)

    # set start time
    label_sets = {"steering" : steering_labels, "acceleration" : acceleration_labels, "session": session_labels}
    for name, label_set in label_sets.items():
        model = Model(name, image_array, label_set)
        model.train()
        model.test()
        model.save()







