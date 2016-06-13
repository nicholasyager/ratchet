#!/usr/bin/python2

import cv2
import numpy as np
import glob
import random
import os

class ANNModel(object):

    def __init__(self, name, images, labels):
        self.name = name
        self.images = images
        self.labels = labels

        # Set criteria and parameters
        self.criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
        self.params = dict(term_crit = self.criteria,
                  train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                  bp_dw_scale = 0.001,
                  bp_moment_scale = 0.0 )

        # Create model
        self.layer_sizes = np.int32([self.images.shape[1], 64, self.labels.shape[1]])
        self.model = cv2.ANN_MLP()
        self.model.create(self.layer_sizes)

    def train(self):

        print 'Training '+self.name
        e1 = cv2.getTickCount()
        num_iter = self.model.train(self.images, self.labels, None, params = self.params)
        print 'Ran for %d iterations' % num_iter
        # set end time
        time = (cv2.getTickCount() - e1)/cv2.getTickFrequency()
        print 'Training duration:', time

    def save(self):
        directory = "mpl_xml"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(directory+'/'+self.name+'.xml')

    def test(self):
        ret, resp = self.model.predict(self.images)
        prediction = resp.argmax(-1)
        print 'Prediction:', prediction
        true_labels = self.labels.argmax(-1)
        print 'True labels:', true_labels

        print 'Testing...'
        train_rate = np.mean(prediction == true_labels)
        print 'Train rate: %f:' % (train_rate*100)

print 'Loading training data...'

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

    num_images = len(training_data)

    e0 = cv2.getTickCount()
    for index, single_npz in enumerate(training_data[:num_images]):
        with np.load(single_npz) as data:
            print float(index)/num_images, data['arr_0'], data['arr_1']
            if index == 0:
                image_array = data['arr_0'].astype('float')
                steering_labels = data['arr_1'][:2].astype('float')
                acceleration_labels = data['arr_1'][:2].astype('float')
            else:
                image_array = np.vstack((image_array, data['arr_0']))
                steering_labels = np.vstack((steering_labels, data['arr_1'][:2]))
                acceleration_labels = np.vstack((acceleration_labels, data['arr_1'][2:]))

    time0 = (cv2.getTickCount() - e0)/ cv2.getTickFrequency()
    print 'Loading image duration:', time0

    # set start time
    label_sets = {"steering" : steering_labels, "acceleration" : acceleration_labels}
    for name, label_set in label_sets.items():
        model = ANNModel(model+"."+name, image_array, label_set)
        model.train()
        model.save()
        model.test()
