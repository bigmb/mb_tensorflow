##tensorflow Dataloader for training and testing

import tensorflow as tf
import numpy as np
import os
from mb_utils.src.logging import Logger

class Dataloader(object):
    """
    Dataloader for training and testing
    Input:
        data: numpy array - data to be loaded
        batch_size: int - batch size
        shuffle: bool - whether to shuffle the data
        num_threads: int - number of threads to be loaded
        capacity: int - capacity of the queue
        min_after_dequeue: int - minimum number of elements in the queue after dequeue
        seed: int - seed for random shuffle
    Output:
        batch_data: tensor - batch data tensor
    """
    def __init__(self, data, batch_size, shuffle=True, num_threads=4, capacity=1000, min_after_dequeue=100, seed=0):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.seed = seed
    
    def get_batch(self):
        """
        Dataloader for training and testing
        Input:
            None
        Output:
            batch_data: tensor - batch data tensor
        """
        if self.shuffle:
            data = tf.train.shuffle_batch([self.data], batch_size=self.batch_size, num_threads=self.num_threads, 
                                            capacity=self.capacity, min_after_dequeue=self.min_after_dequeue, seed=self.seed)
        else:
            data = tf.train.batch([self.data], batch_size=self.batch_size, num_threads=self.num_threads, 
                                            capacity=self.capacity)
        return data
    
    def get_batch_with_label(self, label):
        """
        Dataloader for training and testing
        Input:
            label: numpy array - label of the data
        Output:
            batch_data: tensor - batch data tensor
        """
        if self.shuffle:
            data, label = tf.train.shuffle_batch([self.data, label], batch_size=self.batch_size, num_threads=self.num_threads, 
                                            capacity=self.capacity, min_after_dequeue=self.min_after_dequeue, seed=self.seed)
        else:
            data, label = tf.train.batch([self.data, label], batch_size=self.batch_size, num_threads=self.num_threads, 
                                            capacity=self.capacity)
        return data, label
    
    @staticmethod
    def augment(self, data, label):
        """
        Augment the data
        Input:
            data: tensor - data to be augmented
            label: tensor - label of the data
        Output:
            data: tensor - augmented data
            label: tensor - augmented label
        """
        data = tf.image.random_flip_left_right(data)
        data = tf.image.random_flip_up_down(data)
        data = tf.image.random_brightness(data, max_delta=0.2)
        data = tf.image.random_contrast(data, lower=0.8, upper=1.2)
        data = tf.image.random_hue(data, max_delta=0.2)
        data = tf.image.random_saturation(data, lower=0.8, upper=1.2)
        return data, label
