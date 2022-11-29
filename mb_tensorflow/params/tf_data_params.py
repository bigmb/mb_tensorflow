#file to import yaml data of tf_data_params

import yaml
import tensorflow as tf
import mb_utils.src.logging as logger

__all__ = ['Tf_Data_Params','tf_version', 'tf_location', 'tf_eager','tf_gpu','tf_disable_eager']

class Tf_Data_Params(yaml.YAMLObject):
    """
    Class to check the tf data from yaml tag : !Tf_Data_Params
    """
    yaml_tag = '!Tf_Data_Params'

    def __init__(self):
        self.get_all_tf_data=yaml_tag.get_all_tf_data

    @staticmethod
    def tf_version(logger=None):
        """
        Function to check the tensorflow version
        :return: tensorflow version
        """
        if logger:
            logger.info('Tensorflow version: {}'.format(tf.__version__))
        return tf.__version__
    
    @staticmethod
    def tf_location(logger=None):
        """
        Function to check the tensorflow location
        :return: tensorflow location
        """
        if logger:
            logger.info('Tensorflow location: {}'.format(tf.__file__))
        return tf.__file__
    
    @staticmethod
    def tf_eager(logger=None):
        """
        Function to check the tensorflow eager execution
        :return: tensorflow eager execution
        """
        if logger:
            logger.info('Tensorflow eager execution: {}'.format(tf.executing_eagerly()))

    @staticmethod
    def tf_gpu(logger=None):
        """
        Function to check the tensorflow gpu
        :return: tensorflow gpu
        """
        if logger:
            logger.info('Tensorflow gpu: {}'.format(tf.test.is_gpu_available()))
        return tf.test.is_gpu_available()

    @staticmethod
    def tf_disable_eager(logger=None):
        """
        Function to disable the tensorflow eager execution
        :return: tensorflow eager execution
        """
        if tf.__version__ < '2.0.0':
            tf.compat.v1.disable_eager_execution()
        if logger:
            logger.info('Tensorflow eager execution: {}'.format(tf.executing_eagerly()))
    
