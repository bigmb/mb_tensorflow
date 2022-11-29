#function for tf data and tf version check

import tensorflow as tf
from mb_utils.src.logging import logger
import yaml
from ..params.tf_data_params import Tf_Data_Params

class tf_data(Tf_Data_Params):
    
    def __init__(self):
        super().__init__()

    def get_data(self,get_all_tf_data):
        """
        Function to get the data of tf
        :param get_all_tf_data: yaml file
        :return: data
        """
        if Tf_Data_Params.get_all_tf_data:
            Tf_Data_Params.tf_version(logger)
            Tf_Data_Params.tf_location(logger)
            Tf_Data_Params.tf_eager(logger)
            Tf_Data_Params.tf_gpu(logger)
            #Tf_Data_Params.tf_disable_eager(logger)
        else:
            return None


