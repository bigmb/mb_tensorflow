import yaml
import os

__all__ = ['ModelParams','BaseParams']

class ModelParams(yaml.YAMLObject):
    """Parameters for defining and creating a model.
    This is an abstract class. The user should subclass from this class to define their own class
    which represents the collection of parameters to create models of a given family.
    
    Params:
        gen : int
            model generation/family number, starting from 1
    """

    yaml_tag = "!ModelParams"

    def __init__(self, gen: int = 1):
        self.gen = gen


class BaseParams(yaml.YAMLObject):
    """Basic parameters for wrangling the data for training models.

    Params:
        data_localpath : path
            path to the data folder
        filename : str
            file name of the final data to be trained. Default is 'training_data.csv'.
    """
    yaml_tag = "!BaseParams"

    def __init__(
        self,
        data_localpath: str = "./data",
        filename: str = "training_data.csv",
    ):

        self.data_localpath = data_localpath
        self.ilename = filename

    def prepared_dirpath(self):
        dirpath = os.path.abspath(self.data_localpath)
        os.path.make_dirs(dirpath)
        return dirpath
