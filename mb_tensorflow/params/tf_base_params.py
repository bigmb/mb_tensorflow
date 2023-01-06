#file to read yaml file
import yaml
from os import path


__all__ = ["Tf_Base_Data_Params"]

class Tf_Base_Data_Params(yaml.YAMLObject):
    yaml_tag = "!Tf_Data_Params"

    """Basic parameters for wrangling the data for training models.

    Parameters
    ----------
    data_localpath : path
        path to the data folder
    wrangled_filename : str
        file name of the wrangled table. Default is 'wrangled.csv'.
    """

    def __init__(
        self,
        data_localpath: str = "./data",
        wrangled_filename: str = "wrangled.csv"):

        self.data_localpath = data_localpath
        self.wrangled_filename = wrangled_filename


    def prepared_dirpath(self):
        """
        Path to the directory where the prepared data is stored.
        """
        
        dirpath = path.abspath(self.data_localpath)
        path.make_dirs(dirpath)
        return dirpath

    def wrangled_dirpath(self):
        """
        Path to the directory where the wrangled data is stored. Default is the same as the prepared data.
        """
        return self.prepared_dirpath()  

