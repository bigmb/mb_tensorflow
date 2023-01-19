import yaml
from os import path

def load_config_file(config_filepath, params_class, logger=None):
    """Loads the config file and prepares necessary paths.

    Parameters
    ----------
    config_filepath : str
        local path to the config file containing all the parameters for training
    params_class : object
        the class that the parameters of the config file will be loaded into - Trianing type 
    logger : logging.Logger or equivalent
        logger for logging messages

    Returns
    -------
    params : object
        all the parameters for creating and training a model. The returning object
        is an instance of the class specified by argument `params_class`
    output_dirpath : str
        path for the output root folder
    log_dirpath : str
        path for the output log folder
    checkpoint_dirpath : str
        path for output checkpoint folder
    """

    import yaml

    # determine the root path as dirname of config_filepath
    if not path.exists(config_filepath):
        raise IOError("Config file '{}' does not exists.".format(config_filepath))

    params = yaml.load(open(config_filepath).read(), Loader=yaml.Loader)

    if not isinstance(params, params_class):
        if not isinstance(params, BaseTrainingParams):
            raise ValueError(
                "The class of the parameters is '{}', not the expected '{}'.".format(
                    type(params), params_class
                )
            )
        else:
            if logger:
                logger.warn(
                    "The class of the parameters is '{}', not the expected '{}'.".format(
                        type(params), params_class
                    )
                )

    # assume there is a member function verify() to verify the params
    if hasattr(params, "verify"):
        params.verify(logger=logger)

    # make sure relevant folders exist
    if hasattr(params, "stage_name"):
        variant = params.stage_name
    else:
        variant = params.variant
    output_dirpath = path.join(params.wrangling_params.wrangled_dirpath(), variant)
    log_dirpath = path.join(output_dirpath, "logs")
    checkpoint_dirpath = path.join(output_dirpath, "checkpoints")
    path.make_dirs(log_dirpath)
    path.make_dirs(checkpoint_dirpath)

    return params, output_dirpath, log_dirpath, checkpoint_dirpath

