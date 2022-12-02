
from mb_utils.src.logging import logger
from mb_pandas.src.dfload import load_any_df
import os
import numpy as np
from ...tf import data_streams

from ..trainer_class import NFlowParams

__all__ = ['load_data_streams']

def load_data_streams(params: NFlowParams, output_dirpath, logger=None):
    """Loads the training and validation data streams, and some key attributes.

    Parameters
    -------
    params : NFlowParams
        all the parameters for creating and training an NFlow model
    output_dirpath : str
        path for the output root folder
    logger : logging.Logger or equivalent
        logger for logging messages

    Returns
    -------
    data_streams : mb_tensorflow.tf.data_streams.DataStreams
        stuff related to the training and validation data streams
    a_means : np.ndarray
        a guess of the mean vector of the data
    a_stds : np.ndarray, optional
        a guess of the vector of standard deviations, each std for one dimension of the data
    """

    from .help_trainer import detect_input_dim, NFlowBatchMaker, make_dataset

    param_list = [
        ("training", params.training_generator_params),
        ("validation", params.validation_generator_params),
    ]

    for ml_kind, generator_params in param_list:
        if generator_params.ml_kind != ml_kind:
            raise ValueError(
                "The `ml_kind` attribute of the {} generator must be '{}'. '{}' given.".format(
                    ml_kind, ml_kind, generator_params.ml_kind
                )
            )

    # load the wrangled dataframe locally or remotely
    wrangled_filepath = params.filepath()
    if os.path.exists(wrangled_filepath):
        df = load_any_df(wrangled_filepath, show_progress=True)
        if params.model_params.input_dim is None:
            params.model_params.input_dim = detect_input_dim(df)
    else:
        raise OSError("Expected '{}' but not found.".format(wrangled_filepath))

    batch_maker_init_args = (df,)
    make_dataset_args = (params.model_params.input_dim,)
    data_streams = load_trainval_streams(
        NFlowBatchMaker,
        make_dataset,
        batch_size=params.batch_size(),
        training_generator_params=params.training_generator_params,
        validation_generator_params=params.validation_generator_params,
        output_dirpath=output_dirpath,
        batch_maker_init_args=batch_maker_init_args,
        make_dataset_args=make_dataset_args,
        logger=logger,
    )
    a_means_and_stds = data_streams.validation_batch_maker.a_means_and_stds()
    if np.any(np.isnan(a_means_and_stds)):
        raise ValueError("NaN value detected in the feature data.")

    input_dim = len(a_means_and_stds)
    a_means = a_means_and_stds[:, 0]
    a_stds = a_means_and_stds[:, 1]
    if logger:
        logger.debug("input_dim={}".format(input_dim))
        logger.debug("a_means[:3]={}".format(a_means[:3]))
        logger.debug("a_stds[:3]={}".format(a_stds[:3]))

    return data_streams, a_means, a_stds
