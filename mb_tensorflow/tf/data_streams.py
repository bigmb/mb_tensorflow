"""Utilities to manage training and validation data streams."""

import pandas as pd
import os

from ..params.params_all import BaseParams
from .base import BaseBatchMaker



class DataStreams(object):
    """Proxy class to hold data streams. It can be used in a with statement.

    Parameters
    ----------
    batch_size : int
        number of events per batch (or subbatch if it is distributed training)
    training_batch_maker : object
        the training batch maker
    training_iterator : mt.base.WorkIterator
        ahead-of-time iterator to iterate over new training batches
    training_dataset : tensorflow.data.Dataset
        training batch stream
    steps_per_epoch : int or None
        number of training steps per epoch
    validation_batch_maker : object
        the validation batch maker
    validation_iterator : mt.base.WorkIterator
        ahead-of-time iterator to iterate over new validation batches
    validation_steps : int or None
        number of validation steps per epoch
    validation_dataset : tensorflow.data.Dataset
        validation batch stream
    """

    def __init__(
        self,
        batch_size,
        training_batch_maker,
        training_iterator,
        training_dataset,
        steps_per_epoch,
        validation_batch_maker,
        validation_iterator,
        validation_dataset,
        validation_steps,
    ):
        self.batch_size = batch_size
        self.training_batch_maker = training_batch_maker
        self.training_iterator = training_iterator
        self.training_dataset = training_dataset
        self.steps_per_epoch = steps_per_epoch
        self.validation_batch_maker = validation_batch_maker
        self.validation_iterator = validation_iterator
        self.validation_dataset = validation_dataset
        self.validation_steps = validation_steps

    def close(self):
        self.validation_iterator.close()
        self.training_iterator.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def __del__(self):
        self.close()


def load_trainval_streams(
    batch_maker_class,
    make_dataset_func,
    batch_size: int = 32,
    training_generator_params: BaseParams = BaseParams(),
    validation_generator_params: BaseParams = BaseParams(),
    output_dirpath: str = None,
    batch_maker_init_args=(),
    batch_maker_init_kwargs={},
    make_dataset_args=(),
    make_dataset_kwargs={},
    logger=None,
):
    """Generic function that loads the training and validation data streams, after things have been prepared.

    Parameters
    ----------
    batch_maker_class : object
        a subclass of :class:`wml.tfkeras.augmentors.BaseBatchMaker` whose constructor contains
        'batch_size', 'params', 'debug_dirpath' and 'logger' at the minimum
    make_dataset_func : object
        a function to make a :class:`tensorflow.data.Dataset` that takes as input an iterator and
        the generator params at the minimum
    batch_size : int
        number of items per batch
    training_generator_params : BaseParams
        all the parameters for augmenting the training items
    validation_generator_params : BaseParams
        all the parameters for augmenting the validation items
    output_dirpath : str
        path for the output root folder
    batch_maker_init_args : tuple
        additional positional arguments for the batch maker's constructor
    batch_maker_init_kwargs : dict
        additional keyword arguments for the batch maker's constructor
    make_dataset_args : tuple
        additional positional arguments for the function to make a dataset
    make_dataset_kwargs : dict
        additional keyword arguments for the function to make a dataset
    logger : logging.Logger or equivalent
        logger for logging messages

    Returns
    -------
    data_streams : DataStreams
        stuff related to the training and validation data streams
    """

    import mt.base.concurrency as _bc
    #from mt.base import debug_exec

    # create a training data augmentor
    if output_dirpath is not None:
        debug_dirpath = os.path.join(output_dirpath, "training")
        os.path.make_dirs(debug_dirpath)
    else:
        debug_dirpath = None
    training_batch_maker = batch_maker_class(
        *batch_maker_init_args,
        batch_size=batch_size,
        params=training_generator_params,
        debug_dirpath=debug_dirpath,
        logger=logger,
        **batch_maker_init_kwargs,) 
        
    steps_per_epoch = len(training_batch_maker)
    training_iterator = _bc.WorkIterator(training_batch_maker, skip_null=True, logger=logger)
        
    training_dataset = make_dataset_func(training_iterator,training_generator_params,*make_dataset_args,**make_dataset_kwargs,)

    if debug_dirpath is not None:
        # image_pair not available in pdh5 yet
        train_filepath = os.path.join(debug_dirpath, "train.csv.zip")
        pd.dfsave(training_batch_maker.df, train_filepath, index=False)

    # create a validation data augmentor
    if output_dirpath is not None:
        debug_dirpath = os.path.join(output_dirpath, "validation")
        os.path.make_dirs(debug_dirpath)
    else:
        debug_dirpath = None
        
    validation_batch_maker = batch_maker_class(*batch_maker_init_args,
        batch_size=batch_size,params=validation_generator_params,debug_dirpath=debug_dirpath,
        logger=logger,**batch_maker_init_kwargs,)

    validation_steps = len(validation_batch_maker)
    validation_iterator = _bc.WorkIterator(validation_batch_maker, skip_null=True, logger=logger)
    validation_dataset = make_dataset_func(
        validation_iterator,
        validation_generator_params,
        *make_dataset_args,
        **make_dataset_kwargs,)

    if debug_dirpath is not None:
        # image_pair not available in pdh5 yet
        val_filepath = os.path.join(debug_dirpath, "val.csv.zip")
        pd.dfsave(validation_batch_maker.df, val_filepath, index=False)

    return DataStreams(
        batch_size,
        training_batch_maker,
        training_iterator,
        training_dataset,
        steps_per_epoch,
        validation_batch_maker,
        validation_iterator,
        validation_dataset,
        validation_steps,
    )
