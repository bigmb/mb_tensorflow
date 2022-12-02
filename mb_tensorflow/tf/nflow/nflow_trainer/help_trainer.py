"""A augmentor iterating over dataframe rows generating data for NFlow training and validation.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from wml.tfkeras import augmentors


__all__ = ["detect_input_dim", "NFlowBatchMaker", "make_dataset"]


def detect_input_dim(df: pd.DataFrame) -> int:
    """Detects the input dimensionality from the dataframe based on its column names."""

    columns = [x for x in df.columns if x.startswith("feat_")]
    return len(columns)


class NFlowBatchMaker(augmentors.BaseBatchMaker):
    """A class-based function flowing over dataframe rows for NFlow training and validation.

    Parameters
    ----------
    df : pandas.DataFrame
        wrangled table for NFlow. It should contain columns `ml_kind` and `ml_weight`. It should
        then contain at least `newfood_mask_url` or `vfr_pred_url`.
    batch_size : int, default 64
        batch size
    params : wml.tfkeras.augmentors.BaseAugmentorParams
        parameters for the augmentor
    debug_dirpath : str, optional
        local dirpath containing some batches
    logger : logging.Logger or equivalent
        logger for debugging purposes

    Notes
    -----
    Make sure to check the notes of :class:`wml.tfkeras.augmentors.BaseAugmentorParams`
    regarding two operating modes: sequential and resampling. Some last events that are not
    sufficient to form a batch may be ignored.

    If the `params.ml_kind` is used to filter the events based on its kind 'training',
    'validation', 'prediction' or 'evaluation'.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 64,
        params: augmentors.BaseAugmentorParams = augmentors.BaseAugmentorParams(),
        debug_dirpath: str = None,
        logger=None,
    ):
        super().__init__(
            df,
            batch_size=batch_size,
            params=params,
            debug_dirpath=debug_dirpath,
            logger=logger,
        )

        self.input_dim = detect_input_dim(df)

    def __call__(self, work_id):
        """Generates the next batch.

        Returns
        ------
        av_feats : numpy.array(shape=(B, input_dim))
            batch of input feature vectors
        """

        df, rng = self.pick_events(work_id)
        N = len(df)

        # allocate arrays
        av_feats = np.empty((N, self.input_dim), dtype=np.float32)

        # augmentation
        item_id = 0
        for _, row in df.iterrows():
            for i in range(self.input_dim):
                av_feats[item_id, i] = row["feat_" + str(i)]
            item_id += 1

        return av_feats

    def a_minmax(self):
        res = np.empty((self.input_dim, 2), dtype=np.float32)
        for d in range(self.input_dim):
            key = "feat_{}".format(d)
            res[d, 0] = self.df[key].min()
            res[d, 1] = self.df[key].max()
        return res

    def a_means_and_stds(self):
        res = np.empty((self.input_dim, 2), dtype=np.float32)
        for d in range(self.input_dim):
            key = "feat_{}".format(d)
            res[d, 0] = self.df[key].mean()
            res[d, 1] = self.df[key].std()
        return res


def make_dataset(
    iterator, params: augmentors.BaseAugmentorParams, input_dim: int
) -> tf.data.Dataset:
    """Wraps an NFlow-batch-maker-as-an-iterator into a tf.data.Dataset.

    Parameters
    ----------
    iterator : mt.base.concurrency.WorkIterator
        an ahead-of-time iterator that makes batches
    params : wml.tfkeras.augmentors.BaseAugmentorParams
        parameters for the above iterator
    input_dim : int
        input feature dimensionality

    Returns
    -------
    tensorflow.data.Dataset
        a Dataset that generates batches forever

    Notes
    -----
    See :class:`NFlowBatchMaker` for more details.
    """

    if input_dim is None:
        raise ValueError("The input dimensionality must not be None.")

    # the generator wrapper
    def generator():
        while True:
            bunch = next(iterator)

            input_dict = {"y": bunch}
            output_dict = {}

            yield input_dict, output_dict

    if tf.__version__ < "2.6":
        raise ImportError("We need at least TF 2.6 to work.")

    # build the signature for TF 2.6 or newer
    input_sig = {"y": tf.TensorSpec(shape=(None, input_dim), dtype=tf.dtypes.float32)}
    output_sig = {}

    # return the final product
    return tf.data.Dataset.from_generator(
        generator, output_signature=(input_sig, output_sig)
    )
