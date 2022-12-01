import tensorflow_probability as tfp
import numpy as np
import typing as tp
from .trainer_class import NFlowParams

__all__ = ['create_maf_bijector']

def create_maf_bijector(
    a_means: np.ndarray,
    a_stds: tp.Optional[np.ndarray] = None,
    params: NFlowParams = NFlowParams(),
    name: str = "maf_bijector",
    logger=None) -> tfp.bijectors.Bijector:
    """
    Creates an NFlow bijector.
    """
    if logger:
        logger.info("Creating NFlow bijector")
    if params.arch != "MAF":
        raise NotImplementedError(
            "Unknown how to implement an NFlow with arch '{}'.".format(params.arch)
        )
    if logger:
        logger.info("Creating MAF bijector")
    
    maf_params = params.maf_params
    if logger:
        logger.info("MAF parameters: {}".format(maf_params))
    
    if not isinstance(maf_params.n_layers, int) or maf_params.n_layers < 0:
        raise ValueError(
            "Expected a positive integer for parameter 'maf_params.n_layers'. "
            "Got: {}.".format(maf_params.n_layers)
        )

    pass