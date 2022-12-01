import numpy as np
import tensorflow_probability as tfp
from .trainer_class import NFlowParams
import typing as tp
from realnvp import create_realnvp_bijector
from maf import create_maf_bijector

def create_nflow_model(
    a_means: np.ndarray,
    a_stds: tp.Optional[np.ndarray] = None,
    params: NFlowParams = NFlowParams(),
    name: str = "nflow_distribution",
    bijector_name: str = "realnvp",
    logger=None,
):
    """Creates an NFlow model that can transform the multivariate normal distribution to data.

    Parameters
    ----------
    a_means : np.ndarray
        a guess of the mean vector of the data
    a_stds : np.ndarray, optional
        a guess of the vector of standard deviations, each std for one dimension of the data. If
        not given, the standard deviations are assumed ones.
    params : NFlowModelParams
        parameters defining the NFLow bijector
    name : str
        model name
    logger : logging.Logger or equivalent
        logger for debugging purposes

    Returns
    -------
    model : tensorflow_probability.distributions.TransformedDistribution
        an NFlow model
    """

    if logger:
        logger.info("Creating NFlow model")
    
    input_dim = len(a_means)
    
    if input_dim is None:
        if params.input_dim is None:
            raise ValueError("You must specify the input dimensionality.")

    if logger:
        logger.info("Input dimensionality: {}".format(input_dim))

    loc = np.zeros(input_dim, dtype=np.float32)
    dist = tfp.distributions.MultivariateNormalDiag(loc=loc)

    if logger:
        logger.info("size of loc : {}".format(loc.shape))
        logger.info("size of dist : {}".format(dist.shape))
        logger.info("Bijector Name : {}".format(bijector_name))

    if bijector_name == "realnvp":
            nflow_bijector = create_realnvp_bijector(
                a_means, a_stds=a_stds, params=params, logger=logger)
    if bijector_name == "maf":
            nflow_bijector = create_maf_bijector(
                a_means, a_stds=a_stds, params=params, logger=logger)

    if logger:
        logger.info("Bijector called.")
        logger.info("Creating TransformedDistribution for the Bijector")

    transformed_dist = tfp.distributions.TransformedDistribution(
        distribution=dist, bijector=nflow_bijector, name=name
    )

    params.input_dim = input_dim
    transformed_dist.model_params = params
    
    if logger:
        logger.info("TransformedDistribution created.")

    return transformed_dist
