from mb_utils.src.logging import Logger
from .trainer_class import NFlowParams
import numpy as np
import typing as tp
import tensorflow_probability as tfp

__all__ = ['create_realnvp_bijector']

def create_realnvp_bijector(
    a_means: np.ndarray,
    a_stds: tp.Optional[np.ndarray] = None,
    params: NFlowParams = NFlowParams(),
    name: str = "realnvp_bijector",
    logger=None,
):
    """Creates an NFlow bijector.

    Parameters
    ----------
    a_means : np.ndarray
        a guess of the mean vector of the data
    a_stds : np.ndarray, optional
        a guess of the vector of standard deviations, each std for one dimension of the data. If
        not given, the standard deviations are assumed ones.
    params : NFlowModelParams
        parameters defining the NFlow bijector
    name : str
        bijector model name
    logger : logging.Logger or equivalent
        logger for debugging purposes

    Returns
    -------
    bijector : tensorflow_probability.bijectors.Bijector
        an NFlow bijector
    """

    if logger:
        logger.info("Creating NFlow bijector")
    if params.arch != "RealNvp":
        raise NotImplementedError(
            "Unknown how to implement an NFlow with arch '{}'.".format(params.arch)
        )
    if logger:
        logger.info("Creating RealNvp bijector")

    realnvp_params = params.RealNvpParams
    if logger:
        logger.info("RealNvp parameters: {}".format(realnvp_params))

    if not isinstance(realnvp_params.layers, int) or realnvp_params.layers < 0:
        raise ValueError(
            "Expected a positive integer for parameter 'realnvp_params.n_layers'. "
            "Got: {}.".format(realnvp_params.layers)
        )
    
    if logger:
        logger.info("Creating RealNvp bijector")
    bijectors = []
    for i in range(realnvp_params.layers):
        if i % 2 == 0:
            fraction_masked = realnvp_params.fraction_masked
        else:
            fraction_masked = -realnvp_params.fraction_masked
        bijector = tfp.bijectors.RealNVP(
            fraction_masked=fraction_masked,
            shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(
                hidden_layers=realnvp_params.hidden_layers,
                shift_only=realnvp_params.shift_only,
            ),
            name="realnvp",
        )

        bijectors.append(bijector)

    if a_stds is not None:
        bijectors.append(tfp.bijectors.Scale(a_stds))
    bijectors.append(tfp.bijectors.Shift(a_means))

    if logger:
        logger.info('RealNvp bijectors: {}'.format(bijectors))

    return tfp.bijectors.Chain(bijectors, name=name)
