from mb_utils.src.logging import Logger
from trainer import RealNvpParams,NFlowParams

def create_nflow_bijector(
    a_means: np.ndarray,
    a_stds: tp.Optional[np.ndarray] = None,
    params: NFlowModelParams = NFlowModelParams(),
    name: str = "nflow_bijector",
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

    from mt import tfp

    if params.arch != "realnvp":
        raise NotImplementedError(
            "Unknown how to implement an NFlow with arch '{}'.".format(params.arch)
        )

    realnvp_params = params.realnvp_params

    if not isinstance(realnvp_params.n_layers, int) or realnvp_params.n_layers < 0:
        raise ValueError(
            "Expected a positive integer for parameter 'realnvp_params.n_layers'. "
            "Got: {}.".format(realnvp_params.n_layers)
        )
    bijectors = []
    for i in range(realnvp_params.n_layers):
        if i % 2 == 0:
            fraction_masked = realnvp_params.fraction_masked
        else:
            fraction_masked = -realnvp_params.fraction_masked
        bijector = tfp.bijectors.RealNVP(
            fraction_masked=fraction_masked,
            shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(
                hidden_layers=realnvp_params.ln_hiddenDims,
                shift_only=True,
            ),
            name="realnvp",
        )

        bijectors.append(bijector)