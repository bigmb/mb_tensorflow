from ...params.params_all import ModelParams

__all__ = ['NFlowParams','RealNvpParams']

class RealNvpParams(ModelParams):
    """
    Parameters for RealNvp models.
    layers : int
        number of layers
    hidden : int
        number of hidden units
    fraction_masked : float
        fraction of features to mask
    gen : int
        model generation/family number, starting from 1
    """
    yaml_tag = "!RealNvp_Params"

    def __init__(self, layers : int =4 ,hidden_layers : int = 256, fraction_masked : float = 0.5, gen: int = 1):
        super().__init__(gen=gen)
        self.layers = layers
        self.hidden_layers = hidden_layers
        self.fraction_masked = fraction_masked    


class NFlowParams(ModelParams):
    """
    Parameters for defining and creating a normalizing flow model.

    Parameters
    ----------
    arch : str
        architecture of the normalizing flow model
    arch_params: ArchParams
        parameters for the architecture of the normalizing flow model
    input_dim : int
        dimension of the input
    gen : int
        model generation/family number, starting from 1
    """
    
    yaml_tag = "!Tf_Nflow"

    def __init__(self, arch: str = "RealNvp", arch_params: RealNvpParams = RealNvpParams(), input_dim: int = 128,gen: int = 1):
        super().__init__(gen=gen)
        self.arch = arch
        self.arch_params = arch_params
        self.input_dim = input_dim
