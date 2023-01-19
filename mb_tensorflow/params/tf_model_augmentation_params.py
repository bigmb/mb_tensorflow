#file to set model and augmentation parameters
import yaml

__all__ = ["Tf_Train_Augmentation_Params","Tf_Val_Augmentation_Params","Tf_Model_Params"]

class Tf_Train_Augmentation_Params(yaml.YAMLObject):
    """
    Class to read the augmentation parameters from yaml tag : !Tf_Augmentation_Params
    """
    yaml_tag = '!Training_Params'

    def __init__(self,
        image_size=[512, 512],  # size of the images
        zoom_range=[0.9, 1.1],  # range for the zoom parameters
        shear_range=[-0.1, 0.1],  # range for the shear parameters
        rotate=[-20, +20],  # range for the rotation angle parameter
        random_fliplr=True,  # whether or not to randomly left-right flip
        random_flipud=True,  # whether or not to randomly up-down flip
        move_range=[-0.1, 0.1],  # range for the move parameters
    ):

        self.image_size = image_size
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.angle_range = rotate
        self.random_fliplr = random_fliplr
        self.random_flipud = random_flipud
        self.move_range = move_range

    
class Tf_Val_Augmentation_Params(yaml.YAMLObject):
    """
    Class to read the augmentation parameters from yaml tag : !Tf_Augmentation_Params
    """
    yaml_tag = '!Validation_Params'

    def __init__(self,
        image_size=[512, 512],  # size of the images
        zoom_range=[0.9, 1.1],  # range for the zoom parameters
        shear_range=[-0.1, 0.1],  # range for the shear parameters
        rotate=[-20, +20],  # range for the rotation angle parameter
        random_fliplr=True,  # whether or not to randomly left-right flip
        random_flipud=True,  # whether or not to randomly up-down flip
        move_range=[-0.1, 0.1],  # range for the move parameters
    ):

        self.image_size = image_size
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.angle_range = rotate
        self.random_fliplr = random_fliplr
        self.random_flipud = random_flipud
        self.move_range = move_range

#check this snippet for the model parameters learning rate and optimizer
class LearningRateParams(yaml.YAMLObject):
    yaml_tag = "!LearningRateParams"

    """Parameters for defining a learning rate policy.

    Parameters
    ----------
    policy : {'const', 'poly', 'step', 'adam', 'adamw', 'rmsprop', 'novograd'}
        learning policy. If it is 'const', 'poly' or 'step', then the optimizer is SGD and a
        learning rate policy will be created. If it is 'adam', 'adamw' or 'rmsprop' then only the
        base will be used for the optimizer. If it is 'novograd', then the base and the weight
        decay factor will be used. The 'adamw' policy only works with tf.keras with TF 1.
    base : float
        base learning rate
    decay_step : float
        decay step
    decay_factor : float
        decay factor
    momentum : float
        momentum
    epochs : int32
        number of epochs, i.e. the number of times we loop the data, each of which we update the
        learning rate
    patience : int32
        maximum number of epochs where validation loss does not improve before we early-stop
    version : int32
        version of the class

    """

    def __init__(
        self,
        policy="poly",
        base=1e-4,
        decay_step=10,
        decay_factor=0.1,
        power=0.9,
        momentum=0.9,
        epochs=10,
        patience=15,
        version=1,
    ):

        self.policy = policy
        self.base = base
        self.decay_step = decay_step
        self.decay_factor = decay_factor
        self.power = power
        self.momentum = momentum
        self.epochs = epochs
        self.patience = patience
        self.version = version
