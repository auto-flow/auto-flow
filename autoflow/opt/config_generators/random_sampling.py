from autoflow.opt.config_generators.base import BaseConfigGenerator




class RandomSampling(BaseConfigGenerator):
    """
        class to implement random sampling from a ConfigSpace
    """

    def __init__(self, configspace, **kwargs):
        """

        Parameters:
        -----------

        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors
        **kwargs:
            see  hyperband.config_generators.base.BaseConfigGenerator for additional arguments
        """

        super().__init__()
        self.configspace = configspace


    def get_config(self, budget):
        return(self.configspace.sample_configuration().get_dictionary(), {})
