#!/usr/bin/env python

from networks.cifar.cifar_model import CifarModel


class Custom(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = args.custom_name
        CifarModel.__init__(self, args)

    def load(self):
        """
        TODO: Write Comment
        """

        if self.args.verbose: print(f"Loading Model...")
        
        self._model = self.load_model(f"{self.log_filepath}model.h5")
        self._model.load_weights(f"{self.log_filepath}model_weights.h5")

    