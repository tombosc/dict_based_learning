#!/usr/bin/env python
from dictlearn.language_model_training import train_language_model
from dictlearn.language_model_configs import lm_config_registry
from dictlearn.main import main


if __name__ == "__main__":
    main(lm_config_registry, train_language_model)
