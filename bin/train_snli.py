#!/usr/bin/env python
from dictlearn.snli_simple_training import train_snli_model
from dictlearn.snli_config_registry import snli_config_registry
from dictlearn.main import main


if __name__ == "__main__":
    main(snli_config_registry, train_snli_model)
