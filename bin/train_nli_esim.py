#!/usr/bin/env python
from dictlearn.nli_esim_training import train_nli_esim_model
from dictlearn.nli_simple_config_registry import snli_config_registry
from dictlearn.main import main


if __name__ == "__main__":
    main(snli_config_registry, train_nli_esim_model)
