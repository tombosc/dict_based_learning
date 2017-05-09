#!/usr/bin/env python
from dictlearn.nli_training import train_snli_model
from dictlearn.nli_esim_config_registry import nli_esim_config_registry
from dictlearn.main import main


if __name__ == "__main__":
    main(nli_esim_config_registry, train_snli_model, model='esim')
