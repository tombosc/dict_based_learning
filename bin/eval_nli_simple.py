#!/usr/bin/env python
from dictlearn.nli_training import evaluate
from dictlearn.main import main_evaluate
from dictlearn.nli_esim_config_registry import nli_esim_config_registry

from functools import partial

if __name__ == "__main__":
    main_evaluate(nli_esim_config_registry, partial(evaluate, model="simple"))
