#!/usr/bin/env python
from dictlearn.extractive_qa_training import evaluate_extractive_qa
from dictlearn.extractive_qa_configs import qa_config_registry
from dictlearn.main import main_evaluate


if __name__ == "__main__":
    main_evaluate(qa_config_registry, evaluate_extractive_qa)
