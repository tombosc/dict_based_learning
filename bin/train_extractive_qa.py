#!/usr/bin/env python
from dictlearn.extractive_qa_training import train_extractive_qa
from dictlearn.extractive_qa_configs import qa_config_registry
from dictlearn.main import main


if __name__ == "__main__":
    main(qa_config_registry, train_extractive_qa)
