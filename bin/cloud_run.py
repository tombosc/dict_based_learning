#!/usr/bin/env python

import json
import tempfile
import os
import subprocess
import argparse

JOB_TEMPLATE = """
{
  "jobDefinition": {
    "name": "squad",
    "clusterId": 249,
    "description": "squad",
    "dockerImage": "trydgx_mila/project2:16",
    "jobType": "BATCH",
    "command": "su bahdanau; git clone git@github.com:tombosc/dict_based_learning.git; cd /data/cf9ffb48-61bd-40dc-a011-b2e7e5acfd72/models; /workspace/dict_based_learning/bin/cloud_train_extractive_qa.sh squad2 squad2 2>err.txt",
    "resources": {
      "gpus": 1,
      "systemMemory": 61440,
      "cpuCores": 2
    },
    "jobDataLocations": [
      {
        "mountPoint": "/data",
        "protocol": "NFSV3"
      }
    ],
    "portMappings": []
  }
}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", help="A path to the template")
    parser.add_argument("--gpus", type=int, default=1, help="Number of gpus")
    parser.add_argument("commands", nargs='+', help="Commands to run")
    args = parser.parse_args()

    if args.template:
        job_specs = json.load(open(args.template))
    else:
        job_specs = json.loads(JOB_TEMPLATE)
    job_specs['jobDefinition']['command'] = ' '.join(args.commands)
    job_specs['jobDefinition']['resources']['gpus'] = args.gpus

    _, path = tempfile.mkstemp()
    try:
        with open(path, 'w') as dst:
            json.dump(job_specs, dst)
        popen = subprocess.Popen(['dgx', 'job', 'submit', '-f', path])
        popen.wait()
    finally:
        os.remove(path)

if __name__ == '__main__':
    main()
