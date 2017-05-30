#!/bin/bash

pip install -r requirements.txt
python cpp_ext/setup.py install
wget https://www.dropbox.com/s/eoi3gdea5zfj75p/large_policy_model?dl=0 -o large_policy_model
