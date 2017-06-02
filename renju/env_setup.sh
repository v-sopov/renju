#!/bin/bash

pip install -r autonomous_reqs.txt
cd cpp_ext
python setup.py install
cd ../
wget https://www.dropbox.com/s/eoi3gdea5zfj75p/large_policy_model -q --show-progress
