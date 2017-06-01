#!/bin/bash

sudo pip3 install -r requirements.txt
cd cpp_ext
sudo python3 setup.py install
cd ../
wget https://www.dropbox.com/s/eoi3gdea5zfj75p/large_policy_model -q --show-progress
