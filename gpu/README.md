# GPU Container setup

**WARNING: This container uses CUDA 11.3 in order to use this container the system need a cuda driver which is at least version 11.3!**

This will set up a fresh git cloned project for the GPU docker step by step, please make sure to complete every step.
**To run an own model instead of the provided one please adjust the .env file accordingly and put the model in some subfolder of the root git.**
``` bash
# Go to git root directory
cd ..

# Copy the environment file
cp .env.sample .env

# Download the bert basic model
wget https://zenodo.org/record/4614023/files/basic_mbert.zip?download=1

# Unzip it 
unzip basic_mbert

# Remove copy right protected files, and download from origin
cd src/util && rm conll18_ud_eval.py && rm iwpt20_xud_eval.py
wget http://universaldependencies.org/conll18/conll18_ud_eval.py && wget https://universaldependencies.org/iwpt20/iwpt20_xud_eval.py

# Go back to root directory
cd ../..

# Build the steps parser and create a new container for it
docker build -t cuda_steps_parser:latest . -f gpu/dockerfile

# Run the docker container on host port 8000 this is for test starting the container
# docker run --rm -p 0.0.0.0:8000:8000 --env STEPS_PARSER_PORT=8000 --gpus all cuda_steps_parser

# For creating as service
docker service create --replicas 1 \
  --name cuda_steps_parser_service \
  --generic-resource "gpu=1" \
  -p 8000 \
  --env STEPS_PARSER_PORT=8000 \
  cuda_steps_parser
```


## Performance analysis

**CPU container:**
|Total runtime: 57.0s | Model runtime: 52.82s|

**GPU container:**
|Total runtime: 59.0s | Model runtime: 54.79s|
