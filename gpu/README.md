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
The performance analysis takes the total time of 10 requests to the steps parser with a normal length wikipedia article. In order to check if the model takes up most of the time or if the overhead concerning deserializing of logits and labels and serializing the results back to the response we timed the total time and the function `_compute_logits_and_labels` in the file `multi_parser.py`. The result was that actually a lot of the time used by the steps parser is spent in the function invoking the model evaluation on the sentences. 

  |Container type | Total runtime | Runtime of cumulative calls to `_compute_logits_and_labels` |
  | ------------- | ------------- | ----------------------------------------------------------- |
  |CPU            |57.0s          | 52.82s                                                      |
  |GPU            |59.0s          | 54.79s                                                      |

**Conclusion:** The overhead of sending the sentences to the GPU is so large, that the calculational efficiency of the GPU does not matter that much. For this container the GPU does not offer superior performance but may improve the ressource utilization of the system.

### Performance: Further investigations
In the file `parse_service.py` the function `parser.parse()` is invoked for every sentence. So even for a large text corpus the GPU model is invoked for every sentence. This seems to be the primary bottleneck as the invokation and up and downloading of data to the GPU for every sentence is a lot of overhead. By further investigating the parse function one can see [here](https://github.com/ShadowItaly/steps-parser/blob/6e874813d14d04d0151e76824f2bdfa8b330c70a/src/models/multi_parser.py#L68) that the `parser.parse()` functions uses a batch size of one even though the class also provides a method to evaluate multiple sentences at once see `parser.evaluate_batch()` or [here](https://github.com/ShadowItaly/steps-parser/blob/6e874813d14d04d0151e76824f2bdfa8b330c70a/src/models/multi_parser.py#L88). By increasing the batch size to process multiple sentences at once instead of one at a time, one may be able to leverage the power of the GPU to improve execution speed additional to the improved GPU utilisation, this may need further investigations.
