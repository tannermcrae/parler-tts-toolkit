#!/bin/bash

# Handle the 'serve' argument from SageMaker
if [ "$1" = "serve" ]; then
    # Start torchserve with the specified configuration.
    # SageMaker will untar the .mar file into the /opt/ml/model directory.
    exec torchserve \
        --start \
        --model-store /opt/ml/model \
        --models tts=tts_model.mar \
        --ts-config config.properties \
        --foreground
else
    # If not 'serve', pass all arguments to torchserve. With SM this else will never execute.
    # Left it in in case you want to run outside of SM.
    exec torchserve "$@"
fi