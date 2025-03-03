#!/bin/bash

# Handle the 'serve' argument from SageMaker
if [ "$1" = "train" ]; then
    # Start torchserve with the specified configuration.
    # SageMaker will untar the .mar file into the /opt/ml/model directory.
    accelerate launch training/train.py training/config/config.json
else
    # If not 'serve', pass all arguments to torchserve. With SM this else will never execute.
    # Left it in in case you want to run outside of SM.
    echo "Training failed. Please check the training logs for more information."
fi

# Remove checkpoints
rm -rf model_artifacts/checkpoint-*

# Archive the model and place it in the /opt/ml/model directory so SageMaker will pick it up.
torch-model-archiver --model-name tts_model \
    --version 1.0 \
    --handler inference/handler.py \
    --extra-files "model_artifacts/,src/" \
    --export-path /opt/ml/model \
    --requirements-file inference/requirements.txt \
    --force



