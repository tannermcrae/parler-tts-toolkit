# Parler TTS Toolkit
Preprocessing scripts, custom training scripts, and inference optimization code for training and running parler tts. The model code and parts of the training script are based on the original ParlerTTS project. The license and notice file reflect that. 

I've modified the training script and added inference code using torchserve to train and run the model in SageMaker.


## Training
Before building the container, you'll need to update your dataset location in the training/config/config.json file

Then, to build the training container, run the following commands

```bash
$ . ./script/build-serve-container.sh train
```

Then follow the instructions in the training notebook to create a training job.

Once the container is built, you can execute a SageMaker training job either via Jupyter Notebook or by executing the training script directly in a CI/CD pipeline. Notebooks are recommended for initial development and testing.

## Inference 
To build the inference container, run the following commands
```bash
$ . ./script/build-serve-container.sh inference
```

Once the container is built and pushed to ECR, you can deploy the model either via Jupyter Notebook or by executing the inference script directly. Notebooks are recommended for initial development and testing.

# License
This project reuses a lot from the original ParlerTTS codebase and is distributed under the terms specified in the LICENSE file (Apache 2.0).

# Acknowledgments
Original ParlerTTS project contributors
