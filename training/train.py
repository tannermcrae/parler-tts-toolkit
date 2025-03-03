#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Train Parler-TTS using 🤗 Accelerate"""

import logging
import os
import re
import sys
import time
import math
import contextlib
from multiprocess import set_start_method
from datetime import timedelta
import inspect
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import DatasetDict, Dataset, IterableDataset, concatenate_datasets

from huggingface_hub import HfApi

import transformers
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry


from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin, DistributedDataParallelKwargs
from accelerate.utils.memory import release_memory

from src.parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
    build_delay_pattern_mask,
)

from utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred,
    log_metric,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)
from arguments import ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments
from data import DataCollatorParlerTTSWithPadding, DataCollatorEncodecWithPadding
from eval import clap_similarity, wer, si_sdr
from trainer import ParlerTTSTrainer
from data_processor import DataProcessor


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_parler_tts", model_args, data_args)

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    if data_args.pad_to_max_length and (
        data_args.max_duration_in_seconds is None
        or data_args.max_prompt_token_length is None
        or data_args.max_description_token_length is None
    ):
        raise ValueError(
            "`pad_to_max_length` is `True` but one of the following parameters has not been set: `max_duration_in_seconds`, `max_prompt_token_length`, `max_description_token_length`"
        )

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    ####### A. Preparation
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=120)), DistributedDataParallelKwargs(find_unused_parameters=False)]

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=None,
        project_dir=training_args.output_dir,
        kwargs_handlers=kwargs_handlers,
    )

    accelerator.init_trackers(
        project_name="parler-tts",
        config={
            "learning_rate": training_args.learning_rate,
            "model_name_or_path": model_args.model_name_or_path,
            "num_train_epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "global_batch_size": training_args.per_device_train_batch_size * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_steps": training_args.warmup_steps,
            "freeze_text_encoder": model_args.freeze_text_encoder,
            "max_duration_in_seconds": data_args.max_duration_in_seconds,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "temperature": model_args.temperature,
        }
    )

    # Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, lett's instantiate the feature extractor, tokenizers and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # load prompt tokenizer
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        model_args.prompt_tokenizer_name or model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side=model_args.prompt_padding_side,
    )

    # load description tokenizer
    description_tokenizer = AutoTokenizer.from_pretrained(
        model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
    )

    # 3. Next, let's load the config.
    config = ParlerTTSConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
 
    if training_args.codebook_weights is not None and len(training_args.codebook_weights) != config.decoder.num_codebooks:
        raise ValueError(f"`codebook_weights` has length {len(training_args.codebook_weights)} when it should be of length {config.decoder.num_codebooks}.")

    # update pad token id and decoder_start_token_id
    config.decoder.update(
        {
            "cross_attention_implementation_strategy": model_args.cross_attention_implementation_strategy
            if model_args.cross_attention_implementation_strategy is not None
            else None,
            "codebook_weights": training_args.codebook_weights if training_args.codebook_weights is not None else config.decoder.codebook_weights
        }
    )
    config.update(
        {
            "pad_token_id": model_args.pad_token_id if model_args.pad_token_id is not None else config.pad_token_id,
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else config.decoder_start_token_id,
        }
    )

    # create model
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        attn_implementation={"decoder": model_args.attn_implementation, "text_encoder": "eager"},
    )

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # derive max & min input length for sample rate & max duration
    audio_encoder_eos_token_id = config.decoder.eos_token_id
    audio_encoder_bos_token_id = model.generation_config.decoder_start_token_id
    num_codebooks = model.decoder.config.num_codebooks

    # Freeze Encoders
    model.freeze_encoders(model_args.freeze_text_encoder)

    # Test all gather - used for warmout and avoiding timeout
    logger.debug(str(accelerator.process_index), main_process_only=False, in_order=True)
    test_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered_tensor = accelerator.gather(test_tensor)
    print("gathered_tensor", gathered_tensor)
    accelerator.wait_for_everyone()

        # Replace all dataset loading/processing code with DataProcessor
    data_processor = DataProcessor(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        feature_extractor=feature_extractor,
        prompt_tokenizer=prompt_tokenizer,
        description_tokenizer=description_tokenizer,
        config=config,
        model=model,
        accelerator=accelerator,
    )
    
    # Get processed datasets and audio max length
    vectorized_datasets, audio_max_length = data_processor.prepare_datasets()
    
    # Short circuit. Check for preprocessing_only option
    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files save at {data_args.save_to_disk}")
        return
    
    # Apply length grouping if needed
    if training_args.group_by_length:
        vectorized_datasets = data_processor.add_target_lengths_for_grouping(vectorized_datasets)


    # 6. Next, we can prepare the training.

    # Define optimizer, LR scheduler, collator
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    # Define Training Schedule
    # Store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        total_train_steps = int(training_args.max_steps)
        num_epochs = sys.maxsize

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(total_train_steps) * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    # Instantiate custom data collator
    data_collator = DataCollatorParlerTTSWithPadding(
        prompt_tokenizer=prompt_tokenizer,
        description_tokenizer=description_tokenizer,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        padding=padding,
        prompt_max_length=data_args.max_prompt_token_length,
        description_max_length=data_args.max_description_token_length,
        audio_max_length=audio_max_length,
    )

    # Extract necessary values from model config before it's wrapped
    audio_encoder_bos_token_id = model.generation_config.decoder_start_token_id
    audio_encoder_eos_token_id = config.decoder.eos_token_id
    num_codebooks = model.decoder.config.num_codebooks

    # Prepare everything with accelerate
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # Find resume checkpoint if available
    resume_step = None
    cur_step = 0
    epochs_trained = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        if training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            with accelerator.local_main_process_first():
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
    else:
        resume_step = None

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    # only the main process saves them
    if accelerator.is_main_process:
        # save feature extractor, tokenizer and config
        if (
            model_args.prompt_tokenizer_name is None
            and model_args.description_tokenizer_name
            or (model_args.prompt_tokenizer_name == model_args.description_tokenizer_name)
        ):
            prompt_tokenizer.save_pretrained(training_args.output_dir)
        else:
            logger.warning(
                f"Prompt tokenizer ('{model_args.prompt_tokenizer_name}') and description tokenizer ('{model_args.description_tokenizer_name}') are not the same. Saving only the prompt tokenizer."
            )
            prompt_tokenizer.save_pretrained(training_args.output_dir)

        feature_extractor.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    # Initialize trainer
    trainer = ParlerTTSTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        training_args=training_args,
        model_args=model_args,
        data_args=data_args,
        vectorized_datasets=vectorized_datasets,
        data_collator=data_collator,
        feature_extractor=feature_extractor,
        prompt_tokenizer=prompt_tokenizer,
        description_tokenizer=description_tokenizer,
        config=config,
        mixed_precision=mixed_precision,
        audio_encoder_bos_token_id=audio_encoder_bos_token_id,
        audio_encoder_eos_token_id=audio_encoder_eos_token_id,
        num_codebooks=num_codebooks,
    )
    
    # Train the model
    if training_args.do_train:
        cur_step = trainer.train(resume_step=resume_step, cur_step=cur_step, epochs_trained=epochs_trained)
    
    # Save final model
    if accelerator.is_main_process:
        logger.info(f"Training complete! Final model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()