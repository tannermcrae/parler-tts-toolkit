import logging
import os
import time
import math
import contextlib
from datetime import timedelta
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler
from accelerate.utils import AutocastKwargs
from accelerate.utils.memory import release_memory
from huggingface_hub import HfApi
from datasets import IterableDataset
from utils import rotate_checkpoints, log_metric

logger = logging.getLogger(__name__)

class ParlerTTSTrainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        accelerator,
        training_args,
        model_args,
        data_args,
        vectorized_datasets,
        data_collator,
        feature_extractor,
        prompt_tokenizer,
        description_tokenizer,
        config,
        mixed_precision,
        audio_encoder_bos_token_id,
        audio_encoder_eos_token_id,
        num_codebooks,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.vectorized_datasets = vectorized_datasets
        self.data_collator = data_collator
        self.feature_extractor = feature_extractor
        self.prompt_tokenizer = prompt_tokenizer
        self.description_tokenizer = description_tokenizer
        self.config = config
        self.mixed_precision = mixed_precision
        
        # Use the values passed directly instead of trying to access them through the model
        self.audio_encoder_bos_token_id = audio_encoder_bos_token_id
        self.audio_encoder_eos_token_id = audio_encoder_eos_token_id
        self.num_codebooks = num_codebooks
        
        # Set derived parameters
        self.per_device_train_batch_size = int(training_args.per_device_train_batch_size)
        self.train_batch_size = self.per_device_train_batch_size * accelerator.num_processes
        self.gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
        self.per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
        
        # Setup generation kwargs
        self.gen_kwargs = {
            "do_sample": model_args.do_sample,
            "temperature": model_args.temperature,
            "max_length": model_args.max_length,
            # Because of the delayed pattern mask, generation might stop earlier because of unexpected behavior
            # on the first tokens of the codebooks that are delayed.
            # This fix the issue.
            "min_new_tokens": self.num_codebooks + 1,
        }
        
        # Setup autocast kwargs for mixed precision
        self.autocast_kwargs = AutocastKwargs(enabled=(mixed_precision != "fp16"))
        
    def compute_metrics(
        self,
        audios,
        descriptions,
        prompts,
        device="cpu",
        compute_clap_similarity_metric=False,
        compute_noise_level_metric=False,
        noise_level_to_compute_clean_wer=None,
    ):
        from eval import clap_similarity, wer, si_sdr
        
        results = {}
        input_ids = descriptions
        texts = self.description_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prompts = self.prompt_tokenizer.batch_decode(prompts, skip_special_tokens=True)
        audios = [a.float().cpu().numpy() for a in audios]

        if compute_clap_similarity_metric:
            clap_score = clap_similarity(
                self.model_args.clap_model_name_or_path, texts, audios, device, input_sampling_rate=self.sampling_rate
            )
            results["clap"] = clap_score

        si_sdr_measures = None
        if compute_noise_level_metric:
            si_sdr_measures = si_sdr(audios, device, input_sampling_rate=self.sampling_rate)

        word_error, transcriptions, clean_word_error, noisy_word_error, percent_clean_samples = wer(
            self.model_args.asr_model_name_or_path,
            prompts,
            audios,
            device,
            self.training_args.per_device_eval_batch_size,
            self.sampling_rate,
            noise_level_to_compute_clean_wer,
            si_sdr_measures,
        )
        results["wer"] = word_error
        if clean_word_error is not None:
            results["clean_wer"] = clean_word_error
            results["noisy_wer"] = noisy_word_error
            results["percent_clean_samples"] = percent_clean_samples

        return results, texts, prompts, audios, transcriptions, si_sdr_measures
        
    def train_step(
        self,
        batch,
        num_items_in_batch,
    ):
        if self.mixed_precision == "fp16":
            # fp16 doesn't work with T5-like models
            with self.accelerator.autocast(autocast_handler=self.autocast_kwargs):
                if self.training_args.parallel_mode.value != "distributed":
                    encoder_outputs = self.model.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                else:
                    encoder_outputs = self.model.module.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                # we optionally project last_hidden_state to avoid recomputing every time
                encoder_hidden_states = encoder_outputs.last_hidden_state
                if (
                    self.config.text_encoder.hidden_size != self.config.decoder.hidden_size
                    and self.config.decoder.cross_attention_hidden_size is None
                ):
                    encoder_hidden_states = (
                        self.model.enc_to_dec_proj(encoder_hidden_states)
                        if self.training_args.parallel_mode.value != "distributed"
                        else self.model.module.enc_to_dec_proj(encoder_hidden_states)
                    )

                if batch.get("attention_mask", None) is not None:
                    encoder_hidden_states = encoder_hidden_states * batch.get("attention_mask", None)[..., None]

                encoder_outputs.last_hidden_state = encoder_hidden_states
                batch["encoder_outputs"] = encoder_outputs

        outputs = self.model(**batch, loss_reduction="sum")
        # CE (data) loss
        ce_loss = (outputs.loss * self.gradient_accumulation_steps * self.accelerator.num_processes) / num_items_in_batch

        metrics = {"loss": ce_loss}
        
        # per CE loss
        per_codebook_losses = outputs.per_codebook_losses
        metrics.update({
            f"codebook_{i}_loss": ((l * self.gradient_accumulation_steps * self.accelerator.num_processes) / num_items_in_batch) 
            for (i,l) in enumerate(per_codebook_losses)
        })
        
        return ce_loss, metrics

    def eval_step(self, batch):
        eval_model = self.model if not self.training_args.torch_compile else self.model._orig_mod

        if self.mixed_precision == "fp16":
            # fp16 doesn't work with T5-like models
            with self.accelerator.autocast(autocast_handler=self.autocast_kwargs):
                if self.training_args.parallel_mode.value != "distributed":
                    encoder_outputs = self.model.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                else:
                    encoder_outputs = self.model.module.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                # we optionally project last_hidden_state to avoid recomputing every time
                encoder_hidden_states = encoder_outputs.last_hidden_state
                if (
                    self.config.text_encoder.hidden_size != self.config.decoder.hidden_size
                    and self.config.decoder.cross_attention_hidden_size is None
                ):
                    encoder_hidden_states = (
                        self.model.enc_to_dec_proj(encoder_hidden_states)
                        if self.training_args.parallel_mode.value != "distributed"
                        else self.model.module.enc_to_dec_proj(encoder_hidden_states)
                    )

                if batch.get("attention_mask", None) is not None:
                    encoder_hidden_states = encoder_hidden_states * batch.get("attention_mask", None)[..., None]

                encoder_outputs.last_hidden_state = encoder_hidden_states
                batch["encoder_outputs"] = encoder_outputs

        with torch.no_grad():
            outputs = eval_model(**batch)
        # CE (data) loss
        ce_loss = outputs.loss
        metrics = {"loss": ce_loss}
        
        # per CE loss
        per_codebook_losses = outputs.per_codebook_losses
        metrics.update({f"codebook_{i}_loss": l for (i,l) in enumerate(per_codebook_losses)})
        return metrics

    def generate_step(self, batch):
        batch.pop("decoder_attention_mask", None)
        eval_model = self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=True)
        if self.training_args.torch_compile:
            # if the model is compiled, we use the original model bc compile is not compatible with .generate
            eval_model = self.model._orig_mod

        output_audios = eval_model.generate(**batch, **self.gen_kwargs)
        output_audios = self.accelerator.pad_across_processes(output_audios, dim=1, pad_index=0)
        return output_audios

    def train(self, resume_step=None, cur_step=0, epochs_trained=0):
        """Main training loop"""
        # Calculate training steps
        if self.training_args.max_steps < 0:
            num_epochs = int(self.training_args.num_train_epochs)
            steps_per_epoch = len(self.vectorized_datasets["train"]) // (self.train_batch_size * self.gradient_accumulation_steps)
            total_train_steps = steps_per_epoch * num_epochs
        elif self.training_args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")
            total_train_steps = int(self.training_args.max_steps)
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_epochs = float('inf')
            steps_per_epoch = total_train_steps

        if self.training_args.eval_steps is None:
            logger.info(f"eval_steps is not set, evaluating at the end of each epoch")
            eval_steps = steps_per_epoch
        else:
            eval_steps = self.training_args.eval_steps
            
        if self.training_args.eval_generation_steps is None:
            eval_generation_steps = eval_steps
        else:
            eval_generation_steps = self.training_args.eval_generation_steps

        num_examples = total_train_steps * self.train_batch_size * self.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info("  Instantaneous batch size per device =" f" {self.per_device_train_batch_size}")
        logger.info("  Gradient accumulation steps =" f" {self.gradient_accumulation_steps}")
        logger.info(
            f"  Total train batch size (w. parallel & distributed) = {self.train_batch_size * self.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {total_train_steps}")

        # Set up push to hub if needed
        if self.accelerator.is_main_process and self.training_args.push_to_hub:
            api = HfApi(token=self.training_args.hub_token)
            # Create repo (repo_name from args or inferred)
            repo_name = self.training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(self.training_args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(self.training_args.output_dir, ".gitignore"), "w+") as gitignore:
                pass
        
        # ======================== Training ================================
        train_time = 0
        train_start = time.time()
        steps_trained_progress_bar = tqdm(
            range(total_train_steps), 
            desc="Train steps ... ", 
            position=0, 
            disable=not self.accelerator.is_local_main_process
        )
        continue_training = True
        
        # Update progress bar if resuming
        if cur_step > 0:
            steps_trained_progress_bar.update(cur_step)
        
        total_batched_samples = resume_step if resume_step is not None else 0
        self.model.train()

        for epoch in range(epochs_trained, num_epochs):
            with self.accelerator.local_main_process_first():
                self.vectorized_datasets["train"] = self.vectorized_datasets["train"].shuffle(self.training_args.seed)
                
            sampler = None
            if self.training_args.group_by_length:
                sampler = LengthGroupedSampler(
                    self.train_batch_size, 
                    lengths=self.vectorized_datasets["train"]["target_length"]
                )
                
            train_dataloader = DataLoader(
                self.vectorized_datasets["train"],
                collate_fn=self.data_collator,
                batch_size=self.per_device_train_batch_size,
                sampler=sampler,
                shuffle=not self.training_args.group_by_length,
                num_workers=self.training_args.dataloader_num_workers,
                pin_memory=self.training_args.dataloader_pin_memory,
            )
            train_dataloader = self.accelerator.prepare(train_dataloader)
            
            if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
                train_dataloader.dataset.set_epoch(epoch)

            if resume_step is not None:
                # Skip the first N batches in the dataloader when resuming from a checkpoint
                logger.info(f"  Skip first {resume_step} batches")
                train_dataloader = self.accelerator.skip_first_batches(train_dataloader, resume_step)
                resume_step = None
                self.accelerator.wait_for_everyone()

            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            train_iterator = iter(train_dataloader)
            num_steps_in_epoch = len(train_dataloader)
            remainder = num_steps_in_epoch % self.gradient_accumulation_steps
            remainder = remainder if remainder != 0 else self.gradient_accumulation_steps
            total_updates = math.ceil(num_steps_in_epoch / self.gradient_accumulation_steps)
            
            update_step = -1
            for _ in range(total_updates):
                update_step += 1
                
                # preload the total batch per step
                batch_samples = []
                num_batches_in_step = self.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                for _ in range(num_batches_in_step):
                    batch_samples += [next(train_iterator)]
                    
                # get num items in batch - if different than BOS and than -100
                num_items_in_batch = sum([
                    (batch["labels"].ne(self.audio_encoder_bos_token_id) | 
                     batch["labels"].ne(-100) | 
                     batch["labels"].ne(self.audio_encoder_eos_token_id)).sum((0,1))[0] 
                    for batch in batch_samples
                ])
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
                
                accumulated_loss = 0
                accumulated_metrics = None
                
                for i, batch in enumerate(batch_samples):
                    total_batched_samples += 1
                    ctx = self.model.no_sync if (i < len(batch_samples) - 1 and self.accelerator.num_processes > 1) else contextlib.nullcontext
                    
                    with ctx():
                        loss, train_metric = self.train_step(batch, num_items_in_batch)
                        self.accelerator.backward(loss)
                        
                        accumulated_loss += loss.detach()
                        if accumulated_metrics is None:
                            accumulated_metrics = {k: v.detach() for k, v in train_metric.items()}
                        else:
                            for k, v in train_metric.items():
                                accumulated_metrics[k] += v.detach()
                
                # Average the accumulated metrics
                accumulated_metrics = {k: v / num_batches_in_step for k, v in accumulated_metrics.items()}
                
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                steps_trained_progress_bar.update(1)
                cur_step += 1
                
                if cur_step % self.training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {accumulated_metrics['loss']}, Learning Rate:"
                        f" {self.lr_scheduler.get_last_lr()[0]})"
                    )
                    accumulated_metrics["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    log_metric(
                        self.accelerator,
                        metrics=accumulated_metrics,
                        learning_rate=self.lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                # save checkpoint and weights after each save_steps and at the end of training
                if (cur_step % self.training_args.save_steps == 0) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(
                        self.training_args.output_dir, 
                        f"checkpoint-{cur_step}-epoch-{epoch}"
                    )
                    # safe_serialization=False to avoid shared tensors saving issue (temporary fix)
                    self.accelerator.save_state(output_dir=intermediate_dir, safe_serialization=False)
                    self.accelerator.wait_for_everyone()
                    
                    if self.accelerator.is_main_process:
                        rotate_checkpoints(
                            self.training_args.save_total_limit, 
                            output_dir=self.training_args.output_dir, 
                            logger=logger
                        )

                        if cur_step == total_train_steps:
                            # un-wrap student model for save
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            unwrapped_model.save_pretrained(self.training_args.output_dir)

                        if self.training_args.push_to_hub:
                            api.upload_folder(
                                repo_id=repo_id,
                                folder_path=self.training_args.output_dir,
                                commit_message=f"Saving train state of step {cur_step}",
                                run_as_future=True,
                            )
                    self.accelerator.wait_for_everyone()

                if self.training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                    train_time += time.time() - train_start
                    # ======================== Evaluating ==============================
                    self.model.eval()
                    eval_metrics = []
                    eval_preds = []
                    eval_descriptions = []
                    eval_prompts = []
                    eval_start = time.time()

                    # release training input batch
                    batch = release_memory(batch)

                    validation_dataloader = DataLoader(
                        self.vectorized_datasets["eval"],
                        collate_fn=self.data_collator,
                        batch_size=self.per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=self.training_args.eval_dataloader_num_workers,
                        pin_memory=self.training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = self.accelerator.prepare(validation_dataloader)

                    for batch in tqdm(
                        validation_dataloader,
                        desc=f"Evaluating - Inference ...",
                        position=2,
                        disable=not self.accelerator.is_local_main_process,
                    ):
                        # Model forward
                        eval_metric = self.eval_step(batch)
                        eval_metric = self.accelerator.gather_for_metrics(eval_metric)
                        eval_metric = {
                            key: val.unsqueeze(0) if val.ndim == 0 else val 
                            for (key, val) in eval_metric.items()
                        }
                        eval_metrics.append(eval_metric)

                    if self.training_args.predict_with_generate and (cur_step % eval_generation_steps == 0 or cur_step == total_train_steps):
                        validation_dataloader = DataLoader(
                            self.vectorized_datasets["eval"],
                            collate_fn=self.data_collator,
                            batch_size=self.per_device_eval_batch_size,
                            drop_last=False,
                            num_workers=self.training_args.eval_dataloader_num_workers,
                            pin_memory=self.training_args.dataloader_pin_memory,
                        )
                        validation_dataloader = self.accelerator.prepare(validation_dataloader)
                        # generation
                        for batch in tqdm(
                            validation_dataloader,
                            desc=f"Evaluating - Generation ...",
                            position=2,
                            disable=not self.accelerator.is_local_main_process,
                        ):
                            generated_audios = self.generate_step(batch)
                            # Gather all predictions and targets
                            generated_audios, input_ids, prompts = self.accelerator.pad_across_processes(
                                (generated_audios, batch["input_ids"], batch["prompt_input_ids"]),
                                dim=1, 
                                pad_index=0
                            )
                            generated_audios, input_ids, prompts = self.accelerator.gather_for_metrics(
                                (generated_audios, input_ids, prompts)
                            )
                            eval_preds.extend(generated_audios.to("cpu"))
                            eval_descriptions.extend(input_ids.to("cpu"))
                            eval_prompts.extend(prompts.to("cpu"))

                    eval_time = time.time() - eval_start
                    # normalize eval metrics
                    eval_metrics = {
                        key: torch.mean(torch.cat([d[key] for d in eval_metrics])).to("cpu") 
                        for key in eval_metrics[0]
                    }

                    # compute metrics
                    metrics_desc = ""
                    if self.training_args.predict_with_generate and (cur_step % eval_generation_steps == 0 or cur_step == total_train_steps):
                        if self.accelerator.is_local_main_process:
                            (
                                metric_values,
                                pred_descriptions,
                                pred_prompts,
                                audios,
                                transcriptions,
                                si_sdr_measures,
                            ) = self.compute_metrics(
                                eval_preds,
                                eval_descriptions,
                                eval_prompts,
                                self.accelerator.device,
                                self.training_args.compute_clap_similarity_metric,
                                self.training_args.compute_noise_level_metric,
                                self.training_args.noise_level_to_compute_clean_wer,
                            )
                            eval_metrics.update(metric_values)
                            metrics_desc = " ".join([f"Eval {key}: {value} |" for key, value in metric_values.items()])
                        self.accelerator.wait_for_everyone()

                    # Print metrics and update progress bar
                    if self.accelerator.is_local_main_process:
                        steps_trained_progress_bar.write(
                            f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                            f" {metrics_desc})"
                        )

                    log_metric(
                        self.accelerator,
                        metrics=eval_metrics,
                        train_time=eval_time,
                        step=cur_step,
                        epoch=epoch,
                        prefix="eval",
                    )

                    # release eval batch and relax metrics
                    eval_metrics, eval_preds, eval_descriptions, eval_prompts, batch, eval_metric = release_memory(
                        eval_metrics, eval_preds, eval_descriptions, eval_prompts, batch, eval_metric
                    )
                    if self.training_args.predict_with_generate and (cur_step % eval_generation_steps == 0 or cur_step == total_train_steps):
                        generated_audios, input_ids, prompts = release_memory(generated_audios, input_ids, prompts)

                    # train mode
                    self.model.train()

                    # flush the train metrics
                    train_start = time.time()

                # break condition
                if cur_step == total_train_steps:
                    continue_training = False
                    break

            if not continue_training:
                break

        self.accelerator.end_training()
        return cur_step