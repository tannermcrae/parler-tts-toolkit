import logging
import os
import torch
import inspect
from tqdm import tqdm
import datasets
from datasets import DatasetDict, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from accelerate import skip_first_batches

from src.parler_tts import build_delay_pattern_mask
from utils import (
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)
from data import combine_datasets, DataCollatorEncodecWithPadding

logger = logging.getLogger(__name__)

# These standalone functions need to live outside the DataProcessor class
# Otherwise the self() invocations will be copied for each process in the dataset.filter() 
# Causing pickling errors.
def create_length_filter(min_length, max_length):
    def filter_fn(length):
        return min_length < length < max_length
    return filter_fn

def create_max_token_length_filter(max_length):
    def filter_fn(tokens):
        return len(tokens) < max_length
    return filter_fn

def create_text_length_filter(max_length):
    def filter_fn(text):
        return len(text) < max_length
    return filter_fn

def create_tokenizer_function(description_tokenizer, prompt_tokenizer):
    def tokenize_text(description, prompt):
        batch = {}
        batch["input_ids"] = description_tokenizer(description.strip())["input_ids"]
        batch["prompt_input_ids"] = prompt_tokenizer(prompt.strip())["input_ids"]
        return batch
    return tokenize_text

# Add this at the module level
def postprocess_audio_labels(labels, bos_token_id, eos_token_id, num_codebooks):
    """Process audio labels to add BOS/EOS tokens and build delay pattern mask.
    
    Args:
        labels: Tensor of shape (codebooks, seq_len)
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID (used as padding)
        num_codebooks: Number of codebooks in the model
    
    Returns:
        Dictionary with processed labels
    """
    # Create BOS tokens tensor (1, codebooks, 1)
    bos_labels = torch.ones((1, num_codebooks, 1), dtype=torch.int) * bos_token_id
    
    # (1, codebooks, seq_len)
    labels = torch.tensor(labels).unsqueeze(0)
    # add bos
    labels = torch.cat([bos_labels, labels], dim=-1)

    labels, delay_pattern_mask = build_delay_pattern_mask(
        labels,
        bos_token_id=bos_token_id,
        pad_token_id=eos_token_id,
        max_length=labels.shape[-1] + num_codebooks,
        num_codebooks=num_codebooks,
    )

    # the first ids of the delay pattern mask are precisely labels, we use the rest of the labels mask
    # to take care of EOS
    labels = torch.where(delay_pattern_mask == -1, eos_token_id, delay_pattern_mask)

    # the first timestamp is associated to a row full of BOS, let's get rid of it
    # we also remove the last timestampts (full of PAD)
    output = {"labels": labels[:, 1:]}
    return output

class DataProcessor:
    def __init__(
        self,
        data_args,
        model_args,
        training_args,
        feature_extractor,
        prompt_tokenizer,
        description_tokenizer,
        config,
        model,
        accelerator,
    ):
        """Initialize the data processor with required components."""
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.feature_extractor = feature_extractor
        self.prompt_tokenizer = prompt_tokenizer
        self.description_tokenizer = description_tokenizer
        self.config = config
        self.model = model
        self.accelerator = accelerator
        self.sampling_rate = feature_extractor.sampling_rate
        
        # Extract model configs
        self.audio_encoder_bos_token_id = model.generation_config.decoder_start_token_id
        self.audio_encoder_eos_token_id = config.decoder.eos_token_id
        self.audio_encoder_pad_token_id = config.decoder.pad_token_id
        self.num_codebooks = model.decoder.config.num_codebooks
        self.max_length = model.generation_config.max_length
        
        # Derived values based on args
        self.padding = "max_length" if data_args.pad_to_max_length else "longest"
        self.max_target_length = int(data_args.max_duration_in_seconds * self.sampling_rate)
        self.min_target_length = int(data_args.min_duration_in_seconds * self.sampling_rate)
        self.target_audio_column_name = data_args.target_audio_column_name
        self.description_column_name = data_args.description_column_name
        self.prompt_column_name = data_args.prompt_column_name
        self.feature_extractor_input_name = feature_extractor.model_input_names[0]
        self.num_workers = data_args.preprocessing_num_workers
        self.bandwidth = model_args.bandwidth
        
    def prepare_datasets(self):
        """Main method to prepare and return vectorized datasets."""
        # Check if datasets are already preprocessed and saved to disk
        dataset_was_precomputed = (self.data_args.save_to_disk is not None and 
                                  os.path.exists(self.data_args.save_to_disk) and 
                                  len(os.listdir(self.data_args.save_to_disk)) > 0)
        
        if dataset_was_precomputed:
            logger.info(f"Loading preprocessed datasets from {self.data_args.save_to_disk}")
            with self.accelerator.local_main_process_first():
                vectorized_datasets = datasets.load_from_disk(self.data_args.save_to_disk)
        else:
            logger.info("Processing datasets from scratch")
            vectorized_datasets = self._process_datasets_from_scratch()
            
            if self.data_args.save_to_disk is not None:
                self._save_to_disk(vectorized_datasets)
        
        # Apply additional filtering and processing
        vectorized_datasets = self._apply_post_processing(vectorized_datasets)
        
        # Compute audio_max_length for padding if needed
        audio_max_length = None
        if self.padding == "max_length":
            audio_max_length = self._compute_audio_max_length(vectorized_datasets)
            
        return vectorized_datasets, audio_max_length
        
    def _process_datasets_from_scratch(self):
        """Process raw datasets from scratch."""
        raw_datasets = self._load_raw_datasets()
        vectorized_datasets = self._tokenize_datasets(raw_datasets)
        vectorized_datasets = self._encode_audio(vectorized_datasets, raw_datasets)
        return vectorized_datasets
        
    def _load_raw_datasets(self):
        """Load raw datasets."""
        raw_datasets = DatasetDict()
        
        columns_to_keep = {
            "target_audio_column_name": self.data_args.target_audio_column_name,
            "prompt_column_name": self.data_args.prompt_column_name,
        }
        if self.data_args.description_column_name is not None:
            columns_to_keep["description_column_name"] = self.data_args.description_column_name
        
        if self.training_args.do_train:
            raw_datasets['train'] = combine_datasets(
                accelerator=self.accelerator,
                dataset_source='s3',
                output_dir=self.data_args.dataset_output_dir,
                dataset_name=self.data_args.train_dataset_name,
                dataset_config_name=self.data_args.train_dataset_config_name,
                metadata_dataset_name=self.data_args.train_metadata_dataset_name,
                splits=self.data_args.train_split_name,
                sampling_rate=self.sampling_rate,
                audio_column_name=self.data_args.target_audio_column_name,
                logger=logger
            )
            
            # Validate all required columns exist
            for key in columns_to_keep:
                if columns_to_keep[key] not in raw_datasets["train"].column_names:
                    raise ValueError(
                        f"--{key} '{columns_to_keep[key]}' not found in dataset '{self.data_args.train_dataset_name}'."
                        f" Make sure to set `--{key}` to the correct audio column - one of"
                        f" {', '.join(raw_datasets['train'].column_names)}."
                    )
            
            # Apply max_train_samples limit if specified
            if self.data_args.max_train_samples is not None:
                raw_datasets["train"] = raw_datasets["train"].select(range(self.data_args.max_train_samples))
        
        if self.training_args.do_eval:
            raw_datasets['eval'] = combine_datasets(
                accelerator=self.accelerator,
                dataset_source='s3',
                output_dir=self.data_args.dataset_output_dir,
                dataset_name=self.data_args.eval_dataset_name if self.data_args.eval_dataset_name else self.data_args.train_dataset_name,
                dataset_config_name=self.data_args.eval_dataset_config_name if self.data_args.eval_dataset_config_name else self.data_args.train_dataset_config_name,
                metadata_dataset_name=self.data_args.eval_metadata_dataset_name,
                splits=self.data_args.eval_split_name,
                sampling_rate=self.sampling_rate,
                audio_column_name=self.data_args.target_audio_column_name,
                logger=logger
            )
            
            # Apply max_eval_samples limit if specified
            if self.data_args.max_eval_samples is not None:
                with self.accelerator.local_main_process_first():
                    raw_datasets["eval"] = (
                        raw_datasets["eval"].shuffle(seed=self.training_args.seed).select(range(self.data_args.max_eval_samples))
                    )
        
        return raw_datasets
    
    def _tokenize_datasets(self, raw_datasets):
        """Tokenize the texts in datasets."""
        # Filter on text length if specified
        if self.description_column_name is not None and self.data_args.max_text_length is not None:
            max_text_length = self.data_args.max_text_length  # Extract this value
            column_name = self.description_column_name  # Extract column name
            
            with self.accelerator.local_main_process_first():
                # Create a function that doesn't capture self
                def text_filter_fn(examples):
                    return DataProcessor._filter_by_text_length(
                        examples[column_name], max_text_length
                    )
                
                # Use the function directly, not a lambda
                raw_datasets = raw_datasets.filter(
                    text_filter_fn,
                    num_proc=self.num_workers,
                    desc="Filtering by text length",
                )
        
        # Process texts through tokenizers
        with self.accelerator.local_main_process_first():
            # Create parameters for tokenization
            desc_tokenizer = self.description_tokenizer
            prompt_tokenizer = self.prompt_tokenizer
            desc_column = self.description_column_name
            prompt_column = self.prompt_column_name
            
            # Create tokenize function that doesn't capture self
            def tokenize_fn(examples):
                return DataProcessor._tokenize_text(
                    examples, 
                    desc_tokenizer,
                    prompt_tokenizer,
                    desc_column,
                    prompt_column
                )
            
            # Use the explicit function
            vectorized_datasets = raw_datasets.map(
                tokenize_fn,
                batched=False,
                remove_columns=next(iter(raw_datasets.values())).column_names,
                num_proc=self.num_workers,
                desc="preprocessing datasets",
            )
        
        return vectorized_datasets
    
    def _encode_audio(self, vectorized_datasets, raw_datasets):
        """Encode audio with the model's audio encoder."""
        logger.info("*** Encode target audio with encodec ***")
        
        # Get audio encoder from model
        audio_decoder = self.model.audio_encoder
        if self.training_args.torch_compile:
            audio_decoder = self.accelerator.prepare_model(audio_decoder, evaluation_mode=True)
        
        # Prepare data collator for encoding
        encoder_data_collator = DataCollatorEncodecWithPadding(
            self.feature_extractor,
            audio_column_name=self.target_audio_column_name,
            feature_extractor_input_name=self.feature_extractor_input_name,
            max_length=self.max_target_length,
            padding=self.padding,
        )
        encoder_signature = set(inspect.signature(audio_decoder.forward).parameters)
        
        # Encoding function
        def apply_audio_decoder(batch):
            len_audio = batch.pop("len_audio")
            audio_decoder.to(batch["input_values"].device).eval()
            if self.bandwidth is not None:
                batch["bandwidth"] = self.bandwidth
            elif "num_quantizers" in encoder_signature:
                batch["num_quantizers"] = self.num_codebooks
            elif "num_codebooks" in encoder_signature:
                batch["num_codebooks"] = self.num_codebooks
            elif "n_quantizers" in encoder_signature:
                batch["n_quantizers"] = self.num_codebooks

            with torch.no_grad():
                labels = audio_decoder.encode(**batch)["audio_codes"]
            output = {}
            output["len_audio"] = len_audio
            # (1, bsz, codebooks, seq_len) -> (bsz, seq_len, codebooks)
            output["labels"] = labels.squeeze(0).transpose(1, 2)

            # if `pad_to_max_length`, the maximum corresponding audio length of the current batch is max_duration*sampling_rate
            max_length = len_audio.max() if self.padding != "max_length" else self.max_target_length
            output["ratio"] = torch.ones_like(len_audio) * labels.shape[-1] / max_length
            return output
        
        # (1, codebooks, seq_len) where seq_len=1
        bos_labels = torch.ones((1, self.num_codebooks, 1)) * self.audio_encoder_bos_token_id
        
        # Postprocessing function to add BOS/EOS tokens and build delay pattern mask
        def create_postprocessor():
            bos_id = self.audio_encoder_bos_token_id
            eos_id = self.audio_encoder_eos_token_id
            num_cb = self.num_codebooks
            
            # Return a function that doesn't capture self
            def process_fn(labels):
                return postprocess_audio_labels(labels, bos_id, eos_id, num_cb)
            return process_fn
        
        # Create the processor function
        processor_fn = create_postprocessor()
        
        # Process each split
        for split in vectorized_datasets:
            data_loader = DataLoader(
                raw_datasets[split],
                batch_size=self.training_args.audio_encoder_per_device_batch_size,
                collate_fn=encoder_data_collator,
                num_workers=self.training_args.dataloader_num_workers,
                pin_memory=True,
            )
            data_loader = self.accelerator.prepare(data_loader)
            total_inference_steps = len(data_loader)

            temp_save_dir = os.path.join(self.data_args.temporary_save_to_disk, split)
            os.makedirs(temp_save_dir, exist_ok=True)
            
            start_step = get_last_codec_checkpoint_step(temp_save_dir)
            self.accelerator.wait_for_everyone()
            
            if start_step > 0:
                logger.info(f"Resuming {split} from step {start_step}")
                # efficiently skip the first n batches
                start_step += 1
                data_loader = skip_first_batches(data_loader, start_step)

            all_generated_labels = []
            all_lens = []
            if start_step < total_inference_steps:
                for i, batch in enumerate(tqdm(data_loader, disable=not self.accelerator.is_local_main_process)):
                    cur_step = start_step + i
                    generate_labels = apply_audio_decoder(batch)
                    generate_labels = self.accelerator.pad_across_processes(generate_labels, dim=1, pad_index=0)
                    generate_labels = self.accelerator.gather_for_metrics(generate_labels)

                    if self.accelerator.is_main_process:
                        lab = generate_labels["labels"].cpu().transpose(1, 2).to(torch.int16)
                        rat = generate_labels["ratio"].cpu().squeeze(1)
                        lens = generate_labels["len_audio"].cpu().squeeze(1)
                        lab = [l[:, : int(ratio * length)] for (l, ratio, length) in zip(lab, rat, lens)]

                        all_generated_labels.extend(lab)
                        all_lens.extend(lens)

                        if ((cur_step + 1) % self.data_args.save_codec_steps == 0) or (
                            cur_step == total_inference_steps - 1
                        ):
                            tmp_labels = Dataset.from_dict({"labels": all_generated_labels, "target_length": all_lens})
                            tmp_labels = tmp_labels.map(
                                processor_fn,
                                num_proc=self.num_workers,
                                input_columns=["labels"],
                                desc="Postprocessing labeling",
                            )
                            save_codec_checkpoint(temp_save_dir, tmp_labels, cur_step)
                            all_generated_labels = []
                            all_lens = []

                self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process and len(all_generated_labels) > 0:
                tmp_labels = Dataset.from_dict({"labels": all_generated_labels, "target_length": all_lens})
                tmp_labels = tmp_labels.map(
                    processor_fn,
                    num_proc=self.num_workers,
                    input_columns=["labels"],
                    desc="Postprocessing labeling",
                )
                save_codec_checkpoint(temp_save_dir, tmp_labels, cur_step)
                all_generated_labels = []
                all_lens = []
            self.accelerator.wait_for_everyone()

            del all_generated_labels
            self.accelerator.wait_for_everyone()

            with self.accelerator.local_main_process_first():
                tmp_labels = load_all_codec_checkpoints(temp_save_dir).select(
                    range(len(vectorized_datasets[split]))
                )
                logger.info(f"Concatenating {split}: {tmp_labels} with {vectorized_datasets[split]}")
                vectorized_datasets[split] = concatenate_datasets([vectorized_datasets[split], tmp_labels], axis=1)

        self.accelerator.free_memory()
        del generate_labels, all_lens
        
        return vectorized_datasets
    
    @staticmethod
    def is_audio_in_length_range(length, min_length, max_length):
        """Filter function to check if audio length is within range."""
        return length > min_length and length < max_length

    def _apply_post_processing(self, vectorized_datasets):
        """Apply post-processing filters to datasets."""
        with self.accelerator.local_main_process_first():
            # Extract parameters first
            min_length = self.min_target_length
            max_length = self.max_target_length
            
            # Create function that doesn't capture self
            def audio_filter_fn(length):
                return DataProcessor._filter_by_audio_length(
                    length, min_length, max_length
                )
            
            # Filter by audio length using static method
            vectorized_datasets = vectorized_datasets.filter(
                audio_filter_fn,
                num_proc=self.num_workers,
                input_columns=["target_length"],
            )
            
            # Filter by description token length if specified
            if self.description_column_name is not None and self.data_args.max_description_token_length is not None:
                max_token_length = self.data_args.max_description_token_length
                
                def token_filter_fn(tokens):
                    return DataProcessor._filter_by_token_length(
                        tokens, max_token_length
                    )
                
                vectorized_datasets = vectorized_datasets.filter(
                    token_filter_fn,
                    num_proc=self.num_workers,
                    input_columns=["input_ids"],
                )
            
            # Filter by prompt token length if specified  
            if self.data_args.max_prompt_token_length is not None:
                max_prompt_length = self.data_args.max_prompt_token_length
                
                def prompt_filter_fn(tokens):
                    return DataProcessor._filter_by_token_length(
                        tokens, max_prompt_length
                    )
                
                vectorized_datasets = vectorized_datasets.filter(
                    prompt_filter_fn,
                    num_proc=self.num_workers,
                    input_columns=["prompt_input_ids"],
                )
                
        return vectorized_datasets
    
    def _compute_audio_max_length(self, vectorized_datasets):
        """Compute maximum audio length for padding."""
        audio_max_length = max(vectorized_datasets["train"]["target_length"])
        with self.accelerator.local_main_process_first():
            max_sample = vectorized_datasets["train"].filter(
                lambda x: x == audio_max_length,
                num_proc=self.num_workers,
                input_columns=["target_length"],
            )
        audio_max_length = max([len(l[0]) for l in max_sample["labels"]])
        return audio_max_length
    
    def _save_to_disk(self, vectorized_datasets):
        """Save processed datasets to disk."""
        if self.accelerator.is_main_process:
            os.makedirs(self.data_args.save_to_disk, exist_ok=True)
            vectorized_datasets.save_to_disk(self.data_args.save_to_disk)
        self.accelerator.wait_for_everyone()
        logger.info(f"Dataset saved at {self.data_args.save_to_disk}")
        
    def add_target_lengths_for_grouping(self, vectorized_datasets):
        """Add target lengths for length-based grouping."""
        if self.training_args.group_by_length:
            with self.accelerator.local_main_process_first():
                vectorized_datasets = vectorized_datasets.map(
                    DataProcessor._add_target_lengths,
                    num_proc=self.num_workers,
                    input_columns=["target_length", "prompt_input_ids", "input_ids"],
                )
                
        return vectorized_datasets

    # --- STATIC METHODS FOR PROCESSING ---
    
    @staticmethod
    def _tokenize_text(examples, description_tokenizer, prompt_tokenizer, description_column, prompt_column):
        """Static tokenization function."""
        batch = {}
        batch["input_ids"] = description_tokenizer(examples[description_column].strip())["input_ids"]
        batch["prompt_input_ids"] = prompt_tokenizer(examples[prompt_column].strip())["input_ids"]
        return batch
    
    @staticmethod
    def _filter_by_text_length(text, max_length):
        """Static text length filter."""
        return len(text) < max_length
    
    @staticmethod
    def _filter_by_token_length(tokens, max_length):
        """Static token length filter."""
        return len(tokens) < max_length
    
    @staticmethod
    def _filter_by_audio_length(length, min_length, max_length):
        """Static audio length filter."""
        return min_length < length < max_length
    
    @staticmethod
    def _process_labels(labels, params):
        """Static method for processing audio labels."""
        bos_token_id, eos_token_id, num_codebooks = params
        return postprocess_audio_labels(labels, bos_token_id, eos_token_id, num_codebooks)
    
    @staticmethod
    def _generate_labels(features, model, feature_extractor, processor):
        """Static function for generating labels."""
        # Extract audio
        audio = features["audio"]
        
        # Process with model and feature extractor
        input_features = processor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").input_features
        
        with torch.no_grad():
            # Generate labels with the model
            outputs = model.encode(input_features.to(model.device))
        
        # Return processed outputs
        return {"labels": outputs, "target_length": len(audio)}
    
    @staticmethod
    def _add_target_lengths(target_length, prompt, description):
        """Static method for computing combined target lengths."""
        return {"target_length": target_length + len(prompt) + len(description)}
