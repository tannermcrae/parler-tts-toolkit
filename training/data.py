import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer
import os

@dataclass
class DataCollatorEncodecWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch or
    to `max_length` if `max_length` is set and `padding=max_length`.
    """

    feature_extractor: AutoFeatureExtractor
    audio_column_name: str
    feature_extractor_input_name: Optional[str] = "input_values"
    max_length: Optional[int] = None
    padding: Optional[str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        audios = [feature[self.audio_column_name]["array"] for feature in features]
        len_audio = [len(audio) for audio in audios]
        if self.max_length is not None:
            audios = [audio[: min(l, self.max_length)] for audio, l in zip(audios, len_audio)]

        # since resampling has already been performed in the 'load_multiple_datasets' function,
        # a fixed sampling_rate(44100hz) is passed to the feature_extractor.
        sampling_rate = self.feature_extractor.sampling_rate
        batch = self.feature_extractor(
            audios, sampling_rate=sampling_rate, return_tensors="pt", padding=self.padding, max_length=self.max_length
        )
        batch["len_audio"] = torch.tensor(len_audio).unsqueeze(1)
        return batch


@dataclass
class DataCollatorParlerTTSWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        prompt_tokenizer (:class:`~transformers.AutoTokenizer`)
            The prompt_tokenizer used for proccessing the data.
        description_tokenizer (:class:`~transformers.AutoTokenizer`)
            The description_tokenizer used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    prompt_tokenizer: AutoTokenizer
    description_tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    prompt_max_length: Optional[int] = None
    description_max_length: Optional[int] = None
    audio_max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        labels = [torch.tensor(feature["labels"]).transpose(0, 1) for feature in features]
        # (bsz, seq_len, num_codebooks)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        if self.audio_max_length is not None and self.padding == "max_length":
            labels = torch.nn.functional.pad(
                labels, pad=(0, 0, 0, max(self.audio_max_length - labels.shape[1], 0)), value=-100
            )

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]

        input_ids = self.description_tokenizer.pad(
            input_ids,
            return_tensors="pt",
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.description_max_length,
        )

        batch = {"labels": labels, **input_ids}

        prompt_input_ids = [{"input_ids": feature["prompt_input_ids"]} for feature in features]
        prompt_input_ids = self.prompt_tokenizer.pad(
            prompt_input_ids,
            return_tensors="pt",
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.prompt_max_length,
        )

        batch["prompt_input_ids"] = prompt_input_ids["input_ids"]
        if "attention_mask" in prompt_input_ids:
            batch["prompt_attention_mask"] = prompt_input_ids["attention_mask"]

        return batch
    
def download_audio_dataset_locally(dataset: Dataset, output_dir: str) -> Dataset:
    """
    Downloads audio files from a Hugging Face dataset to local disk and updates the paths.
    """
    import os
    import soundfile as sf
    from tqdm import tqdm
    
    # Create output directory if it doesn't exist and get its absolute path
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    def process_and_save_audio(example, idx):
        # Extract filename from S3 path
        filename = example['audio']['path'].split('/')[-1]
        
        # Create local path
        local_path = os.path.join(output_dir, filename)
        
        # Save audio to disk
        sf.write(local_path, example['audio']['array'], example['audio']['sampling_rate'])
        
        # Update path to local file (create a new dict to avoid modifying the original)
        example['audio'] = {
            'path': local_path,
            'array': example['audio']['array'],
            'sampling_rate': example['audio']['sampling_rate']
        }
        return example
    
    # Process all examples and show progress
    updated_dataset = dataset.map(
        process_and_save_audio,
        with_indices=True,
        desc="Downloading audio files"
    )
    
    return updated_dataset




def pull_dataset(src: str, name: str, config: str = None, splits: Optional[str] = None) -> Dataset:
    if src == 'hf':
        dataset = load_dataset(name, config, split=splits)
    elif src == 's3':
        file_extension = name.split('.')[-1]
        dataset = load_dataset(file_extension, data_files=name)
        # If loading from file and we get a DatasetDict with 'train', extract that dataset
        if isinstance(dataset, datasets.DatasetDict) and 'train' in dataset:
            dataset = dataset['train']

    else:
        raise ValueError(f"Invalid dataset source: {src}")
    
    return dataset

def combine_datasets(
    accelerator: Accelerator,
    dataset_source: str, # or 'hf' or 's3'
    dataset_name: str,
    output_dir: str,
    dataset_config_name: str,
    metadata_dataset_name: str,
    splits: Optional[str] = None,
    sampling_rate: Optional[int] = None,
    audio_column_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Union[Dataset, IterableDataset]:

        
    dataset: Dataset = pull_dataset(dataset_source, dataset_name, dataset_config_name, splits)

    # Save the arrow artifacts to disk. HF lazy loads the audio so we need to do this for faster epochs.
    if dataset_source != 'local':

        if not output_dir:
            raise ValueError("output_dir must be provided if dataset_source is not local")

        dataset = download_audio_dataset_locally(dataset, output_dir)

    # Normalize audio sampling rate on dataset.
    if sampling_rate is not None and audio_column_name is not None:
        dataset = dataset.cast_column(audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate))

    # Get metadata dataset.
    metadata_dataset: Dataset = pull_dataset(dataset_source, metadata_dataset_name, dataset_config_name, splits)

    logger.info(f'Merging {dataset_name} - {splits} with {metadata_dataset_name} - {splits}')

    metadata_columns_to_remove = set(metadata_dataset.column_names).intersection(set(dataset.column_names))

    # Remove columns that are present in both datasets.
    metadata_columns_to_remove = set(metadata_dataset.column_names).intersection(set(dataset.column_names))
    metadata_dataset = metadata_dataset.remove_columns(metadata_columns_to_remove)

    # Concatenate datasets.
    dataset = concatenate_datasets([dataset, metadata_dataset], axis=1)

    return dataset