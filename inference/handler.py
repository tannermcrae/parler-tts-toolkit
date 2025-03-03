import os
import sys
import torch
import logging
import numpy as np
import io
import wave
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoTokenizer
import scipy
import json
import base64

# This gets picked up from the unarchived model when the torchserve container is running. 
# During archival, we package up src/parler_tts with the files and the source directory
# is placed on the python path.
from parler_tts import ParlerTTSForConditionalGeneration

logger = logging.getLogger(__name__)

class TTSHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        
    def initialize(self, context):
        """Initialize model and tokenizer."""
        logger.info("Initializing model...")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Directory contents: {os.listdir('.')}")
            
        try:
            self.manifest = context.manifest
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            
            # Set up device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logger.info(f"Selected device: {self.device}")
            if self.device.type == "cpu":
                logger.warning("WARNING: Using CPU despite being on a GPU instance!")
                logger.warning("This might indicate a CUDA configuration issue.")
            
            # Initialize model with detailed logging
            logger.info("Loading model...")
            
            # Initialize model
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            self.model.eval()
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "parler-tts/parler-tts-mini-v1",
                padding_side="left"
            )
            
            logger.info(f"Initialized model on device: {self.device}")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def preprocess(self, requests):
        """Process input data."""
        data = requests[0].get("data")
        if data is None:
            data = requests[0].get("body")
        
        if isinstance(data, (bytes, bytearray)):
            data = data.decode('utf-8')
            
        if isinstance(data, str):
            data = json.loads(data)
            
        inputs = data if isinstance(data, list) else [data]
        
        texts = []
        descriptions = []
        for item in inputs:
            texts.append(item.get("text", ""))
            descriptions.append(item.get("description", ""))
            
        return texts, descriptions

    def ping(self):
        """Simple health check that returns 200 status code."""
        return {
            "status": "Healthy",
            "code": 200
        }

    def inference(self, inputs):
        """Run model inference."""
        texts, descriptions = inputs
        
        with torch.no_grad():
            input_tokens = self.tokenizer(
                descriptions,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            prompt_tokens = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            generation = self.model.generate(
                input_ids=input_tokens.input_ids,
                attention_mask=input_tokens.attention_mask,
                prompt_input_ids=prompt_tokens.input_ids,
                prompt_attention_mask=prompt_tokens.attention_mask,
                do_sample=True,
                return_dict_in_generate=True,
            )
            
            audio_arrays = []
            for i in range(len(texts)):
                audio = generation.sequences[i, :generation.audios_length[i]]
                audio_array = audio.to(torch.float32).cpu().numpy().squeeze()
                audio_arrays.append(audio_array)
            
            return audio_arrays

    def postprocess(self, inference_output):
        """Convert audio arrays to base64 strings."""
        encoded_arrays = []
        for audio_array in inference_output:
            # Convert to float32 if it's not already.  Important for consistency.
            audio_array = audio_array.astype(np.float32)
            audio_base64 = base64.b64encode(audio_array.tobytes()).decode('utf-8')
            encoded_arrays.append(audio_base64)
            
        return encoded_arrays
    
