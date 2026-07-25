"""
Main Architechture for Vision Transformer Model
Vision path: processor --> encoder --> projector
Text path: tokenizer --> embedder
"""

# import Torch and Huggingface
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLm,
)

class DoraVisionLanguageModel(nn.Module):
    def __init__(
        self,
        vision_ckpt: str,
        language_ckpt: str,
        modality_input_dim: int = 768,
        modality_output_dim: int = 576,
    ):
        super().__init__()
        
        # vision tower
        self.vision_encoder = AutoModel.from_pretrained(vision_ckpt).vision_model
        self.vision_processor = AutoProcessor.from_pretrained(vision_ckpt)
        self.modality_projector = nn.Linear(
            modality_input_dim,
            modality_output_dim,
            bias=False,
        )

        # language tower
        self.tokenizer = AutoTokenizer.from_pretrained(language_ckpt)
        self.llm = AutoModelForCausalLM.from_pretrained(language_ckpt)

    def forward(self, text, images):
        processed_image = self.vision_precessor(
            image = [images],
            return_tensors = "pt",
        ).to(self.llm.device)
        image_embd = self.vision_encoder(**processed_image).last_hidden_state
        
        #[B, N_image_tokens, d_model] (eg: [B, 768, d_model])
        image_embd = self.modality_projector(image_embd).to(dtype=self.llm.device.dtype)
        
        # process pad tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        input_ids = self.tokenizer(
            text,
            return_tensors = "pt",
            padding = True,
            truncation=True,
        ).input_ids.to(self.llm.device) # Shape: [B, N_text_tokens]
        
        # Shape: [B, N_text_tokens, d_model]
        token_embd = self.llm.model.embed_tokens(input_ids)

        # [batch_size, num_image_tokens + num_text_tokens, embedding_dim]
        #dim=1: [image, Text]
        combined_embd = torch.cat((image_embd, token_embd), dim=1)

        # Shape: [B, N_image_tokens + N_text_tokens, vocab_size]
        logits = self.llm(inputs_embeds=combined_embd).logits
        
        # [:, 768:, :] - remove image_embd because I only take from image_embed to end --> remove image_embd
        text_logits = logits[:, image_embd.size(1):, :] # Shape: [B, N_text_tokens, vocab_size]
        
        # from text to image_embd
        # 3D - [B, Tokens, vocab_size]
        shift_logits = text_logits[:, :-1, :].contiguous() # remove the last token
        shift_labels = input_ids[:, 1:].contiguous() # remove the first label

        # flatten everything before entering the loss function
        # logits 3D -> 2D [B * (N_text_tokens - 1), vocab_size]
        # labels 2D -> 1D [B * (N_text_tokens - 1)]
        loss.F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        return logits, loss

