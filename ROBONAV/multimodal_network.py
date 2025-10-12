# multimodal_fusion.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel, CLIPProcessor
from typing import List, Optional, Union
from PIL import Image
import numpy as np

# -----------------------------
# AI2-THOR action label set
# -----------------------------
# You can expand this list, but keep ordering stable; num_actions is derived from it.
AI2THOR_ACTION_LABELS: List[str] = [
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "DropHandObject",
    "SliceObject",
]

class MultimodalFusionModel(nn.Module):
    """
    Image + prompt → (action logits | generated text | VQA text | progress)
    Modes:
      - action:    next-action classification over AI2-THOR label set
      - description: text generation for next-step instruction
      - vqa:       text generation answering a question about the scene
      - progress:  scalar progress estimate in [0,1]
    """

    def __init__(
        self,
        text_decoder_name: str = "distilgpt2",
        image_encoder_name: str = "openai/clip-vit-base-patch32",
        action_labels: Optional[List[str]] = None,
        cross_attn_heads: int = 8,
    ):
        super().__init__()

        # ------- Backbones -------
        self.image_encoder = CLIPModel.from_pretrained(image_encoder_name)
        self.image_processor = CLIPProcessor.from_pretrained(image_encoder_name)

        self.text_decoder = AutoModelForCausalLM.from_pretrained(text_decoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_decoder_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------- Dims -------
        self.image_feature_dim = 512  # CLIP ViT-B/32 pooled embedding dim
        self.text_hidden_dim = self.text_decoder.config.hidden_size
        self.text_vocab_size = self.text_decoder.config.vocab_size

        # ------- Fusion / attention -------
        self.fusion_projection = nn.Linear(self.image_feature_dim, self.text_hidden_dim)
        self.gate = nn.Parameter(torch.ones(1, 1, self.text_hidden_dim))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.text_hidden_dim,
            num_heads=cross_attn_heads,
            batch_first=True,
        )

        self.context_adapter = nn.Sequential(
            nn.Linear(self.text_hidden_dim, self.text_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.text_hidden_dim, self.text_hidden_dim),
        )

        # Learnable modality embeddings
        self.image_modality_embedding = nn.Parameter(
            torch.randn(1, 1, self.text_hidden_dim) * 0.02
        )
        self.text_modality_embedding = nn.Parameter(
            torch.randn(1, 1, self.text_hidden_dim) * 0.02
        )

        # ------- Heads -------
        self.action_labels = action_labels if action_labels is not None else AI2THOR_ACTION_LABELS
        self.num_actions = len(self.action_labels)

        self.action_predictor = nn.Sequential(
            nn.Linear(self.text_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
        )

        self.progress_encoder = nn.LSTM(
            input_size=self.text_hidden_dim,
            hidden_size=128,
            batch_first=True,
        )
        self.progress_head = nn.Linear(128, 1)  # logits; sigmoid at inference

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_logits = nn.BCEWithLogitsLoss()

        # Helpful maps
        self.action_to_id = {a: i for i, a in enumerate(self.action_labels)}
        self.id_to_action = {i: a for i, a in enumerate(self.action_labels)}

        # Diagnostics
        print("📊 Model Dimensions:")
        print(f"  Image feature dim: {self.image_feature_dim}")
        print(f"  Text hidden dim:   {self.text_hidden_dim}")
        print(f"  Text vocab size:   {self.text_vocab_size}")
        print(f"  #Actions:          {self.num_actions}")

    # -----------------------------
    # Utilities
    # -----------------------------
    def _encode_images(
        self,
        images: Union[List[Image.Image], torch.Tensor, np.ndarray],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns CLIP image features of shape [B, image_feature_dim]
        Accepts:
          - list of PIL.Image
          - torch.Tensor/np.ndarray either raw images or already preprocessed pixel_values
        """
        if isinstance(images, list) and all(isinstance(im, Image.Image) for im in images):
            batch = self.image_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = self.image_encoder.get_image_features(**batch)  # [B, D]
        else:
            # Assume pixel_values already prepared: [B, 3, H, W]
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)
            pixel_values = images.to(device)
            with torch.no_grad():
                feats = self.image_encoder.get_image_features(pixel_values=pixel_values)  # [B, D]
        return feats  # [B, 512]

    def _encode_prompt_tokens(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        tok_embeds = self.text_decoder.transformer.wte(enc["input_ids"])  # [B, Lp, H]
        return tok_embeds

    def _fuse_image_text(self, img_feats: torch.Tensor, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """
        img_feats: [B, 512] → proj → [B, H]
        prompt_embeddings: [B, Lp, H]
        Returns: fused_embeddings [B, Lp, H]
        """
        img_ctx = self.fusion_projection(img_feats)  # [B, H]
        img_ctx = img_ctx.unsqueeze(1)  # [B, 1, H]
        img_ctx = img_ctx * torch.sigmoid(self.gate) + self.image_modality_embedding  # [B, 1, H]

        txt = prompt_embeddings + self.text_modality_embedding  # [B, Lp, H]

        fused, _ = self.cross_attention(query=txt, key=img_ctx, value=img_ctx)  # [B, Lp, H]
        fused = self.context_adapter(fused)  # [B, Lp, H]
        return fused

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(
        self,
        images: Union[List[Image.Image], torch.Tensor, np.ndarray],
        prompts: List[str],
        targets: Optional[torch.Tensor] = None,
        mode: str = "action",
        max_length: int = 64,
        temperature: float = 0.7,
    ):
        """
        Args:
          images: list of PIL or preprocessed pixel_values [B,3,H,W]
          prompts: list[str] (B)
          targets:
            - mode='action': LongTensor [B] (class ids) for training
            - mode='description'/'vqa': LongTensor [B, L] token ids for teacher-forced LM training
            - mode='progress': FloatTensor [B] or [B,1] in [0,1] for training
          mode: 'action' | 'description' | 'vqa' | 'progress'
        Returns:
          - action: dict(loss?, logits:[B,A])
          - description/vqa: dict(loss?, texts? or lm_loss/logits?)
          - progress: dict(loss?, score:[B,1] or logits)
        """
        device = next(self.parameters()).device

        # 1) Encode
        img_feats = self._encode_images(images, device)              # [B, 512]
        prompt_embeds = self._encode_prompt_tokens(prompts, device)  # [B, Lp, H]

        # 2) Fuse
        fused = self._fuse_image_text(img_feats, prompt_embeds)      # [B, Lp, H]
        fused_pooled = fused.mean(dim=1)                              # [B, H] (simple pool)

        # 3) Branch by mode
        if mode == "action":
            logits = self.action_predictor(fused_pooled)  # [B, A]
            out = {"logits": logits}
            if targets is not None:
                loss = self.ce_loss(logits, targets.long())
                out["loss"] = loss
            return out

        elif mode in ("description", "vqa"):
            # If targets provided → teacher-forced LM loss on caption/answer
            if targets is not None:
                # targets are token ids [B, L]
                # Concatenate fused prompt context + target tokens as inputs_embeds
                target_embeds = self.text_decoder.transformer.wte(targets)  # [B, L, H]
                inputs_embeds = torch.cat([fused, target_embeds], dim=1)    # [B, Lp+L, H]

                # Mask out fused prompt positions in labels
                prefix_mask = torch.full(
                    (targets.size(0), fused.size(1)), -100, dtype=torch.long, device=device
                )
                labels = torch.cat([prefix_mask, targets], dim=1)  # [B, Lp+L]

                outputs = self.text_decoder(inputs_embeds=inputs_embeds, labels=labels)
                return {"loss": outputs.loss, "lm_logits": outputs.logits}

            # Else → generate text from fused embeddings
            gen_ids = self.text_decoder.generate(
                inputs_embeds=fused,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            return {"texts": texts}

        elif mode == "progress":
            # Sequence → LSTM → scalar progress
            seq_enc, _ = self.progress_encoder(fused)  # [B, Lp, 128]
            pooled = seq_enc.mean(dim=1)               # [B, 128]
            logits = self.progress_head(pooled).squeeze(-1)  # [B]
            out = {"logits": logits}
            if targets is not None:
                targets = targets.float().view(-1)
                loss = self.bce_logits(logits, targets)
                out["loss"] = loss
            else:
                out["score"] = torch.sigmoid(logits)  # [B] in [0,1]
            return out

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'action', 'description', 'vqa', 'progress'.")

    # -----------------------------
    # Convenience helpers
    # -----------------------------
    def action_id(self, name: str) -> int:
        return self.action_to_id[name]

    def action_name(self, idx: int) -> str:
        return self.id_to_action[idx]
