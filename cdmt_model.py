# cdmt_model.py

import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
import timm


class CDMTConfig:
    def __init__(
        self,
        num_labels: int,
        d_model: int = 512,
        fusion_layers: int = 6,
        dropout: float = 0.2,
        temperature: float = 1.5,
        lambda_emo: float = 1.0,
        lambda_cdr: float = 0.6,
        lambda_dis: float = 0.4,
        lambda_margin: float = 0.3,
        margin: float = 0.4,
        use_had: bool = True,
    ):
        self.num_labels = num_labels
        self.d_model = d_model
        self.fusion_layers = fusion_layers
        self.dropout = dropout
        self.temperature = temperature
        self.lambda_emo = lambda_emo
        self.lambda_cdr = lambda_cdr
        self.lambda_dis = lambda_dis
        self.lambda_margin = lambda_margin
        self.margin = margin
        self.use_had = use_had


class FusionTransformer(nn.Module):
    """
    Simple multimodal fusion transformer using TransformerEncoder.
    Visual + text tokens are concatenated; we keep a modality mask
    so that downstream analyses know which tokens came from where.
    """

    def __init__(self, d_model: int, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Simple learned query for attention pooling
        self.query = nn.Parameter(torch.randn(d_model))

        # Lightweight gating coefficients for modalities
        # gamma_v, gamma_t ∈ [0,1]
        self.gate_v = nn.Parameter(torch.tensor(0.5))
        self.gate_t = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x_v: torch.Tensor,
        x_t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_v: [B, Nv, D]
            x_t: [B, Nt, D]
            attn_mask: optional [B, Nv+Nt] bool mask
        Returns:
            z: [B, D] pooled multimodal embedding
            x_all: [B, Nv+Nt, D] fused token embeddings
        """

        # Apply simple gating before fusion
        g_v = torch.sigmoid(self.gate_v)
        g_t = torch.sigmoid(self.gate_t)
        x_v = g_v * x_v
        x_t = g_t * x_t

        x_all = torch.cat([x_v, x_t], dim=1)  # [B, Nv+Nt, D]

        x_fused = self.encoder(x_all, src_key_padding_mask=attn_mask)

        # Attention pooling with a learnable query
        q = self.query.unsqueeze(0).unsqueeze(1)  # [1,1,D]
        q = q.expand(x_fused.size(0), -1, -1)     # [B,1,D]

        attn_scores = torch.matmul(q, x_fused.transpose(1, 2)) / math.sqrt(x_fused.size(-1))
        attn_weights = torch.softmax(attn_scores, dim=-1)      # [B,1,N]
        z = torch.matmul(attn_weights, x_fused).squeeze(1)     # [B,D]

        return z, x_fused


class CDMT(nn.Module):
    """
    Cognitive Dissonance-aware Multimodal Transformer (CDMT).
    - Visual encoder: ViT-B/16 from timm
    - Text encoder: BERT-base
    - Fusion: Transformer encoder
    - Heads: emotion classifier + disagreement head
    - Human affect embedding: learnable label embeddings
    """

    def __init__(self, config: CDMTConfig):
        super().__init__()
        self.cfg = config

        # Visual encoder: ViT-B/16
        self.vision = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,  # remove classifier head
        )
        d_v = self.vision.num_features

        # Text encoder: BERT-base
        self.text = BertModel.from_pretrained("bert-base-uncased")
        d_t = self.text.config.hidden_size

        # Projections to shared multimodal space
        self.proj_v = nn.Linear(d_v, config.d_model)
        self.proj_t = nn.Linear(d_t, config.d_model)

        # Fusion transformer
        self.fusion = FusionTransformer(
            d_model=config.d_model,
            n_layers=config.fusion_layers,
            n_heads=8,
            dropout=config.dropout,
        )

        # Emotion classifier head
        self.dropout = nn.Dropout(config.dropout)
        self.cls_head = nn.Linear(config.d_model, config.num_labels)

        # Human affect embedding table
        self.affect_embed = nn.Embedding(config.num_labels, config.d_model)

        # Optional HAD head
        if config.use_had:
            self.had_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 1),
            )
        else:
            self.had_head = None

        # Loss functions
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    # ---------- Encoding helpers ----------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,3,224,224]
        returns: [B,Nv,D] after projection
        """
        # timm ViT returns [B, Nv+1, d_v]; first token is CLS
        feats = self.vision.patch_embed(images)  # [B,Nv,d_v]
        # Add cls token + pos embed as in timm forward
        cls_token = self.vision.cls_token.expand(feats.shape[0], -1, -1)
        feats = torch.cat((cls_token, feats), dim=1)
        feats = feats + self.vision.pos_embed[:, : feats.size(1)]
        feats = self.vision.pos_drop(feats)
        for blk in self.vision.blocks:
            feats = blk(feats)
        feats = self.vision.norm(feats)  # [B, Nv+1, d_v]
        # Drop CLS or keep it as an additional token; here we keep all tokens
        feats_proj = self.proj_v(feats)  # [B,Nv+1,D]
        return feats_proj

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids, attention_mask: [B, Nt]
        returns: [B,Nt,D] after projection
        """
        outputs = self.text(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [B,Nt,d_t]
        feats_proj = self.proj_t(token_embeddings)    # [B,Nt,D]
        return feats_proj

    # ---------- Forward + losses ----------

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        tau_dis: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        batch must contain:
            images: [B,3,224,224]
            input_ids: [B,Nt]
            attention_mask: [B,Nt]
            labels: [B]
        """
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        B = images.size(0)

        # Encode
        x_v = self.encode_image(images)  # [B,Nv,D]
        x_t = self.encode_text(input_ids, attention_mask)  # [B,Nt,D]

        # Build padding mask for fusion (False = keep, True = pad)
        # Here, image tokens are fully valid; text tokens use attention_mask
        Nv = x_v.size(1)
        Nt = x_t.size(1)
        pad_mask_text = (attention_mask == 0)  # [B,Nt]
        pad_mask = torch.cat(
            [torch.zeros(B, Nv, dtype=torch.bool, device=images.device), pad_mask_text],
            dim=1,
        )

        # Fusion
        z, tokens_fused = self.fusion(x_v, x_t, attn_mask=pad_mask)  # z:[B,D]

        # Emotion prediction
        z_drop = self.dropout(z)
        logits = self.cls_head(z_drop)  # [B,C]
        probs = F.softmax(logits / self.cfg.temperature, dim=-1)

        # Classification loss
        loss_emo = self.ce(logits, labels)

        # Human affect embedding and CDS
        h = self.affect_embed(labels)   # [B,D]
        cds = torch.sum((z - h) ** 2, dim=-1)  # [B]
        loss_cdr = cds.mean()

        # Margin loss: distance to correct vs closest other center
        all_centers = self.affect_embed.weight  # [C,D]
        # [B, C, D]
        z_exp = z.unsqueeze(1).expand(-1, self.cfg.num_labels, -1)
        dist_all = torch.sum((z_exp - all_centers.unsqueeze(0)) ** 2, dim=-1)  # [B,C]
        # Distance to true center
        dist_true = dist_all.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        # Mask out true label and take min over others
        mask_true = F.one_hot(labels, num_classes=self.cfg.num_labels).bool()
        dist_others = dist_all.masked_fill(mask_true, float("inf"))
        dist_closest_other, _ = torch.min(dist_others, dim=1)  # [B]

        margin_term = F.relu(
            self.cfg.margin + dist_true - dist_closest_other
        )  # [B]
        loss_margin = margin_term.mean()

        # HAD head
        loss_dis = torch.tensor(0.0, device=images.device)
        dis_logits = None
        dis_probs = None
        y_dis = None

        if self.cfg.use_had and self.had_head is not None and tau_dis is not None:
            with torch.no_grad():
                y_dis = (cds > tau_dis).float()  # [B]
            dis_logits = self.had_head(z).squeeze(-1)  # [B]
            dis_probs = torch.sigmoid(dis_logits)
            loss_dis = self.bce(dis_logits, y_dis)

        loss_total = (
            self.cfg.lambda_emo * loss_emo
            + self.cfg.lambda_cdr * loss_cdr
            + self.cfg.lambda_margin * loss_margin
        )
        if self.cfg.use_had and tau_dis is not None:
            loss_total = loss_total + self.cfg.lambda_dis * loss_dis

        return {
            "loss_total": loss_total,
            "loss_emo": loss_emo,
            "loss_cdr": loss_cdr,
            "loss_margin": loss_margin,
            "loss_dis": loss_dis,
            "logits": logits,
            "probs": probs,
            "cds": cds,
            "dis_logits": dis_logits,
            "dis_probs": dis_probs,
            "y_dis": y_dis,
            "z": z,
        }
