import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint

import utils
from modules import BiLSTMLayer, TemporalConv
from modules.criterions import SeqKD


class LearnableCausalAttention(nn.Module):
    """Learnable spatial mask attention with a counterfactual branch."""

    def __init__(self, channels, reduction=8, dropout=0.0):
        super(LearnableCausalAttention, self).__init__()
        reduced_channels = max(channels // reduction, 16)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.mask_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, reduced_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduced_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.scale = math.sqrt(float(channels))

    @staticmethod
    def _sample_counterfactual_mask(mask):
        gaussian = torch.sigmoid(torch.randn_like(mask))
        uniform = torch.rand_like(mask)
        choose_gaussian = torch.rand_like(mask).ge(0.5)
        return torch.where(choose_gaussian, gaussian, uniform)

    def _attend(self, q_map, k_map, v_map, bias):
        batch, channels, height, width = q_map.shape
        locations = height * width
        q = q_map.reshape(batch, channels, locations).transpose(1, 2)
        k = k_map.reshape(batch, channels, locations)
        v = v_map.reshape(batch, channels, locations).transpose(1, 2)
        scores = torch.matmul(q, k) / self.scale
        scores = scores + bias.reshape(batch, 1, locations)
        attention = scores.softmax(dim=-1)
        attended = torch.matmul(attention, v)
        return attended.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, features, return_counterfactual=False):
        qkv = self.qkv(features)
        q_map, k_map, v_map = qkv.chunk(3, dim=1)
        factual_mask = self.mask_mlp(torch.cat([q_map, k_map], dim=1))
        factual = self._attend(q_map, k_map, v_map, factual_mask)
        factual = self.norm(self.project(factual) + features)
        factual = self.dropout(factual)

        counterfactual = None
        counterfactual_mask = None
        if return_counterfactual:
            counterfactual_mask = self._sample_counterfactual_mask(factual_mask.detach())
            counterfactual = self._attend(q_map, k_map, v_map, counterfactual_mask)
            counterfactual = self.norm(self.project(counterfactual) + features)
            counterfactual = self.dropout(counterfactual)
        return factual, counterfactual, factual_mask


class CausalResNetEncoder(nn.Module):
    """ResNet visual encoder with LMA/CFA at multiple spatial stages."""

    def __init__(
            self,
            c2d_type="resnet34",
            pretrained=True,
            cfe_reduction=8,
            gradient_checkpointing=True,
    ):
        super(CausalResNetEncoder, self).__init__()
        backbone = getattr(models, c2d_type)(pretrained=pretrained)
        self.gradient_checkpointing = gradient_checkpointing
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        layer3_channels = backbone.layer3[-1].bn2.num_features
        layer4_channels = backbone.layer4[-1].bn2.num_features
        self.cfe3 = LearnableCausalAttention(layer3_channels, cfe_reduction)
        self.cfe4 = LearnableCausalAttention(layer4_channels, cfe_reduction)
        self.output_size = layer4_channels

    def _run_layer(self, layer, features):
        if self.gradient_checkpointing and self.training:
            return checkpoint(layer, features)
        return layer(features)

    def forward(self, frames, return_counterfactual=False):
        features = self.maxpool(self.relu(self.bn1(self.conv1(frames))))
        if self.gradient_checkpointing and self.training and not features.requires_grad:
            features.requires_grad_(True)
        features = self._run_layer(self.layer1, features)
        features = self._run_layer(self.layer2, features)
        features = self._run_layer(self.layer3, features)
        features, _, layer3_mask = self.cfe3(features, return_counterfactual=False)
        features = self._run_layer(self.layer4, features)
        factual, counterfactual, layer4_mask = self.cfe4(
            features,
            return_counterfactual=return_counterfactual,
        )
        factual = self.avgpool(factual).flatten(1)
        if counterfactual is not None:
            counterfactual = self.avgpool(counterfactual).flatten(1)
        return {
            "factual": factual,
            "counterfactual": counterfactual,
            "attention_maps": {
                "layer3": layer3_mask,
                "layer4": layer4_mask,
            },
        }


class InteractiveContextLearning(nn.Module):
    """The K5-P2-K5-P2 visual sequence is refined as a T x D image."""

    def __init__(self, channels):
        super(InteractiveContextLearning, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _downsample(features, mode):
        image = features.transpose(1, 2).unsqueeze(1)
        if mode == "avg":
            return F.avg_pool2d(image, kernel_size=2, stride=2, ceil_mode=True)
        return F.max_pool2d(image, kernel_size=2, stride=2, ceil_mode=True)

    @staticmethod
    def _upsample(features, size):
        return F.interpolate(features, size=size, mode="bilinear", align_corners=False)

    def forward(self, features):
        # features: T x B x D
        sequence = features.transpose(0, 1)
        target_size = (sequence.size(1), sequence.size(2))
        avg_features = self._downsample(sequence, "avg")
        max_features = self._downsample(sequence, "max")
        avg_activation = torch.sigmoid(avg_features)
        max_activation = torch.sigmoid(max_features)
        avg_cross = avg_features * max_activation
        max_cross = max_features * avg_activation
        combined = self._upsample(avg_cross, target_size) + self._upsample(max_cross, target_size)
        gate = self.fuse(combined).squeeze(1)
        enhanced = sequence * gate
        return enhanced.transpose(0, 1)


def _ctc_forced_align(log_probs, target, target_length, input_length):
    """Return one or more frame indices for every target token."""
    frames = min(int(input_length), log_probs.size(0))
    labels = target[:int(target_length)].tolist()
    label_count = len(labels)
    if label_count == 0 or frames <= 0:
        return []

    states = label_count * 2 + 1
    log_values = log_probs[:frames].detach().cpu()
    negative_inf = -1.0e30
    scores = torch.full((frames, states), negative_inf, dtype=torch.float64)
    backpointers = torch.full((frames, states), -1, dtype=torch.long)

    scores[0, 0] = float(log_values[0, 0])
    if states > 1:
        scores[0, 1] = float(log_values[0, labels[0]])

    for timestep in range(1, frames):
        for state in range(states):
            candidates = [(float(scores[timestep - 1, state]), state)]
            if state > 0:
                candidates.append((float(scores[timestep - 1, state - 1]), state - 1))
            if state > 1 and state % 2 == 0 and labels[state // 2 - 1] != labels[state // 2 - 2]:
                candidates.append((float(scores[timestep - 1, state - 2]), state - 2))
            best_score, best_state = max(candidates, key=lambda item: item[0])
            state_index = state
            if state % 2 == 0:
                state_index = 0
            scores[timestep, state] = best_score + float(log_values[timestep, state_index])
            backpointers[timestep, state] = best_state

    final_state = states - 1
    if states > 1 and float(scores[-1, states - 2]) > float(scores[-1, states - 1]):
        final_state = states - 2

    state_indices = []
    state = final_state
    for timestep in range(frames - 1, -1, -1):
        state_indices.append((timestep, state))
        state = int(backpointers[timestep, state])
        if state < 0 and timestep > 0:
            break
    state_indices.reverse()

    aligned_frames = [[] for _ in range(label_count)]
    for timestep, state in state_indices:
        if state >= 1 and state % 2 == 1 and state // 2 < label_count:
            aligned_frames[state // 2].append(timestep)
    return aligned_frames


class CEMSLRModel(nn.Module):
    """Paper-guided implementation of CEM-SLR on top of VAC."""

    def __init__(
            self,
            num_classes,
            c2d_type,
            conv_type,
            use_bn=False,
            hidden_size=1024,
            gloss_dict=None,
            loss_weights=None,
            weight_norm=True,
            share_classifier=True,
            pretrained_backbone=True,
            gradient_checkpointing=True,
            cfe_reduction=8,
            text_model_path=None,
            text_hidden_size=768,
            freeze_text_encoder=True,
            text_max_length=8,
            causal_beta=1.0,
    ):
        super(CEMSLRModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.loss_weights = loss_weights
        self.gloss_dict = gloss_dict
        self.text_hidden_size = text_hidden_size
        self.freeze_text_encoder = freeze_text_encoder
        self.text_max_length = text_max_length
        self.causal_beta = causal_beta
        self.i2g = {value[0]: key for key, value in gloss_dict.items()}

        self.visual_encoder = CausalResNetEncoder(
            c2d_type=c2d_type,
            pretrained=pretrained_backbone,
            cfe_reduction=cfe_reduction,
            gradient_checkpointing=gradient_checkpointing,
        )
        visual_size = self.visual_encoder.output_size
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.frame_classifier = nn.Linear(visual_size, num_classes)

        self.conv1d = TemporalConv(
            input_size=visual_size,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes,
        )
        self.interactive_context = InteractiveContextLearning(hidden_size)
        self.temporal_model = BiLSTMLayer(
            rnn_type="LSTM",
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)
        self.conv1d.fc = self.classifier if share_classifier else nn.Linear(hidden_size, num_classes)
        self.text_classifier = nn.Linear(hidden_size, num_classes)
        self.multimodal_merger = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )
        self.multimodal_classifier = nn.Linear(hidden_size, num_classes)
        self.counterfactual_logits = nn.Parameter(torch.full((num_classes,), -4.0))

        self.text_tokenizer = None
        self.text_encoder = None
        self.gloss_embedding = None
        if text_model_path:
            from transformers import AutoModel, AutoTokenizer

            self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_model_path,
                local_files_only=True,
                use_fast=True,
            )
            self.text_encoder = AutoModel.from_pretrained(
                text_model_path,
                local_files_only=True,
            )
            self.text_encoder.eval()
            if freeze_text_encoder:
                for parameter in self.text_encoder.parameters():
                    parameter.requires_grad = False
            self.text_projection = nn.Sequential(
                nn.Linear(text_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
        else:
            self.gloss_embedding = nn.Embedding(num_classes, hidden_size)
            self.text_projection = nn.Identity()

        self.decoder = utils.Decode(gloss_dict, num_classes, "beam")
        self.loss = dict()
        self.criterion_init()

    def train(self, mode=True):
        super(CEMSLRModel, self).train(mode)
        if self.text_encoder is not None and self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def criterion_init(self):
        self.loss["CTCLoss"] = nn.CTCLoss(reduction="none", zero_infinity=False)
        self.loss["distillation"] = SeqKD(T=8)
        return self.loss

    def _resnet_forward(self, videos, video_lengths):
        batch, timesteps, channels, height, width = videos.shape
        frames = videos.reshape(batch * timesteps, channels, height, width)
        visual = self.visual_encoder(
            frames,
            return_counterfactual=self.training,
        )
        factual = visual["factual"].reshape(batch, timesteps, -1)
        counterfactual = None
        if visual["counterfactual"] is not None:
            counterfactual = visual["counterfactual"].reshape(batch, timesteps, -1)
        return factual, counterfactual, visual["attention_maps"]

    @staticmethod
    def _sequence_mask(lengths, max_length):
        positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
        return positions < lengths.unsqueeze(1)

    def _encode_text(self, gloss_ids, gloss_mask):
        if self.text_encoder is None:
            features = self.gloss_embedding(gloss_ids)
            return self.text_projection(features)

        valid_ids = gloss_ids[gloss_mask].long()
        glosses = [self.i2g[int(gloss_id)] for gloss_id in valid_ids.tolist()]
        if not glosses:
            return torch.zeros(
                gloss_mask.shape + (self.hidden_size,),
                device=gloss_mask.device,
                dtype=self.visual_projection[0].weight.dtype,
            )

        tokenized = self.text_tokenizer(
            glosses,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        device = self.text_projection[0].weight.device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        special_mask = tokenized["special_tokens_mask"].to(device)
        with torch.no_grad():
            encoded = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        content_mask = attention_mask * (1 - special_mask)
        content_mask = content_mask.float()
        denominator = content_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (encoded * content_mask.unsqueeze(-1)).sum(dim=1) / denominator
        pooled = self.text_projection(pooled)
        projected = pooled.new_zeros(
            (gloss_mask.size(0), gloss_mask.size(1), self.hidden_size)
        )
        projected[gloss_mask] = pooled
        return projected

    def _prepare_gloss_data(self, label, label_lgt, features, logits, lengths):
        batch = features.size(0)
        label_lengths = label_lgt.long().cpu().tolist()
        labels = label.long().cpu().split(label_lengths)
        max_glosses = max(label_lengths) if label_lengths else 0
        if max_glosses == 0:
            empty = features.new_zeros((batch, 0, features.size(-1)))
            return {
                "features": empty,
                "labels": torch.empty((batch, 0), dtype=torch.long, device=features.device),
                "mask": torch.zeros((batch, 0), dtype=torch.bool, device=features.device),
            }

        gloss_features = []
        gloss_labels = torch.full(
            (batch, max_glosses),
            -100,
            dtype=torch.long,
            device=features.device,
        )
        gloss_mask = torch.zeros(
            (batch, max_glosses),
            dtype=torch.bool,
            device=features.device,
        )
        sequence_log_probs = logits.log_softmax(dim=-1)
        for sample_index, sample_labels in enumerate(labels):
            sample_length = int(lengths[sample_index].item())
            if len(sample_labels) > sample_length:
                raise ValueError(
                    "CTC alignment requires at least one frame per gloss: {} > {}".format(
                        len(sample_labels),
                        sample_length,
                    )
                )
            aligned = _ctc_forced_align(
                sequence_log_probs[:, sample_index, :],
                sample_labels,
                len(sample_labels),
                sample_length,
            )
            sample_features = features.new_zeros((max_glosses, features.size(-1)))
            sample_labels_tensor = sample_labels.to(features.device)
            for gloss_index, frame_indices in enumerate(aligned):
                if gloss_index >= max_glosses:
                    break
                if frame_indices:
                    frame_tensor = torch.tensor(
                        frame_indices,
                        dtype=torch.long,
                        device=features.device,
                    ).clamp(min=0, max=features.size(1) - 1)
                    sample_features[gloss_index] = features[sample_index].index_select(
                        0, frame_tensor
                    ).mean(dim=0)
                gloss_labels[sample_index, gloss_index] = sample_labels_tensor[gloss_index]
                gloss_mask[sample_index, gloss_index] = True
            gloss_features.append(sample_features)
        return {
            "features": torch.stack(gloss_features, dim=0),
            "labels": gloss_labels,
            "mask": gloss_mask,
        }

    def _fuse_logits(self, visual_logits, text_logits, hidden_logits):
        visual_prob = torch.sigmoid(visual_logits)
        text_prob = torch.sigmoid(text_logits)
        hidden_prob = torch.sigmoid(hidden_logits)
        harmonic = visual_prob * text_prob * hidden_prob
        additive = torch.log((visual_prob + text_prob + hidden_prob).clamp(min=1.0e-6))
        return (
            self.causal_beta / (self.causal_beta + 1.0) * harmonic
            + 1.0 / (self.causal_beta + 1.0) * additive
        )

    def _causal_fusion(self, factual_features, conv_logits, feat_len, label, label_lgt):
        gloss_data = self._prepare_gloss_data(
            label,
            label_lgt,
            factual_features,
            conv_logits,
            feat_len,
        )
        visual_features = gloss_data["features"]
        gloss_mask = gloss_data["mask"]
        gloss_ids = gloss_data["labels"].clamp(min=0)
        text_features = self._encode_text(gloss_ids, gloss_mask)

        visual_logits = self.classifier(visual_features)
        text_logits = self.text_classifier(text_features)
        hidden = self.multimodal_merger(torch.cat([visual_features, text_features], dim=-1))
        hidden_logits = self.multimodal_classifier(hidden)
        fused_logits = self._fuse_logits(visual_logits, text_logits, hidden_logits)

        counterfactual_hidden = self.counterfactual_logits.reshape(1, 1, -1).expand_as(hidden_logits)
        counterfactual_fused = self._fuse_logits(
            visual_logits.detach(),
            text_logits.detach(),
            counterfactual_hidden,
        )
        return {
            "gloss_labels": gloss_data["labels"],
            "gloss_mask": gloss_mask,
            "visual_gloss_logits": visual_logits,
            "text_gloss_logits": text_logits,
            "fused_gloss_logits": fused_logits,
            "counterfactual_gloss_logits": counterfactual_fused,
        }

    def forward(self, x, len_x, label=None, label_lgt=None):
        factual, counterfactual, attention_maps = self._resnet_forward(x, len_x)
        factual_features = factual.transpose(1, 2)
        projected_sequence = self.visual_projection(factual)
        visual_sequence = projected_sequence.transpose(1, 2)
        conv_outputs = self.conv1d(factual_features, len_x)
        feat_len = conv_outputs["feat_len"]
        interactive_features = self.interactive_context(conv_outputs["visual_feat"])
        temporal_outputs = self.temporal_model(interactive_features, feat_len)
        outputs = self.classifier(temporal_outputs["predictions"])

        cfa_logits = None
        if counterfactual is not None:
            factual_logits = self.frame_classifier(factual)
            counterfactual_logits = self.frame_classifier(counterfactual)
            cfa_logits = (factual_logits - counterfactual_logits).transpose(0, 1)

        fusion_outputs = None
        if self.training and label is not None and label_lgt is not None:
            fusion_outputs = self._causal_fusion(
                projected_sequence,
                conv_outputs["conv_logits"],
                feat_len,
                label,
                label_lgt,
            )

        pred = None
        conv_pred = None
        if not self.training:
            pred = self.decoder.decode(outputs, feat_len, batch_first=False, probs=False)
            conv_pred = self.decoder.decode(
                conv_outputs["conv_logits"],
                feat_len,
                batch_first=False,
                probs=False,
            )

        output = {
            "framewise_features": factual,
            "visual_features": factual_features,
            "feat_len": feat_len,
            "frame_length": len_x,
            "attention_maps": attention_maps,
            "conv_logits": conv_outputs["conv_logits"],
            "sequence_logits": outputs,
            "cfa_logits": cfa_logits,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }
        if fusion_outputs is not None:
            output.update(fusion_outputs)
        return output

    def _ctc_loss(self, logits, label, input_length, target_length):
        return self.loss["CTCLoss"](
            logits.log_softmax(dim=-1),
            label.cpu().int(),
            input_length.cpu().int(),
            target_length.cpu().int(),
        ).mean()

    def _fusion_losses(self, output):
        mask = output["gloss_mask"]
        labels = output["gloss_labels"]
        if mask.sum().item() == 0:
            zero = output["fused_gloss_logits"].sum() * 0.0
            return zero, zero

        def masked_cross_entropy(logits):
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(mask)
            return losses[mask].mean()

        causal_kl = (
            masked_cross_entropy(output["fused_gloss_logits"])
            + masked_cross_entropy(output["visual_gloss_logits"])
            + masked_cross_entropy(output["text_gloss_logits"])
        )

        factual = F.softmax(output["fused_gloss_logits"].detach(), dim=-1)
        counterfactual_log_probs = F.log_softmax(
            output["counterfactual_gloss_logits"],
            dim=-1,
        )
        separation = F.kl_div(
            counterfactual_log_probs,
            factual,
            reduction="none",
        ).sum(dim=-1)
        causal_separation = separation[mask].mean()
        return causal_kl, causal_separation

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0.0
        for name, weight in self.loss_weights.items():
            if name == "ConvCTC":
                loss = loss + weight * self._ctc_loss(
                    ret_dict["conv_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                )
            elif name == "SeqCTC":
                loss = loss + weight * self._ctc_loss(
                    ret_dict["sequence_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                )
            elif name == "CFA":
                if ret_dict.get("cfa_logits") is None:
                    raise ValueError("CFA loss requires a counterfactual visual branch.")
                loss = loss + weight * self._ctc_loss(
                    ret_dict["cfa_logits"],
                    label,
                    ret_dict["frame_length"],
                    label_lgt,
                )
            elif name == "CauKL":
                causal_kl, _ = self._fusion_losses(ret_dict)
                loss = loss + weight * causal_kl
            elif name == "CauSep":
                _, causal_separation = self._fusion_losses(ret_dict)
                loss = loss + weight * causal_separation
            elif name == "Dist":
                loss = loss + weight * self.loss["distillation"](
                    ret_dict["conv_logits"],
                    ret_dict["sequence_logits"].detach(),
                    use_blank=False,
                )
            else:
                raise ValueError("Unknown loss weight: {}".format(name))
        return loss
