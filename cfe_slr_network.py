import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint

import utils
from modules import BiLSTMLayer, TemporalConv


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, strength):
        ctx.strength = strength
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class GradientReversal(nn.Module):
    def __init__(self, strength=1.0):
        super(GradientReversal, self).__init__()
        self.strength = strength

    def forward(self, inputs):
        return _GradientReversal.apply(inputs, self.strength)


class SemanticPreservingPerturbation(nn.Module):
    """Photometric intervention proxy that preserves the sign motion."""

    def __init__(self, brightness=0.15, contrast=0.15, gamma=0.15, noise=0.01):
        super(SemanticPreservingPerturbation, self).__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.noise = noise

    def forward(self, videos):
        batch = videos.size(0)
        shape = (batch, 1, 1, 1, 1)
        contrast = 1.0 + torch.empty(shape, device=videos.device).uniform_(
            -self.contrast, self.contrast
        )
        brightness = torch.empty(shape, device=videos.device).uniform_(
            -self.brightness, self.brightness
        )
        gamma = torch.empty(shape, device=videos.device).uniform_(
            1.0 - self.gamma, 1.0 + self.gamma
        )

        # Videos are normalized around zero by the feeder.
        unit_videos = ((videos + 1.0) * 0.5).clamp(0.0, 1.0)
        intervened = unit_videos.pow(gamma)
        intervened = (intervened - 0.5) * 2.0
        intervened = intervened * contrast + brightness
        if self.noise > 0.0:
            intervened = intervened + torch.randn_like(intervened) * self.noise
        return intervened.clamp(-1.0, 1.0)


class CFEResNetEncoder(nn.Module):
    def __init__(
            self,
            c2d_type="resnet34",
            pretrained=True,
            gradient_checkpointing=True,
    ):
        super(CFEResNetEncoder, self).__init__()
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
        self.output_size = backbone.layer4[-1].bn2.num_features

    def _run_layer(self, layer, features, use_checkpoint):
        if use_checkpoint:
            return checkpoint(layer, features)
        return layer(features)

    def forward(self, frames, use_checkpoint=None):
        if use_checkpoint is None:
            use_checkpoint = self.gradient_checkpointing and self.training
        features = self.maxpool(self.relu(self.bn1(self.conv1(frames))))
        if use_checkpoint and not features.requires_grad:
            features.requires_grad_(True)
        features = self._run_layer(self.layer1, features, use_checkpoint)
        features = self._run_layer(self.layer2, features, use_checkpoint)
        features = self._run_layer(self.layer3, features, use_checkpoint)
        features = self._run_layer(self.layer4, features, use_checkpoint)
        return self.avgpool(features).flatten(1)


class InteractiveContextLearning(nn.Module):
    """The K5-P2-K5-P2 sequence is refined as a T x D image."""

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
        sequence = features.transpose(0, 1)
        target_size = (sequence.size(1), sequence.size(2))
        avg_features = self._downsample(sequence, "avg")
        max_features = self._downsample(sequence, "max")
        avg_cross = avg_features * torch.sigmoid(max_features)
        max_cross = max_features * torch.sigmoid(avg_features)
        combined = self._upsample(avg_cross, target_size) + self._upsample(
            max_cross, target_size
        )
        gate = self.fuse(combined).squeeze(1)
        return (sequence * gate).transpose(0, 1)


def _ctc_forced_align(log_probs, target, target_length, input_length):
    """Return frame indices for each target token using the current CTC path."""
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
            if (
                state > 1
                and state % 2 == 0
                and labels[state // 2 - 1] != labels[state // 2 - 2]
            ):
                candidates.append(
                    (float(scores[timestep - 1, state - 2]), state - 2)
                )
            best_score, best_state = max(candidates, key=lambda item: item[0])
            state_index = state if state % 2 == 1 else 0
            scores[timestep, state] = (
                best_score + float(log_values[timestep, state_index])
            )
            backpointers[timestep, state] = best_state

    final_state = states - 1
    if states > 1 and scores[-1, states - 2] > scores[-1, states - 1]:
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


def _masked_mean(features, lengths):
    mask = torch.arange(features.size(1), device=features.device).unsqueeze(0)
    mask = mask < lengths.to(features.device).unsqueeze(1)
    mask = mask.unsqueeze(-1).to(features.dtype)
    denominator = mask.sum(dim=1).clamp(min=1.0)
    return (features * mask).sum(dim=1) / denominator


def _linear_hsic(left, right):
    """Scaled HSIC, with a covariance fallback for very small batches."""
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    if left.size(0) < 3:
        cross_covariance = left.t().matmul(right) / max(left.size(0) - 1, 1)
        return cross_covariance.pow(2).sum()
    left = left / math.sqrt(float(left.size(1)))
    right = right / math.sqrt(float(right.size(1)))
    left_kernel = left.matmul(left.t())
    right_kernel = right.matmul(right.t())
    count = left.size(0)
    centering = torch.eye(count, device=left.device) - (1.0 / count)
    left_kernel = centering.matmul(left_kernel).matmul(centering)
    right_kernel = centering.matmul(right_kernel).matmul(centering)
    return (left_kernel * right_kernel).sum() / float((count - 1) ** 2)


class PrivilegedGlossTeacher(nn.Module):
    """Optional training-only text teacher fed by ground-truth glosses."""

    def __init__(
            self,
            gloss_dict,
            num_classes,
            hidden_size,
            text_model_path,
            text_hidden_size=768,
            freeze_text_encoder=True,
            max_length=8,
    ):
        super(PrivilegedGlossTeacher, self).__init__()
        from transformers import AutoModel, AutoTokenizer

        self.i2g = {value[0]: key for key, value in gloss_dict.items()}
        self.hidden_size = hidden_size
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
        self.freeze_text_encoder = freeze_text_encoder
        self.max_length = max_length
        self.projection = nn.Sequential(
            nn.Linear(text_hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def train(self, mode=True):
        super(PrivilegedGlossTeacher, self).train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def forward(self, gloss_ids, gloss_mask):
        valid_ids = gloss_ids[gloss_mask].long()
        glosses = [self.i2g[int(gloss_id)] for gloss_id in valid_ids.tolist()]
        dtype = self.projection[0].weight.dtype
        if not glosses:
            return torch.zeros(
                gloss_mask.shape + (self.classifier.out_features,),
                device=gloss_mask.device,
                dtype=dtype,
            )

        tokenized = self.text_tokenizer(
            glosses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        device = self.projection[0].weight.device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        special_mask = tokenized["special_tokens_mask"].to(device)
        with torch.no_grad():
            encoded = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        content_mask = attention_mask.float() * (1 - special_mask.float())
        denominator = content_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (encoded * content_mask.unsqueeze(-1)).sum(dim=1) / denominator
        pooled = self.projection(pooled)
        logits = self.classifier(pooled)

        projected = logits.new_zeros(
            (gloss_mask.size(0), gloss_mask.size(1), logits.size(-1))
        )
        projected[gloss_mask] = logits
        return projected


class CFESLRModel(nn.Module):
    """Semantic/environment disentangling CSLR with intervention consistency."""

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
            num_environments=9,
            environment_size=256,
            adversary_strength=0.1,
            perturbation_brightness=0.15,
            perturbation_contrast=0.15,
            perturbation_gamma=0.15,
            perturbation_noise=0.01,
            text_model_path=None,
            text_hidden_size=768,
            freeze_text_encoder=True,
            text_max_length=8,
    ):
        super(CFESLRModel, self).__init__()
        if gloss_dict is None:
            raise ValueError("gloss_dict is required")
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.loss_weights = loss_weights or {}
        self.num_environments = num_environments

        self.visual_encoder = CFEResNetEncoder(
            c2d_type=c2d_type,
            pretrained=pretrained_backbone,
            gradient_checkpointing=gradient_checkpointing,
        )
        visual_size = self.visual_encoder.output_size
        self.semantic_projection = nn.Sequential(
            nn.Linear(visual_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.environment_projection = nn.Sequential(
            nn.Linear(visual_size, environment_size),
            nn.LayerNorm(environment_size),
            nn.GELU(),
        )
        self.environment_classifier = nn.Linear(environment_size, num_environments)
        self.semantic_environment_classifier = nn.Linear(
            hidden_size, num_environments
        )
        self.environment_adversary = GradientReversal(adversary_strength)

        self.conv1d = TemporalConv(
            input_size=hidden_size,
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
        self.conv1d.fc = self.classifier if share_classifier else nn.Linear(
            hidden_size, num_classes
        )

        self.perturbation = SemanticPreservingPerturbation(
            brightness=perturbation_brightness,
            contrast=perturbation_contrast,
            gamma=perturbation_gamma,
            noise=perturbation_noise,
        )
        self.gloss_dict = gloss_dict
        self.text_teacher = None
        if text_model_path and self.loss_weights.get("PrivilegedTeacher", 0.0) > 0:
            self.text_teacher = PrivilegedGlossTeacher(
                gloss_dict=gloss_dict,
                num_classes=num_classes,
                hidden_size=hidden_size,
                text_model_path=text_model_path,
                text_hidden_size=text_hidden_size,
                freeze_text_encoder=freeze_text_encoder,
                max_length=text_max_length,
            )

        self.decoder = utils.Decode(gloss_dict, num_classes, "beam")
        self.loss = dict()
        self.criterion_init()

    def train(self, mode=True):
        super(CFESLRModel, self).train(mode)
        if self.text_teacher is not None:
            self.text_teacher.train(mode)
        return self

    def criterion_init(self):
        self.loss["CTCLoss"] = nn.CTCLoss(reduction="none", zero_infinity=False)

    def _extract_frames(self, videos):
        batch, timesteps, channels, height, width = videos.shape
        frames = videos.reshape(batch * timesteps, channels, height, width)
        encoded = self.visual_encoder(
            frames,
            use_checkpoint=self.visual_encoder.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled(),
        )
        return encoded.reshape(batch, timesteps, -1)

    def _encode_sequence(self, videos, video_lengths):
        frame_features = self._extract_frames(videos)
        semantic_features = self.semantic_projection(frame_features)
        environment_features = self.environment_projection(frame_features)

        conv_outputs = self.conv1d(semantic_features.transpose(1, 2), video_lengths)
        interactive_features = self.interactive_context(conv_outputs["visual_feat"])
        temporal_outputs = self.temporal_model(
            interactive_features, conv_outputs["feat_len"]
        )
        sequence_logits = self.classifier(temporal_outputs["predictions"])

        semantic_pooled = _masked_mean(semantic_features, video_lengths)
        environment_pooled = _masked_mean(environment_features, video_lengths)
        return {
            "frame_features": frame_features,
            "semantic_features": semantic_features,
            "environment_features": environment_features,
            "semantic_pooled": semantic_pooled,
            "environment_pooled": environment_pooled,
            "conv_logits": conv_outputs["conv_logits"],
            "sequence_logits": sequence_logits,
            "feat_len": conv_outputs["feat_len"],
        }

    @staticmethod
    def _is_enabled(loss_weights, name):
        return loss_weights.get(name, 0.0) > 0.0

    def _privileged_teacher_loss(self, output, label, label_lgt):
        if self.text_teacher is None:
            return output["sequence_logits"].sum() * 0.0
        label_lengths = label_lgt.long().cpu().tolist()
        labels = label.long().cpu().split(label_lengths)
        max_glosses = max(label_lengths) if label_lengths else 0
        if max_glosses == 0:
            return output["sequence_logits"].sum() * 0.0

        gloss_labels = torch.full(
            (len(labels), max_glosses),
            -100,
            dtype=torch.long,
            device=label.device,
        )
        gloss_mask = torch.zeros(
            (len(labels), max_glosses),
            dtype=torch.bool,
            device=label.device,
        )
        for sample_index, sample_labels in enumerate(labels):
            sample_labels = sample_labels.to(label.device)
            length = sample_labels.numel()
            gloss_labels[sample_index, :length] = sample_labels
            gloss_mask[sample_index, :length] = True

        teacher_logits = self.text_teacher(
            gloss_labels.clamp(min=0),
            gloss_mask,
        ).detach()
        teacher_probs = teacher_logits.softmax(dim=-1)
        sequence_log_probs = output["sequence_logits"].log_softmax(dim=-1)

        losses = []
        for sample_index, sample_labels in enumerate(labels):
            active_length = int(output["feat_len"][sample_index].item())
            aligned = _ctc_forced_align(
                sequence_log_probs[:, sample_index, :],
                sample_labels,
                len(sample_labels),
                active_length,
            )
            for gloss_index, frame_indices in enumerate(aligned):
                if not frame_indices:
                    continue
                frames = torch.tensor(
                    frame_indices,
                    dtype=torch.long,
                    device=label.device,
                )
                student = sequence_log_probs[:, sample_index, :].index_select(0, frames)
                target = teacher_probs[sample_index, gloss_index].unsqueeze(0)
                losses.append(
                    -(target * student).sum(dim=-1).mean()
                )
        if not losses:
            return output["sequence_logits"].sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
            self,
            x,
            len_x,
            label=None,
            label_lgt=None,
            env_ids=None,
            return_counterfactual=False,
    ):
        clean = self._encode_sequence(x, len_x)
        semantic_pooled = clean["semantic_pooled"]
        environment_pooled = clean["environment_pooled"]
        environment_logits = self.environment_classifier(environment_pooled)
        semantic_environment_logits = self.semantic_environment_classifier(
            self.environment_adversary(semantic_pooled)
        )

        counterfactual = None
        should_perturb = (
            self.training and self._is_enabled(self.loss_weights, "Consistency")
        ) or return_counterfactual
        if should_perturb:
            with torch.no_grad():
                intervened_videos = self.perturbation(x)
                counterfactual = self._encode_sequence(intervened_videos, len_x)

        predictions = None
        conv_predictions = None
        if not self.training:
            predictions = self.decoder.decode(
                clean["sequence_logits"],
                clean["feat_len"],
                batch_first=False,
                probs=False,
            )
            conv_predictions = self.decoder.decode(
                clean["conv_logits"],
                clean["feat_len"],
                batch_first=False,
                probs=False,
            )

        output = {
            "framewise_features": clean["frame_features"],
            "semantic_features": clean["semantic_features"],
            "environment_features": clean["environment_features"],
            "semantic_pooled": semantic_pooled,
            "environment_pooled": environment_pooled,
            "environment_logits": environment_logits,
            "semantic_environment_logits": semantic_environment_logits,
            "feat_len": clean["feat_len"],
            "frame_length": len_x,
            "conv_logits": clean["conv_logits"],
            "sequence_logits": clean["sequence_logits"],
            "recognized_sents": predictions,
            "conv_sents": conv_predictions,
            "env_ids": env_ids,
            "counterfactual": counterfactual,
        }
        if (
            self.training
            and label is not None
            and label_lgt is not None
            and self._is_enabled(self.loss_weights, "PrivilegedTeacher")
        ):
            output["privileged_teacher_loss"] = self._privileged_teacher_loss(
                output, label, label_lgt
            )
        return output

    def _ctc_loss(
            self,
            logits,
            label,
            input_length,
            target_length,
            reduce=True
    ):
        losses = self.loss["CTCLoss"](
            logits.log_softmax(dim=-1),
            label.cpu().int(),
            input_length.cpu().int(),
            target_length.cpu().int(),
        )
        return losses.mean() if reduce else losses

    def _group_risk_variance(self, ret_dict, label, label_lgt):
        env_ids = ret_dict.get("env_ids")
        zero = ret_dict["sequence_logits"].sum() * 0.0
        if env_ids is None:
            return zero
        conv_risks = self._ctc_loss(
            ret_dict["conv_logits"],
            label,
            ret_dict["feat_len"],
            label_lgt,
            reduce=False,
        )
        sequence_risks = self._ctc_loss(
            ret_dict["sequence_logits"],
            label,
            ret_dict["feat_len"],
            label_lgt,
            reduce=False,
        )
        sample_risks = 0.5 * (conv_risks + sequence_risks)
        group_means = []
        for environment in torch.unique(env_ids):
            group_means.append(sample_risks[env_ids == environment].mean())
        if len(group_means) < 2:
            return zero
        return torch.stack(group_means).var(unbiased=False)

    @staticmethod
    def _prediction_kl(student_logits, teacher_logits, lengths):
        teacher_probs = teacher_logits.detach().softmax(dim=-1)
        student_log_probs = student_logits.log_softmax(dim=-1)
        losses = (
            teacher_probs
            * (teacher_probs.clamp(min=1.0e-8).log() - student_log_probs)
        ).sum(dim=-1)
        mask = torch.arange(losses.size(0), device=losses.device).unsqueeze(1)
        mask = mask < lengths.to(losses.device).unsqueeze(0)
        return losses[mask].mean()

    def criterion_calculation(self, ret_dict, label, label_lgt):
        total_loss = 0.0
        zero = ret_dict["sequence_logits"].sum() * 0.0
        for name, weight in self.loss_weights.items():
            if name == "ConvCTC":
                current = self._ctc_loss(
                    ret_dict["conv_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                )
            elif name == "SeqCTC":
                current = self._ctc_loss(
                    ret_dict["sequence_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                )
            elif name == "Consistency":
                counterfactual = ret_dict.get("counterfactual")
                if counterfactual is None:
                    current = zero
                else:
                    current = self._prediction_kl(
                        ret_dict["sequence_logits"],
                        counterfactual["sequence_logits"],
                        ret_dict["feat_len"],
                    )
            elif name == "EnvironmentCE":
                env_ids = ret_dict.get("env_ids")
                current = zero if env_ids is None else F.cross_entropy(
                    ret_dict["environment_logits"], env_ids.long()
                )
            elif name == "EnvironmentAdv":
                env_ids = ret_dict.get("env_ids")
                current = zero if env_ids is None else F.cross_entropy(
                    ret_dict["semantic_environment_logits"], env_ids.long()
                )
            elif name == "InvariantRisk":
                current = self._group_risk_variance(
                    ret_dict,
                    label,
                    label_lgt,
                )
            elif name == "Disentangle":
                current = _linear_hsic(
                    ret_dict["semantic_pooled"],
                    ret_dict["environment_pooled"],
                )
            elif name == "PrivilegedTeacher":
                current = ret_dict.get("privileged_teacher_loss", zero)
            else:
                raise ValueError("Unknown loss weight: {}".format(name))
            total_loss = total_loss + float(weight) * current
        return total_loss
