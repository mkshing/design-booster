import random
import torch
from torch import nn
from transformers import (
    CLIPTextConfig,
    CLIPPreTrainedModel,
)
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from transformers.models.resnet.modeling_resnet import ResNetStage
from x_transformers import Encoder


class DesignBoosterConfig(CLIPTextConfig):
    def __init__(
            self,
            num_channels=4,
            num_image_prompt_tokens=20,
            layer_type="bottleneck",
            hidden_act="relu",
            transformer_depth=12,
            transformer_heads=8,
            dropout_prob_text=0.1,
            dropout_prob_image=0.6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.num_image_prompt_tokens = num_image_prompt_tokens
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.transformer_depth = transformer_depth
        self.transformer_heads = transformer_heads
        self.dropout_prob_text = dropout_prob_text
        self.dropout_prob_image = dropout_prob_image


class DesignBoosterPreTrainedModel(CLIPPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DesignBoosterConfig
    base_model_prefix = "design"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class LightImageEncoder(nn.Module):
    def __init__(self, config: DesignBoosterConfig):
        super().__init__()
        self.resnet_layer = nn.Sequential(
            ResNetStage(config=config, in_channels=config.num_channels, out_channels=config.num_channels, stride=1),
            ResNetStage(config=config, in_channels=config.num_channels, out_channels=config.num_channels, stride=1),
            ResNetStage(config=config, in_channels=config.num_channels, out_channels=config.num_channels, stride=2),
            ResNetStage(config=config, in_channels=config.num_channels, out_channels=config.num_channels, stride=1),
            ResNetStage(config=config, in_channels=config.num_channels, out_channels=config.num_channels, stride=1),
        )
        self.convolution = nn.Conv2d(config.num_channels, config.num_channels, kernel_size=3)

        self.linear = nn.Sequential(
            nn.Flatten(),
            # TODO: in_features is not flexible
            nn.Linear(
                in_features=3600,
                out_features=config.num_image_prompt_tokens*config.projection_dim
            ),
        )
        self.num_image_prompt_tokens = config.num_image_prompt_tokens
        self.projection_dim = config.projection_dim

    def forward(self, x):
        out = self.resnet_layer(x)
        out = self.convolution(out)
        out = self.linear(out)
        out = out.reshape((-1, self.num_image_prompt_tokens, self.projection_dim))
        return out


class DesignBoosterModel(DesignBoosterPreTrainedModel):
    def __init__(self, config: DesignBoosterConfig):
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)
        self.image_model = LightImageEncoder(config)
        self.transformer = Encoder(
            dim=config.projection_dim,
            depth=config.transformer_depth,
            heads=config.transformer_heads
        )
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_only_text_encoder(self):
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            pixel_values=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bsz = input_ids.shape[0]
        dropped_text = False
        if self.training and random.random() < self.config.dropout_prob_text:
            seq_len = input_ids.shape[1]
            out_text = torch.zeros(
                (bsz, seq_len, self.config.projection_dim),
                device=input_ids.device
            )
            dropped_text = True
        else:
            out_text = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                # return_dict=return_dict,
            ).last_hidden_state
        # TODO: avoid dropping both
        if self.training and not dropped_text and random.random() < self.config.dropout_prob_image:
            out_image = torch.zeros(
                (bsz, self.config.num_image_prompt_tokens, self.config.projection_dim),
                device=input_ids.device
            )
        else:
            out_image = self.image_model(pixel_values)
        out = torch.cat((out_text, out_image), dim=1)
        out = self.transformer(out)
        return (out,)

