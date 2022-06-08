import torch
import torch.nn as nn
from transformers import PerceiverConfig
from model import Embeddings
from transformers.models.perceiver.modeling_perceiver import PerceiverEncoder, PerceiverEmbeddings
from transformers.modeling_utils import ModuleUtilsMixin

class Perceiver_Model(nn.Module, ModuleUtilsMixin):
    def __init__(self, config):
        super().__init__()

        Perceiver_Config = PerceiverConfig(num_latents=192,
                                           d_latents=48,
                                           num_self_attends_per_block=1,
                                           d_model=config["transformer_dim"],
                                           num_blocks=1,
                                           num_self_attention_heads=config["num_head"],
                                           num_cross_attention_heads=config["num_head"],
                                           attention_probs_dropout_prob=config["attention_dropout"],
                                           layer_norm_eps=1e-05)
        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)

        self.latent_embeddings = PerceiverEmbeddings(Perceiver_Config)
        for idx in range(self.num_layers):
            setattr(self, f"transformer_{idx}", PerceiverEncoder(Perceiver_Config, kv_dim=Perceiver_Config.d_model))

        self.norm = nn.LayerNorm(48)

    def forward(self, input_ids, mask=None):

        X = self.embeddings(input_ids)

        batch_size, seq_length, _ = X.size()
        device = X.device

        if mask is None:
            mask = torch.ones(((batch_size, seq_length)), device=device)
        extended_attention_mask = self.invert_attention_mask(mask)

        latent = self.latent_embeddings(batch_size=batch_size)

        for idx in range(self.num_layers):
            latent = getattr(self, f"transformer_{idx}")(hidden_states = latent,
                                                         inputs = X, inputs_mask = extended_attention_mask).last_hidden_state

        latent = self.norm(latent)

        return latent