import copy
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from loguru import logger

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config

        self.embeddings = Embeddings(self.config)
        self.vit_encoder = VitEncoder(self.config)

    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoded, attn_weights = self.vit_encoder(embedding_output)
        return encoded

class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        self.img_size = self.config['tf_img_size']
        self.in_channels = self.config['tf_in_channels']
        self.patch_size = self.config['tf_patch_size']
        self.hidden_size = self.config['tf_hidden_size']
        self.num_patches = self.config['tf_num_patches']
        self.drop_prob = self.config['tf_drop_prob']

        self.patch_embeddings = Conv2d(in_channels=self.in_channels,
                                       out_channels=self.hidden_size,
                                       kernel_size=self.patch_size,
                                       stride=self.patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size))
        self.droppout = Dropout(self.drop_prob)

    def forward(self, x):
        x = self.patch_embeddings(x)    # output shape: [b, hidden_size, h//path_size, w//patch_size] = [b, 768, 16, 16]
        x = x.flatten(2)    # output shape: [b, hidden_size, num_patches] = [b, 768, 16*16]
        x = x.transpose(-1, -2)    # output shape: [b, num_patches, hidden_size] = [b, 16*16, 768]
        b, n, hidden_size = x.shape

        embeddings = x + self.position_embeddings[:, :n, :] # crop the position_embeddings to the same size as x
        output = self.droppout(embeddings)
        return output
    
class VitEncoder(nn.Module):
    def __init__(self, config):
        super(VitEncoder, self).__init__()
        self.config = config
        self.hidden_size = self.config['tf_hidden_size']
        self.num_layers = self.config['tf_num_layers']
        self.vis = self.config['tf_vis']

        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(self.hidden_size, eps=1e-6)
        for _ in range(self.num_layers):
            layer = Block(self.config)
            self.layer.append(copy.deepcopy(layer))
        
    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.config = config
        self.hidden_size = self.config['tf_hidden_size']
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(self.config)
        self.attn = Attention(self.config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
    

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.config = config
        self.hidden_size = self.config['tf_hidden_size']
        self.mlp_dim = self.config['tf_mlp_dim']
        self.drop_prob = self.config['tf_drop_prob']

        self.fc1 = Linear(self.hidden_size, self.mlp_dim)
        self.fc2 = Linear(self.mlp_dim, self.hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(self.drop_prob)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.vis = self.config['tf_vis']
        self.num_attention_heads = self.config['tf_num_heads']
        self.hidden_size = self.config['tf_hidden_size']
        self.attention_drop_prob = self.config['tf_attention_drop_prob']

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(self.hidden_size, self.all_head_size)
        self.key = Linear(self.hidden_size, self.all_head_size)
        self.value = Linear(self.hidden_size, self.all_head_size)

        self.out = Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = Dropout(self.attention_drop_prob)
        self.proj_dropout = Dropout(self.attention_drop_prob)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class PostLayer(nn.Module):
    def __init__(self, config):
        super(PostLayer, self).__init__()
        self.config = config
        self.hidden_size = self.config['tf_hidden_size']
        self.input_h = self.config['tf_input_h']
        self.input_w = self.config['tf_input_w']
        self.out_channels = self.config['tf_out_channels']

        self.convert_hidden = nn.Sequential(
            nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.out_channels,
            kernel_size=1,
        ),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, hidden_states):
        b, n, hidden_size = hidden_states.shape
        x = hidden_states.permute(0, 2, 1).contiguous()
        x = x.view(b, hidden_size, self.input_h, self.input_w)
        x = self.convert_hidden(x)
        return x


def load_pretrained_transformer(config, model):
    npz_path = config['tf_pretrained_model']
    # Load the .npz file
    pretrained_dict = np.load(npz_path, allow_pickle=True)

    # This will hold the renamed state dict
    new_state_dict = {}

    for key, value in pretrained_dict.items():
        # Remove the 'Transformer/' prefix
        new_key = key.replace('Transformer/', '')
        # Replace the 'encoderblock_' with 'vit_encoder.layer.'
        new_key = new_key.replace('encoderblock_', 'vit_encoder.layer.')
        # Replace 'LayerNorm_0' and 'LayerNorm_2' with 'attention_norm' and 'ffn_norm' respectively
        new_key = new_key.replace('LayerNorm_0', 'attention_norm')
        new_key = new_key.replace('LayerNorm_2', 'ffn_norm')
        # Replace 'MlpBlock_3' with 'ffn'
        new_key = new_key.replace('MlpBlock_3/', 'ffn.')
        # Replace 'MultiHeadDotProductAttention_1' with 'attn'
        new_key = new_key.replace('MultiHeadDotProductAttention_1', 'attn')
        # Replace 'Dense_' with 'fc'
        new_key = new_key.replace('Dense_', 'fc')
        # Replace 'kernel' with 'weight'
        new_key = new_key.replace('kernel', 'weight')
        # The 'scale' in the pre-trained is equivalent to 'weight' in PyTorch's LayerNorm
        new_key = new_key.replace('/scale', '.weight')
        # Convert numpy arrays to torch tensors
        value = torch.from_numpy(value)
        # Add to new state dict
        new_state_dict[new_key] = value

    # Load the new state dict into the model
    try:
        model.load_state_dict(new_state_dict, strict=False)
        logger.info("Weights loaded successfully.")
        return new_state_dict
    except Exception as e:
        logger.debug("An error occurred while loading weights: ", e)


def compare_state_dict(model, pre_state_dict):
    # Keep the initial state_dict for comparison
    initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Load the provided state_dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(pre_state_dict, strict=False)

    # Iterate through the initial and current state_dict to check for changes
    for key in initial_state_dict:
        if key in pre_state_dict:
            if not torch.equal(initial_state_dict[key], model.state_dict()[key]):
                logger.info(f"Weights updated for layer: {key}")
            else:
                logger.info(f"No change in weights for layer: {key}")
        else:
            if key not in missing_keys:
                logger.info(f"Missing in provided state_dict but present in the model: {key}")

    for key in unexpected_keys:
        logger.info(f"Key in provided state_dict not present in the model: {key}")

    return missing_keys, unexpected_keys