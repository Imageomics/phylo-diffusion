import torch
import torch.nn as nn
from functools import partial
import clip
import open_clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel

from transformers import BertTokenizerFast  # TODO: add to reuquirements
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c

class HeirClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=[3, 6, 9, 38], key='class', device='cuda'):
        super().__init__()
        assert embed_dim % len(n_classes) == 0
        self.key = key
        self.device = device
        self.embed_heir_dim = embed_dim//len(n_classes)
        self.embedding_layers = []
        self.embedding_level0 = nn.Embedding(n_classes[0], self.embed_heir_dim)
        self.embedding_level1 = nn.Embedding(n_classes[1], self.embed_heir_dim)
        self.embedding_level2 = nn.Embedding(n_classes[2], self.embed_heir_dim)
        self.embedding_level3 = nn.Embedding(n_classes[3], self.embed_heir_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        batch_size = len(batch[key][0])
        heir_classes = batch[key]

        heir_classes = [[int(num) for num in item.split(', ')] for item in heir_classes[0]]
        transformed_list = [list(pair) for pair in zip(*heir_classes)]
        tensor_list = [torch.tensor(sublist).to(self.device) for sublist in transformed_list]
        tensor_reshaped = [torch.reshape(sublist, (batch_size, 1)) for sublist in tensor_list]

        embedding_list = [self.embedding_level0(tensor_reshaped[0]), self.embedding_level1(tensor_reshaped[1]), 
                          self.embedding_level2(tensor_reshaped[2]), self.embedding_level3(tensor_reshaped[3])]
        

        embedding = torch.cat(embedding_list, dim=-1)
        return embedding
    
class HeirClassEmbedderMultiLevel(nn.Module):
    def __init__(self, embed_dim, n_classes=[3, 6, 9, 38], key='class', device='cuda'):
        super().__init__()
        assert embed_dim % len(n_classes) == 0
        self.key = key
        self.device = device
        self.n_classes = n_classes
        self.embed_heir_dim = embed_dim//len(n_classes)
        self.embedding_layers = []
        self.embedding_layers = nn.ModuleList()
        for i in list(n_classes):
            embedding = nn.Embedding(i, self.embed_heir_dim)
            self.embedding_layers.append(embedding.to(self.device))
        
    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        batch_size = len(batch[key][0])
        hier_classes = batch[key]

        hier_classes = [[int(num) for num in item.split(', ')] for item in hier_classes[0]]
        transformed_list = [list(pair) for pair in zip(*hier_classes)]
        tensor_list = [torch.tensor(sublist).to(self.device) for sublist in transformed_list]
        tensor_reshaped = [torch.reshape(sublist, (batch_size, 1)) for sublist in tensor_list]

        embedding_list = []
        for i in range(len(self.n_classes)):
            embedding_list.append(self.embedding_layers[i](tensor_reshaped[i]))
        
        embedding = torch.cat(embedding_list, dim=-1)

        return embedding
    
class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text



class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

### not using - hugging face implementation
class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer.projection_dim = 512
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output
        # return pooled_output
        return z

    def encode(self, text):
        return self(text)
    
class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z
