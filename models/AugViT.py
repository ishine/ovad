"""
     OVAD: Speech detector in adversarial conditions

     Copyright 2023 by Lukasz Smietanka and Tomasz Maka

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.
"""

from torchaudio.transforms import MFCC
from torch_audiomentations import LowPassFilter
import torch
import numpy as np
from typing import Union


def to_patches(data: torch.Tensor) -> torch.Tensor:
    """Split input data to patches containing MFCC coefficients for each frame.

    Args:
      data: MFCC coefficients

    Returns:
      A set of patches
    """

    batches, features, sequences = data.shape
    result = torch.zeros(batches, sequences, features)
    for i, v in enumerate(data):
        for j in range(sequences):
            result[i, j, :] = v[:, j]
    return result


def positional_encoding(n_seq: int, hidden_dim: int) -> torch.Tensor:
    """Generate the vector containing positional information for all frames.

    Args:
      n_seq: number of frames
      hidden_dim: dimensionality of AugViT hidden space

    Returns:
      vector of positional information of the sequences
    """
    result = torch.ones(n_seq, hidden_dim)

    for i in range(n_seq):
        for j in range(hidden_dim):
            if j % 2 == 0:
                result[i][j] = np.sin(i / (10000 ** (j / hidden_dim)))
            else:
                result[i][j] = np.cos(i / (10000 ** ((j - 1) / hidden_dim)))

    return result


class AugHeadAttention(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        """Initializes the instance of the augmented attention layer.

        Args:
          hidden_dim: dimensionality of AugViT hidden space
        """
        super(AugHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim

        self.d_head = hidden_dim

        self.q_map = torch.nn.ModuleList([
            torch.nn.Linear(self.d_head, self.d_head)
            for _ in range(2)])

        self.k_map = torch.nn.ModuleList([
            torch.nn.Linear(self.d_head, self.d_head)
            for _ in range(2)])

        self.v_map = torch.nn.ModuleList([
            torch.nn.Linear(self.d_head, self.d_head)
            for _ in range(2)])

        self.softmax = torch.nn.Softmax(dim=1)

        self.mlp = torch.nn.Linear(in_features=2*self.hidden_dim,
                                   out_features=self.hidden_dim)

    def forward(self, sequences: list) -> torch.Tensor:
        """
        Args:
          sequences: input of augmented attention layer

        Returns:
          output of augmented attention layer
        """
        result = []

        for i, _ in enumerate(sequences[0]):
            seq_result = []
            for ih, head in enumerate(range(2)):
                q_map = self.q_map[head]
                k_map = self.k_map[head]
                v_map = self.v_map[head]

                seq = sequences[ih][i]
                q, k, v = q_map(seq), k_map(seq), v_map(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        out = torch.cat([torch.unsqueeze(r, dim=0) for r in result])

        return self.mlp(out)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int):
        """ Initializes the instance of the Multi-Head attention layer.

        Args:
          hidden_dim: dimensionality of AugViT hidden space
          n_heads: number of heads in attention stage
        """
        super(MultiHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.d_head = int(hidden_dim / n_heads)

        self.q_map = torch.nn.ModuleList([
            torch.nn.Linear(self.d_head, self.d_head)
            for _ in range(self.n_heads)])

        self.k_map = torch.nn.ModuleList([
            torch.nn.Linear(self.d_head, self.d_head)
            for _ in range(self.n_heads)])

        self.v_map = torch.nn.ModuleList([
            torch.nn.Linear(self.d_head, self.d_head)
            for _ in range(self.n_heads)])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, sequences: list) -> torch.Tensor:
        """
        Args:
          sequences: input of attention layer

        Returns:
          output of attention layer
        """
        result = []
        for s in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_map = self.q_map[head]
                k_map = self.k_map[head]
                v_map = self.v_map[head]

                seq = s[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_map(seq), k_map(seq), v_map(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)

            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class Block(torch.nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: int = 4):
        """Initializes the instance of the vision transformer block.

        Args:
          hidden_dim: dimensionality of AugViT hidden space
          n_heads: number of heads in attention stage
          mlp_ratio: number of items in MLP (mlp_ratio*hidden_dim)
        """
        super(Block, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.norm_1 = torch.nn.LayerNorm(hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(hidden_dim)

        self.aht = AugHeadAttention(hidden_dim)
        self.msa = MultiHeadAttention(hidden_dim, n_heads)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_ratio * hidden_dim, hidden_dim))

    def forward(self, x: Union[list, torch.Tensor]) -> torch.Tensor:
        """
        Args:
          x: input of ViT block

        Returns:
          output of ViT block
        """

        if isinstance(x, list):
            out = x[0] + self.aht([self.norm_1(x[0]), self.norm_1(x[1])])
        else:
            out = x + self.msa(self.norm_1(x))
        out = out + self.mlp(self.norm_2(out))
        return out


class Model(torch.nn.Module):
    def __init__(self, n_features: int, n_sequences: int, fs: int,
                 hidden_dim: int = 8, n_blocks: int = 10, device: str = 'cpu'):
        """Initializes the instance of the AugViT model.

        Args:
          n_features: number of input MFCC features
          n_sequences: number of frames in the signal
          fs: sampling rate [Hz]
          hidden_dim: dimensionality of AugViT hidden space
          n_blocks: number of ViT blocks
          device: device type used to allocate the torch data
        """
        super(Model, self).__init__()

        self.device = torch.device(device)

        self.fs = fs

        self.n_features = n_features
        self.n_sequences = n_sequences

        self.hidden_dim = hidden_dim

        self.linear_mapper = torch.nn.Linear(self.n_features, self.hidden_dim)

        self.class_token = torch.nn.Parameter(torch.rand(1, self.hidden_dim))

        self.pos_embed = torch.nn.Parameter(torch.tensor(
            positional_encoding(self.n_sequences, self.hidden_dim)))
        self.pos_embed.requires_grad = False

        self.blocks = torch.nn.ModuleList(
            [Block(hidden_dim, 2) for _ in range(n_blocks)])

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 1), torch.nn.Sigmoid())

        self.mfcc = MFCC(sample_rate=fs, n_mfcc=20,
                         melkwargs={"n_fft": 1102, "hop_length": 1103,
                                    "n_mels": 128, "center": True,
                                    "win_length": 1100})
        self.a = LowPassFilter(sample_rate=self.fs)

    def forward(self, samples: torch.Tensor, aug: bool = False) -> dict:
        """
        Args:
          samples: input audio data
          aug: If it equals True, input data are augmented.

        Returns:
          dictionary contained the data class [speech(1) / non-speech(0)]
          and speech occurence probability [0..1]
        """

        x_org = torch.squeeze(self.mfcc(samples), dim=1)

        if aug is False:
            x_aug = torch.squeeze(self.mfcc(samples), dim=1)
        else:
            x_aug = torch.squeeze(self.mfcc(self.a(samples)), dim=1)

        patches_org = to_patches(x_org).to(self.device)
        patches_aug = to_patches(x_aug).to(self.device)

        tokens_org = self.linear_mapper(patches_org).to(self.device)
        tokens_aug = self.linear_mapper(patches_aug).to(self.device)

        pos_embed = self.pos_embed.repeat(len(tokens_org), 1, 1)

        tokens_org = tokens_org * pos_embed
        tokens_aug = tokens_aug * pos_embed

        output_org = torch.stack([torch.vstack((self.class_token,
                                                tokens_org[i]))
                                  for i in range(len(tokens_org))])
        output_aug = torch.stack([torch.vstack((self.class_token,
                                                tokens_aug[i]))
                                  for i in range(len(tokens_aug))])

        for ib, block in enumerate(self.blocks):
            if ib == 0:
                output = block([output_org, output_aug])
            else:
                output = block(output)

        out = self.mlp(output[:, 0])
        return {"class": torch.round(out), "prob": out[:, 0]}
