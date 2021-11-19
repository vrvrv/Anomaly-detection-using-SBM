# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""The NCSNv2 model."""
import torch
import torch.nn as nn
import functools

from .utils import get_sigmas
from .blocks import (
    CondRefineBlock,
    RefineBlock,
    ResidualBlock,
    ncsn_conv3x3,
    ConditionalResidualBlock,
    get_act
)
from .normalization import get_normalization

CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3


def get_network(config):
    """image size : 128 ~ 256"""
    return functools.partial(NCSNv2, config=config)


class NCSNv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.centered = config.data.centered
        self.norm = get_normalization(config)
        self.nf = nf = config.model.nf

        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(get_sigmas(config)))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, nf, 3, stride=1, padding=1)

        self.normalizer = self.norm(nf, config.model.num_scales)
        self.end_conv = nn.Conv2d(nf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.nf, self.nf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.nf, self.nf, resample=None, act=act,
                          normalization=self.norm)
        ])

        self.res2 = nn.ModuleList([
            ResidualBlock(self.nf, 2 * self.nf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                          normalization=self.norm)
        ])

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                          normalization=self.norm, dilation=2)
        ])


        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                          normalization=self.norm, adjust_padding=False, dilation=4),
            ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([2 * self.nf], 2 * self.nf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.nf, 2 * self.nf], 2 * self.nf, act=act)
        self.refine3 = RefineBlock([2 * self.nf, 2 * self.nf], self.nf, act=act)
        self.refine4 = RefineBlock([self.nf, self.nf], self.nf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.centered:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output


class NCSNv2(nn.Module):
    """NCSNv2 model architecture for 256px images."""

    def __init__(self, config):
        super().__init__()
        self.centered = config.data.centered
        self.norm = get_normalization(config)
        self.nf = nf = config.model.nf
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(get_sigmas(config)))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, nf, 3, stride=1, padding=1)
        self.normalizer = self.norm(nf, config.model.num_scales)

        self.end_conv = nn.Conv2d(nf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.nf, self.nf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.nf, self.nf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.nf, 2 * self.nf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res31 = nn.ModuleList([
            ResidualBlock(2 * self.nf, 2 * self.nf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.nf, 2 * self.nf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.nf, 4 * self.nf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(4 * self.nf, 4 * self.nf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.nf, 4 * self.nf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(4 * self.nf, 4 * self.nf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([4 * self.nf], 4 * self.nf, act=act, start=True)
        self.refine2 = RefineBlock([4 * self.nf, 4 * self.nf], 2 * self.nf, act=act)
        self.refine3 = RefineBlock([2 * self.nf, 2 * self.nf], 2 * self.nf, act=act)
        self.refine31 = RefineBlock([2 * self.nf, 2 * self.nf], 2 * self.nf, act=act)
        self.refine4 = RefineBlock([2 * self.nf, 2 * self.nf], self.nf, act=act)
        self.refine5 = RefineBlock([self.nf, self.nf], self.nf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.centered:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer31 = self._compute_cond_module(self.res31, layer3)
        layer4 = self._compute_cond_module(self.res4, layer31)
        layer5 = self._compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref31 = self.refine31([layer31, ref2], layer31.shape[2:])
        ref3 = self.refine3([layer3, ref31], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output
