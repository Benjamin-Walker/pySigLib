# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from abc import ABC, abstractmethod
import torch

class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.saved_for_y = ()

    def save_for_backward(self, *args):
        self.saved_tensors = args

    def save_for_grad_y(self, *args):
        self.saved_for_y = args

class AmbientKernel(ABC):

    @abstractmethod
    def __call__(self, ctx, x, y):
        pass

    @abstractmethod
    def grad_x(self, ctx, derivs):
        pass

    @abstractmethod
    def grad_y(self, ctx, derivs):
        pass

class LinearKernel(AmbientKernel):

    def __call__(self, ctx, x, y):
        dx = torch.diff(x, dim=1)
        dy = torch.diff(y, dim=1)
        ctx.save_for_backward(dx, dy)
        return torch.bmm(dx, dy.permute(0, 2, 1))

    def grad_x(self, ctx, derivs):
        dx, dy = ctx.saved_tensors
        out = torch.empty((dx.shape[0], dx.shape[1] + 1, dy.shape[1]), dtype=torch.float64, device=derivs.device)
        out[:, 0, :] = 0
        out[:, 1:, :] = derivs
        out[:, :-1, :] -= derivs
        return torch.bmm(out, dy)

    def grad_y(self, ctx, derivs):
        dx, dy = ctx.saved_tensors
        out = torch.empty((dx.shape[0], dx.shape[1], dy.shape[1] + 1), dtype=torch.float64, device=derivs.device)
        out[:, :, 0] = 0
        out[:, :, 1:] = derivs
        out[:, :, :-1] -= derivs
        return torch.bmm(out.permute(0, 2, 1), dx)

class ScaledLinearKernel(AmbientKernel):
    def __init__(self, scale : float = 1.):
        self.linear_kernel = LinearKernel()
        self.scale = scale
        self._scale_sq = scale ** 2

    def __call__(self, ctx, x, y):
        return self.linear_kernel(ctx, x * self._scale_sq, y)

    def grad_x(self, ctx, derivs):
        return self.linear_kernel.grad_x(ctx, derivs) * self._scale_sq

    def grad_y(self, ctx, derivs):
        return self.linear_kernel.grad_y(ctx, derivs)

class RBFKernel(AmbientKernel):
    def __init__(self, sigma : float):
        self.sigma = sigma
        self._one_over_sigma = 1. / sigma
        self._scale = 2 * self._one_over_sigma

    def __call__(self, ctx, x, y):
        dist = torch.bmm(x * self._scale, y.permute(0, 2, 1))

        x2 = torch.pow(x, 2)
        y2 = torch.pow(y, 2)
        x2 = torch.sum(x2, dim=2) * self._one_over_sigma
        y2 = torch.sum(y2, dim=2) * self._one_over_sigma

        dist -= torch.reshape(x2, (x.shape[0], x.shape[1], 1)) + torch.reshape(y2, (x.shape[0], 1, y.shape[1]))
        torch.exp(dist, out=dist)

        ctx.save_for_backward(x, y, dist.clone())

        buff = torch.empty_like(dist[:, :-1, :])
        torch.diff(dist, dim=1, out=buff)
        dist.resize_((dist.shape[0], dist.shape[1] - 1, dist.shape[2] - 1))
        torch.diff(buff, dim=2, out=dist)
        return dist

    def grad_x(self, ctx, derivs):
        x, y, out = ctx.saved_tensors

        dout = torch.zeros_like(out)
        dout[:, 1:, 1:] += derivs
        dout[:, :-1, :-1] += derivs
        dout[:, 1:, :-1] -= derivs
        dout[:, :-1, 1:] -= derivs

        dout *= out
        dout *= 2. * self._one_over_sigma

        ctx.save_for_grad_y(x, y, dout)
        return torch.bmm(dout, y) - x * torch.sum(dout, dim=2).unsqueeze(-1)

    def grad_y(self, ctx, derivs):

        if ctx.saved_for_y:
            x, y, dout = ctx.saved_for_y#
        else:
            x, y, out = ctx.saved_tensors

            dout = torch.zeros_like(out)
            dout[:, 1:, 1:] += derivs
            dout[:, :-1, :-1] += derivs
            dout[:, 1:, :-1] -= derivs
            dout[:, :-1, 1:] -= derivs

            dout *= out
            dout *= 2. * self._one_over_sigma

        return torch.bmm(dout.permute(0, 2, 1), x) - y * torch.sum(dout, dim=1).unsqueeze(-1)