#
#    Based on Second Library (MIT license):
#    https://github.com/traveller59/second.pytorch
#
#    Copyright (c) 2018 Yan Yan

#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:

#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#
#***************************************************************************************/

import numpy as np
from functools import partial
from ..optimizers import OptimWrapper


class LRSchedulerStep(object):

    def __init__(self, fai_optimizer: OptimWrapper, total_step, lr_phases,
                 mom_phases):
        # if not isinstance(fai_optimizer, OptimWrapper):
        #     raise TypeError('{} is not a fastai OptimWrapper'.format(
        #         type(fai_optimizer).__name__))
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append(
                    (int(start * total_step),
                     int(lr_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.lr_phases.append(
                    (int(start * total_step), total_step, lambda_func))
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append(
                    (int(start * total_step),
                     int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append(
                    (int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0

    def step(self, step):
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    # print(pct, start, end)
    """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycleScheduler(LRSchedulerStep):

    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor,
                 pct_start):
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        low_lr = self.lr_max / self.div_factor
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)),
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4)))
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)
