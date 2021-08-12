import torch
import numpy as np
from torch.nn.parallel import DataParallel


class CustomDataParallel(DataParallel):
    """Custom DataParallel method for performing scatter operation
    outside of torch's DataParallel.
    """

    def __init__(self, module, **kwargs):
        super(CustomDataParallel, self).__init__(module, **kwargs)
        self.get_loss = self.module.get_loss
        self.cfg = self.module.cfg

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # self._sync_params()
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)

        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        if not hasattr(inputs[0], 'scatter'):
            raise NotImplementedError(
                f"Please implement scatter for {inputs[0]} for multi gpu execution."
            )
        inputs = inputs[0].scatter(inputs[0], len(self.device_ids))

        return inputs, [kwargs for _ in range(len(inputs))]
