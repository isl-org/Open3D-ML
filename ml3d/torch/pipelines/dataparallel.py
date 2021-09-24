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

        if len(self.device_ids) == 1:
            if hasattr(inputs[0], 'to'):
                inputs[0].to(self.device_ids[0])
            return self.module(inputs[0], **kwargs)

        inputs, kwargs = self.customscatter(inputs, kwargs, self.device_ids)

        self.module.cuda()
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)

        return self.gather(outputs, self.output_device)

    def customscatter(self, inputs, kwargs, device_ids):
        """Custom scatter method to override default method.
        Scatter batch dimension based on custom scatter implemented
        in custom batcher.

        Agrs:
            inputs: Object of type custom batcher.
            kwargs: Optional keyword arguments.
            device_ids: List of device ids.

        Returns:
            Returns a list of inputs of length num_devices.
            Each input is transfered to different device id.
        """
        if not hasattr(inputs[0], 'scatter'):
            try:
                return self.scatter(inputs, kwargs, device_ids)
            except:
                raise NotImplementedError(
                    f"Please implement scatter for {inputs[0]} for multi gpu execution."
                )
        inputs = inputs[0].scatter(inputs[0], len(device_ids))
        for i in range(len(inputs)):
            inputs[i].to(torch.device(device_ids[i]))

        return inputs, [kwargs for _ in range(len(inputs))]
