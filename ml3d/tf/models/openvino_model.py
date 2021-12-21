import os
import sys
import subprocess

import numpy as np

from openvino.inference_engine import IECore


class OpenVINOModel:
    """Class providing OpenVINO backend for TensorFlow models.

    Class performs conversion to OpenVINO IR format using Model Optimizer tool.
    """

    def __init__(self, base_model):
        self.ie = IECore()
        self.exec_net = None
        self.base_model = base_model
        self.input_names = None
        self.input_ids = None
        self.device = "CPU"

    def _read_tf_model(self, inputs):
        # Dump a model in SavedModel format
        self.base_model.save('model')

        # Read input signatures (names, shapes)
        signatures = self.base_model._get_save_spec()

        self.input_names = []
        self.input_ids = []
        input_shapes = []
        for inp in signatures:
            inp_idx = int(inp.name[inp.name.find('_') + 1:]) - 1
            shape = list(inputs[inp_idx].shape)

            if np.prod(shape) == 0:
                continue

            self.input_names.append(inp.name)
            self.input_ids.append(inp_idx)
            input_shapes.append(str(shape))

        # Run Model Optimizer to get IR
        subprocess.run([
            sys.executable,
            '-m',
            'mo',
            '--saved_model_dir=model',
            '--input_shape',
            ','.join(input_shapes),
            '--input',
            ','.join(self.input_names),
            '--extension',
            os.path.join(os.path.dirname(__file__), 'utils', 'openvino_ext'),
        ],
                       check=True)

        self.exec_net = self.ie.load_network('saved_model.xml', self.device)

        try:
            os.remove('saved_model.xml')
            os.remove('saved_model.bin')
            os.rmdir('model')
        except Exception:
            pass

    def __call__(self, inputs):
        if self.exec_net is None:
            self._read_tf_model(inputs)

        tensors = {}
        for idx, name in zip(self.input_ids, self.input_names):
            tensors[name] = inputs[idx]

        output = self.exec_net.infer(tensors)
        output = next(iter(output.values()))
        return output

    def to(self, device):
        self.device = device.upper()
