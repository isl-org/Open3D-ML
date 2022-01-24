import io
import copy

import torch

from openvino.inference_engine import IECore

from .. import dataloaders


def pointpillars_extract_feats(self, x):
    x = self.backbone(x)
    x = self.neck(x)
    return x


class OpenVINOModel:
    """Class providing OpenVINO backend for PyTorch models.

    OpenVINO model is initialized from ONNX representation of PyTorch graph.
    """

    def __init__(self, base_model):
        self.ie = IECore()
        self.ie.add_extension(
            "/home/dkurt/openvino_pytorch_unpool/user_ie_extensions/build/libuser_cpu_extension.so",
            "CPU")
        self.exec_net = None
        self.base_model = base_model
        self.device = "CPU"

        # A workaround for unsupported torch.square by ONNX
        torch.square = lambda x: torch.pow(x, 2)

    def _get_input_names(self, inputs):
        names = []
        for name, tensor in inputs.items():
            if isinstance(tensor, list):
                for i in range(len(tensor)):
                    names.append(name + str(i))
            else:
                names.append(name)
        return names

    def _get_inputs(self, inputs, export=False):
        if isinstance(inputs, dataloaders.concat_batcher.KPConvBatch):
            inputs = {
                'features': inputs.features,
                'points': inputs.points,
                'neighbors': inputs.neighbors,
                'pools': inputs.pools,
                'upsamples': inputs.upsamples,
            }
        elif isinstance(inputs, dataloaders.concat_batcher.ObjectDetectBatch):
            voxels, num_points, coors = self.base_model.voxelize(inputs.point)
            voxel_features = self.base_model.voxel_encoder(
                voxels, num_points, coors)
            batch_size = coors[-1, 0].item() + 1
            x = self.base_model.middle_encoder(voxel_features, coors,
                                               batch_size)

            inputs = {
                'x': x,
            }
        elif isinstance(inputs, dataloaders.concat_batcher.SparseConvUnetBatch):
            pos_list = []
            feat_list = []
            index_map_list = []

            for i in range(len(inputs.batch_lengths)):
                pos = inputs.point[i]
                feat = inputs.feat[i]
                feat, pos, index_map = self.base_model.input_layer(feat, pos)
                pos_list.append(pos)
                feat_list.append(feat)
                index_map_list.append(torch.tensor(index_map, dtype=torch.long))

            inputs = {
                'feat': feat_list,
                'pos': pos_list,
                'index_map_list': index_map_list,
            }
        elif not isinstance(inputs, dict):
            raise TypeError(f"Unknown inputs type: {inputs.__class__}")
        return inputs

    def _read_torch_model(self, inputs):
        inputs = copy.deepcopy(inputs)
        tensors = self._get_inputs(inputs)
        input_names = self._get_input_names(tensors)

        # Forward origin inputs instead of export <tensors>
        origin_forward = self.base_model.forward
        self.base_model.forward = lambda x: origin_forward(inputs)
        if self.base_model.__class__.__name__ == "SparseConvUnet":

            def sparseconvunet_input_layer(feat, pos):
                feat = tensors["feat"].pop(0)
                pos = tensors["pos"].pop(0)
                index_map = tensors["index_map_list"].pop(0)
                tensors["feat"].append(feat)
                tensors["pos"].append(pos)
                tensors["index_map_list"].append(index_map)
                return feat, pos, index_map

            self.base_model.input_layer.forward = sparseconvunet_input_layer
        elif self.base_model.__class__.__name__ == "PointPillars":
            self.base_model.extract_feats = lambda *args: pointpillars_extract_feats(
                self.base_model, tensors[input_names[0]])

        buf = io.BytesIO()
        self.base_model.device = torch.device('cpu')
        self.base_model.eval()
        torch.onnx.export(self.base_model,
                          tensors,
                          buf,
                          operator_export_type=torch.onnx.OperatorExportTypes.
                          ONNX_FALLTHROUGH,
                          input_names=input_names)

        self.base_model.forward = origin_forward

        net = self.ie.read_network(buf.getvalue(), b'', init_from_buffer=True)
        self.exec_net = self.ie.load_network(net, str(self.device).upper())

    def forward(self, inputs):
        if self.exec_net is None:
            self._read_torch_model(inputs)

        inputs = self._get_inputs(inputs)

        tensors = {}
        for name, tensor in inputs.items():
            if name == 'labels':
                continue
            if isinstance(tensor, list):
                for i in range(len(tensor)):
                    if tensor[i].nelement() > 0:
                        tensors[name + str(i)] = tensor[i].detach().numpy()
            else:
                if tensor.nelement() > 0:
                    tensors[name] = tensor.detach().numpy()

        output = self.exec_net.infer(tensors)

        if len(output) == 1:
            output = next(iter(output.values()))
            return torch.tensor(output)
        else:
            return tuple([torch.tensor(out) for out in output.values()])

    def __call__(self, inputs):
        return self.forward(inputs)

    def load_state_dict(self, *args):
        self.base_model.load_state_dict(*args)

    def eval(self):
        pass

    @property
    def cfg(self):
        return self.base_model.cfg

    @property
    def classes(self):
        return self.base_model.classes

    def inference_end(self, *args):
        return self.base_model.inference_end(*args)

    def preprocess(self, *args):
        return self.base_model.preprocess(*args)

    def transform(self, *args):
        return self.base_model.transform(*args)

    def update_probs(self, *args):
        self.base_model.trans_point_sampler = self.trans_point_sampler
        return self.base_model.update_probs(*args)

    def to(self, device):
        self.device = device
