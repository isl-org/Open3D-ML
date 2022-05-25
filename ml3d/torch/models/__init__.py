"""Networks for torch."""

from .randlanet import RandLANet
from .kpconv import KPFCNN
from .point_pillars import PointPillars
from .sparseconvnet import SparseConvUnet
from .point_rcnn import PointRCNN
from .point_transformer import PointTransformer
from .pvcnn import PVCNN
from .sparseconvnet_megamodel import SparseConvUnetMegaModel

__all__ = [
    'RandLANet', 'KPFCNN', 'PointPillars', 'PointRCNN', 'SparseConvUnet',
    'PointTransformer', 'PVCNN', 'SparseConvUnetMegaModel'
]

try:
    from .openvino_model import OpenVINOModel
    __all__.append("OpenVINOModel")
except Exception:
    pass
