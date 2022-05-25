"""3D ML pipelines for torch."""

from .semantic_segmentation import SemanticSegmentation
from .object_detection import ObjectDetection
from .semantic_segmentation_multi_head import SemanticSegmentationMultiHead

__all__ = [
    'SemanticSegmentation', 'ObjectDetection', 'SemanticSegmentationMultiHead'
]
