from mo.front.extractor import FrontExtractorOp
from extensions.ops.ReduceOps import ReduceMin
from mo.graph.graph import Node


class MinFrontExtractor(FrontExtractorOp):
    op = 'Min'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceMin.update_node_stat(node,
                                   {'keep_dims': node.pb.attr['keep_dims'].b})
        return cls.enabled
