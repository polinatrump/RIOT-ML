import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay
from tvm.relay import dataflow_pattern as dfp
from tvm.relay.transform import function_pass, InferType
from tvm.micro import export_model_library_format

import copy

from .fusion_block_iteratee import create_fusion_block_iteratee
from .dyn_slice_fixed_size import dyn_slice_fixed_size

from .op.copy_inplace import copy_in_place_spatial
from .op.fusion_iter_worker import fusion_iter_worker

from .utils import CollectOpShapeInfo

class MultiFusionNetworkRewriter(dfp.DFPatternCallback):
    def __init__(self):
        super().__init__()
        
        # Define the patterns for conv2d, avg_pool2d, and dense
        self.pool = dfp.is_op("nn.max_pool2d")(dfp.wildcard())
        self.conv2d = dfp.is_op("nn.conv2d")(dfp.wildcard(), dfp.wildcard())
        self.avg_pool2d = dfp.is_op("nn.avg_pool2d")(self.conv2d | self.pool)
        self.reshape = dfp.is_op("reshape")(self.avg_pool2d | self.conv2d)
        self.dense = dfp.is_op("nn.dense")(self.avg_pool2d | self.conv2d | self.reshape, dfp.wildcard())

        # Define the overall pattern: multiple conv2d -> avg_pool2d -> dense
        self.pattern = self.dense

        # To record input/output shapes and other attributes
        self.op_info = []

        self.external_funcs = None

    def callback(self, pre, post, node_map):
        # Step 2: Record Input/Output Shapes and Attributes
        
        conv_nodes = node_map[self.conv2d]
        avg_pool_node = node_map[self.avg_pool2d][0]
        dense_node = node_map[self.dense][0]
        reshape_node = node_map[self.reshape][0]
        
        op_shape_collector = CollectOpShapeInfo()
        op_shape_collector.visit(dense_node)
        self.op_info = op_shape_collector.op_info

        return dense_node
    

class MultiFusionConv2DWorker(relay.ExprMutator):
    def __init__(self, new_input):
        super().__init__()
        self.new_input = new_input  # The new input expression to replace the original input

    def visit_call(self, call):
        # Check if the call node is a `conv2d` operation
        for arg in call.args:
            self.visit(arg)
        if isinstance(call.op, relay.op.Op) and call.op.name == "nn.conv2d":
            # Replace the input of the `conv2d` with `self.new_input`
            modified_conv = relay.nn.conv2d(self.new_input, call.args[1], **call.attrs)
            return modified_conv

        # For other operations, continue visiting as usual
        return call