import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay, topi
import tvm.te

def int_list(l):
    return [int(i) for i in l]

class CollectOpShapeInfo(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.op_info = []
        self.op_to_info_map_ = {}

    def collect_op_info(self, op):
        input_shape = tuple(int(dim) for dim in op.args[0].checked_type.shape)
        output_shape = tuple(int(dim) for dim in op.checked_type.shape)
        kernel_size = int_list([*getattr(op.attrs, 'kernel_size', getattr(op.attrs, 'pool_size', [1 ,1]))])
        padding = int_list([*getattr(op.attrs, 'padding', [0, 0])])
        strides = int_list([*getattr(op.attrs, 'strides', [1, 1])])
        groups = int(getattr(op.attrs, 'groups', 1))
        op_info_ = {
            'op_name': op.op.name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'kernel_size': kernel_size,
            'padding': padding,
            'strides': strides,
            'groups': groups,
            'input_tile_size': None,
            'input_tile_strides': None,
            'first_conv': False
        }
        if op.op.name == 'nn.conv2d':
            op_info_['first_conv'] = True
            for i in self.op_info:
                i["first_conv"] = False
        self.op_info.append(op_info_)
        self.op_to_info_map_[op] = op_info_

    def visit_call(self, call):
        self.collect_op_info(call)
        for arg in call.args:
            self.visit(arg)

    # def op_info(self):
    #     return self.op_info
    
    @property
    def op_to_info_map(self):
        return self.op_to_info_map_
    

class ReWriteInputsShape(relay.ExprMutator):
    """This pass partitions the subgraph based on the if conditin

    """
    def __init__(self, name_to_shape):
        super().__init__()
        self.name_to_shape = name_to_shape

    def visit_function(self, fn):
        """This function returns concatenated add operators for a one add operator.
        It creates multiple add operators.

        :param call:
        :return:
        """

        new_params = []
        for x in range(len(fn.params)):
            new_params.append(self.visit(fn.params[x]))

        new_body = self.visit(fn.body)
        func = relay.Function(list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs)
        return func

    def visit_var(self, var):
        if var.name_hint in self.name_to_shape:
            print(f'Change Shape of params {var.name_hint}, {var.type_annotation.shape} to {self.name_to_shape[var.name_hint]}')
            d = self.name_to_shape[var.name_hint]
            var_new = relay.var(var.name_hint, shape=d, dtype=var.type_annotation.dtype)
            return var_new
        else:
            print("Do nothing for other cases")
            return var

class InferCallNodeType(relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)
        relay.transform.InferTypeLocal(call)
        return call
    
    def visit_var(self, var):
        relay.transform.InferTypeLocal(var)
        return var
    
    def visit_constant(self, cnst):
        relay.transform.InferTypeLocal(cnst)
        return cnst

    def visit_function(self, fn):
        for p in fn.params:
            self.visit(p)
        self.visit(fn.body)
        relay.transform.InferTypeLocal(fn)
        return fn
    
class EliminateIterateeDummyCallNode(relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def visit_call(self, call):
        if getattr(call.attrs, 'iteratee_dummy', 0) == 1:
            breakpoint()
            return relay.zeros((1,), "float32")
        else:
            return call
    def visit_var(self, var):
        return var
    
        
class ReWriteSwapVars(relay.ExprMutator):
    """This pass partitions the subgraph based on the if conditin

    """
    def __init__(self, name_to_var):
        super().__init__()
        self.name_to_shape = name_to_var

    def visit_var(self, var):
        if var.name_hint in self.name_to_shape:
            print(f'Change Shape of params {var.name_hint}, {var.type_annotation.shape} to {self.name_to_shape[var.name_hint]}')
            d = self.name_to_shape[var.name_hint]
            return d
        else:
            print("Do nothing for other cases")
            return var


class Conv2DInputReplacer(relay.ExprMutator):
    def __init__(self, new_input):
        super().__init__()
        self.new_input = new_input  # The new input expression to replace the original input

    def visit_call(self, call):
        # Check if the call node is a `conv2d` operation
        if isinstance(call.op, relay.op.Op) and call.op.name == "nn.conv2d":
            # Replace the input of the `conv2d` with `self.new_input`
            modified_conv = relay.nn.conv2d(self.new_input, call.args[1], **call.attrs)
            return modified_conv

        # For other operations, continue visiting as usual
        return call