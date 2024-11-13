import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay
from tvm.relay import dataflow_pattern as dfp
from tvm.relay.transform import function_pass, InferType
from tvm.micro import export_model_library_format
import numpy as np

from model_converter import compile_per_model_eval
from mlmci_utils import generate_mlmci_files
from partial_conv.relay_rewrite import rewrite_conv_chain_to_function

# from tvm.relay.op.strategy.generic import conv2d_strategy

# # conv2d_strategy.register("generic", conv2d_strategy.__wrapped__) # use unoptimized conv2d to avoid kernel_vec copy


from partial_conv.dyn_slice_fixed_size import dyn_slice_fixed_size
from partial_conv.op.copy_inplace import copy_in_place_spatial
from partial_conv.utils import CollectOpShapeInfo, ReWriteInputsShape, ReWriteSwapVars, InferCallNodeType, EliminateIterateeDummyCallNode



from partial_conv.op.fusion_iter_worker import fusion_iter_worker
from partial_conv.op.strategy import conv2d

from partial_conv.fusion_block_iteratee import create_fusion_block_iteratee_with_cache, create_zeros_for_vars

if __name__=="__main__":
    data = relay.var("data", shape=(1, 1, 28, 28))
    # data = relay.var("data", shape=(1, 3, 15, 15))
###################### Big ##############################333
    weight1 = relay.var("weight1", shape=(64, 1, 7, 7))
    weight2 = relay.var("weight2", shape=(64, 64, 3, 3))
    weight3 = relay.var("weight3", shape=(64, 64, 3, 3))
    dense_weight = relay.var("dense_weight", shape=(10, 64))

    params = {"weight1": tvm.nd.array(np.random.rand(64, 1, 7, 7).astype(np.float32)),
                "weight2": tvm.nd.array(np.random.rand(64, 64, 3, 3).astype(np.float32)),
                "weight3": tvm.nd.array(np.random.rand(64, 64, 3, 3).astype(np.float32)),
                "dense_weight": tvm.nd.array(np.random.rand(10, 64).astype(np.float32)),
                }

    conv1 = relay.nn.conv2d(data, weight1, kernel_size=(7, 7))
    conv2 = relay.nn.conv2d(conv1, weight2, kernel_size=(3, 3),
                             strides=(2,2), padding=(1,1)
                             )
    conv3 = relay.nn.conv2d(conv2, weight3, kernel_size=(3, 3))
    func = relay.Function([data, weight1, weight2, weight3], conv3)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    iteratee_func, conv_chain_params, cache_vars, new_input_layout, new_input_stride, output_shape, input_shape = create_fusion_block_iteratee_with_cache(mod["main"].body, 1)

    cache_zeros = create_zeros_for_vars(cache_vars)

    in_w = input_shape[2]
    in_h = input_shape[3]
    iter_begin = [0,0,0,0]
    iter_end = [0, 0, in_w - new_input_layout[0] + 1, in_h - new_input_layout[1] + 1]
    iter_strides = [1,1, *new_input_stride]
    # iterator = relay.zeros(shape=(4,), dtype="int32")
    # iter_output_var = relay.zeros(shape=(1,64,15,15), dtype="float32")

    # # dummy_var to keep the iteratee not to be optimized out
    # dummy_var = relay.Call(iteratee_func, [iterator, data, weight1, weight2, weight3], tvm.ir.make_node("DictAttrs", iteratee_dummy=1))
    # dummy_var = relay.zeros(shape=(4,), dtype="float32")
    
    iter_worker = fusion_iter_worker(iter_begin, iter_end, iter_strides, output_shape,
                                     conv_chain_params, iteratee_func, cache_vars=cache_vars)
    

    iter_body = relay.Function(relay.analysis.free_vars(iter_worker), iter_worker)

    # iter_body = EliminateIterateeDummyCallNode().visit(iter_body)

    iteratee_mod = tvm.IRModule.from_expr(iter_body)
    # iteratee_mod['iteratee'] = iteratee_func
    iteratee_mod = relay.transform.InferType()(iteratee_mod)


    # breakpoint()
    
    RUNTIME = tvm.relay.backend.Runtime("crt", {'system-lib':False}) # should not use 'system-lib:true' while AoT
    EXECUTOR = tvm.relay.backend.Executor(
        "aot",
        {
        "unpacked-api": True, 
        "interface-api": "c", 
        "workspace-byte-alignment": 4,
        "link-params": True,
        },
    )
    TARGET = tvm.target.target.micro('host')
    TARGET = "c -keys=partial_conv,arm_cpu,cpu -mcpu=cortex-m4+nodsp -model=nrf52840"

    from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll


    with tvm.transform.PassContext(opt_level=0, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True, # what is usmp? -> Enable Unified Static Memory Planning
                                                    "tir.usmp.algorithm": "greedy_by_size",
                                                    "relay.backend.use_auto_scheduler": True,
                                                    "relay.remove_standalone_reshapes.enable": False
                                                    },
                                                    # instruments=[PrintBeforeAll(),PrintAfterAll()]
                                                    ): 
        # print(params.keys())
        # opt_module, _ = relay.optimize(mod, target=TARGET)
        module = relay.build(iteratee_mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)

    from tvm.micro import export_model_library_format

    export_model_library_format(module, "./models/default/default.tar")
    generate_mlmci_files(module, params, "./")

