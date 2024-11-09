import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay, topi
import tvm.te

from tvm.ir import Attrs
import tvm.relay.op.op as _op
import copy
from tvm.target import generic_func, override_native_generic_func
import math
from tvm.ir import register_intrin_lowering, Op, register_op_attr


# Define the new operator in Relay
relay.op.op.register("copy_in_place_spatial")
op_name = "copy_in_place_spatial"

# call default relation functions
def copy_in_place_spatial_rel(args, attrs):
    return args[0]
_op.get(op_name).add_type_rel("CopyInPlacespatialTypeRel", copy_in_place_spatial_rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
_op.register_stateful(op_name, True)


def copy_in_place_spatial(dst, src, begin_indices, layout="NCHW"):
    attrs = tvm.ir.make_node("DictAttrs", layout=layout)
    # breakpoint()
    return relay.Call(relay.op.get("copy_in_place_spatial"), [dst, src, begin_indices], attrs)


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper



# Define the compute function for the my_add operator
# def wrap_iter_func_compute_tir(attrs, inputs, output_type):
#     def _iter_func_compute_tir(ins, outs):
#         ins_data = [i.data for i in ins]
#         begin = attrs["iter_begin"]
#         end = attrs["iter_end"]
#         strides = attrs["iter_strides"]
#         func = attrs["relay_func"]
#         ib = tvm.tir.ir_builder.create()
#         iterator = ib.buffer_ptr(ins[0])
#         iter_var = ib.buffer_ptr(ins[1])
#         iterator[2] = begin[2]
#         with ib.while_loop(iterator[2] < end[2]):
#             iterator[3] = begin[3]
#             with ib.while_loop(iterator[3] < end[3]):
#                 ib.emit(tvm.tir.call_extern("int32", 
#                     "tvmgen_default_" + func.attrs["global_symbol"],
#                         *ins_data, outs[0].data))
#                 ib.emit(tvm.tir.call_intrin("int32", 
#                     "tir.memcpy",
#                         ins_data[1], outs[0].data, math.prod(outs[0].shape) * dtype_bytes[outs[0].dtype]))
#                 iterator[3] += strides[3]
#             iterator[2] += strides[2]
#         return ib.get()
    
#     return _iter_func_compute_tir

from tvm.topi import utils

@relay.op.op.register_compute("copy_in_place_spatial")
def copy_in_place_spatial_compute(attrs, inputs, output_type):
    print("We are now at copy_in_place_spatial_compute")
    
    dst, src, indices= inputs

    # bg_x = utils.get_const_int(bg_x)
    # bg_y = utils.get_const_int(bg_y)
    
    size_x = src.shape[2]
    size_y = src.shape[3]
    size_c = src.shape[1]
    size_n = src.shape[0]
    def gen_ib(dst_buf, src_buf, indices_buf):
        ib = tvm.tir.ir_builder.create()
        dst = ib.buffer_ptr(dst_buf)
        src = ib.buffer_ptr(src_buf)
        indices = ib.buffer_ptr(indices_buf)
        with ib.for_range(0 , size_n, "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  size_c, "i") as i:
                with ib.for_range(0, size_x, "j") as j:
                    with ib.for_range(0, size_y, "k") as k:
                        dst[n + indices[0], i + indices[1], j + indices[2], k+ indices[3]] = src[n + indices[0], i + indices[1], j + indices[2], k+ indices[3]]

        return ib.get()
    out_buf = tvm.tir.decl_buffer(dst.shape, dst.dtype, dst.op.name, elem_offset=tvm.tir.Var("elem_offset", "int32"))
    
    return [tvm.te.extern((1,), [dst, src, indices],
               lambda ins, outs: gen_ib(ins[0], ins[1], ins[2]),
            name="copy_in_place_spatial_compute.generic", 
            # out_buffers=[out_buf],
            dtype=output_type.dtype,
            )]

@override_native_generic_func("copy_in_place_spatial_strategy")
def copy_in_place_spatial_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        copy_in_place_spatial_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="iter_func.generic",
    )
    return strategy
_op.register_strategy("copy_in_place_spatial", copy_in_place_spatial_strategy)

if __name__ == "__main__":
    dst = relay.var("A", shape=(1,4, 12, 12))
    src = relay.var("B", shape=(1, 4, 2, 2))
    bg_indices = relay.var("ind", shape=(4,) , dtype="int32")
    body = copy_in_place_spatial(dst, src, bg_indices) 
    body = relay.Tuple([body + relay.const(2.0), dst])
    func = relay.Function([dst, src, bg_indices], body)

    mod = tvm.IRModule.from_expr(func)

    mod = relay.transform.InferType()(mod)

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

    from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll


    with tvm.transform.PassContext(opt_level=0, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True, # what is usmp? -> Enable Unified Static Memory Planning
                                                    "tir.usmp.algorithm": "hill_climb",
                                                    "relay.backend.use_auto_scheduler": True,
                                                    },
                                                    # instruments=[PrintBeforeAll(),PrintAfterAll()]
                                                    ): 
        # print(params.keys())
        # opt_module, _ = relay.optimize(mod, target=TARGET)
        module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)
        # breakpoint()
    from tvm.micro import export_model_library_format

    export_model_library_format(module, "./test_copy_inplace_spatial.tar")
