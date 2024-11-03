# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Definition of generic operator strategy."""
# pylint: disable=invalid-name,unused-argument
import logging
import re

from tvm import _ffi, ir, te, topi
from tvm.target import generic_func, override_native_generic_func
from tvm.topi.utils import get_const_float, get_const_int, get_const_tuple, get_float_tuple

from tvm.relay.op import op as _op

from ..compute import conv2d as _compute_conv2d
from ..compute import depthwise_conv2d as _compute_depthwise_conv2d

logger = logging.getLogger("strategy")


def naive_schedule(_, outs, target):
    """Return the naive default schedule.
    This function acts as a placeholder for op implementations that uses auto-scheduler.
    Implemenations using this function should only be used along with auto-scheduler.
    """
    if "gpu" in target.keys:
        # For GPU, we at least need thread binding to make a valid schedule.
        # So the naive schedule cannot be compiled.
        logger.debug(
            "Cannot compile for GPU targets if no tuned schedule is found. "
            "Please see the warning messages above for more information about the failed workloads."
        )
    return te.create_schedule(outs[-1].op)


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper


def wrap_topi_compute(topi_compute):
    """Wrap TOPI compute which doesn't use attrs"""

    def wrapper(attrs, inputs, out_type):
        return [topi_compute(*inputs)]

    return wrapper


def get_conv2d_in_channels(data_shape, data_layout):
    """Get conv2d input channels"""
    data_shape = get_const_tuple(data_shape)
    if len(data_shape) == 4:
        idx = data_layout.find("C")
        assert idx >= 0, f"Invalid conv2d data layout {data_layout}"
        return data_shape[idx]
    if re.match(r"NCHW\d*c", data_layout):
        # NCHW[8]c
        return data_shape[1] * data_shape[4]
    raise ValueError(f"Unknown conv2d data layout {data_layout}")


def get_conv2d_out_channels(kernel_shape, kernel_layout):
    """Get conv2d output channels"""
    kernel_shape = get_const_tuple(kernel_shape)
    if len(kernel_shape) == 4:
        idx = kernel_layout.find("O")
        assert idx >= 0, f"Invalid conv2d kernel layout {kernel_layout}"
        return kernel_shape[idx]
    if re.match(r"OIHW\d*i\d*o", kernel_layout):
        return kernel_shape[0] * kernel_shape[5]
    if re.match(r"OIHW\d*o", kernel_layout):
        return kernel_shape[0] * kernel_shape[4]
    raise ValueError(f"Unknown conv2d kernel layout {kernel_layout}")


def is_depthwise_conv2d(data_shape, data_layout, kernel_shape, kernel_layout, groups):
    ic = get_conv2d_in_channels(data_shape, data_layout)
    oc = get_conv2d_out_channels(kernel_shape, kernel_layout)
    return ic == oc == groups

get_auto_scheduler_rewritten_layout = _ffi.get_global_func(
    "relay.attrs.get_auto_scheduler_rewritten_layout"
)
get_meta_schedule_original_shape = _ffi.get_global_func(
    "relay.attrs.get_meta_schedule_original_shape"
)

# conv2d
def wrap_compute_conv2d(
    topi_compute,
    *,
    need_data_layout=False,
    need_kernel_layout=False,
    need_out_layout=False,
    has_groups=False,
    need_auto_scheduler_layout=False,
    need_meta_schedule_layout=False,
):
    """Wrap conv2d topi compute"""

    def _compute_conv2d(attrs, inputs, out_type):
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        dilation = get_const_tuple(attrs.dilation)
        data_layout = attrs.get_str("data_layout")
        kernel_layout = attrs.get_str("kernel_layout")
        out_layout = attrs.get_str("out_layout")
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        args = [inputs[0], inputs[1], strides, padding, dilation]
        if has_groups:
            args.append(attrs.groups)
        if need_data_layout:
            args.append(data_layout)
        if need_kernel_layout:
            args.append(kernel_layout)
        if need_out_layout:
            args.append(out_layout)
        args.append(out_dtype)
        if need_auto_scheduler_layout:
            args.append(get_auto_scheduler_rewritten_layout(attrs))
        elif need_meta_schedule_layout:
            args.append("")
            args.append(get_meta_schedule_original_shape(attrs))
        return [topi_compute(*args)]

    return _compute_conv2d

from tvm.relay.op.strategy.generic import conv2d_strategy

@conv2d_strategy.register("partial_conv")
def conv2d_strategy(attrs, inputs, out_type, target):
    """conv2d generic strategy"""
    logger.warning("conv2d is not optimized for this platform.")
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(_compute_conv2d.conv2d_nchw),
                wrap_topi_schedule(topi.generic.schedule_conv2d_nchw),
                name="conv2d_nchw.partial_conv",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                wrap_topi_schedule(topi.generic.schedule_conv2d_nhwc),
                name="conv2d_nhwc.partial_conv",
            )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_hwcn),
                wrap_topi_schedule(topi.generic.schedule_conv2d_hwcn),
                name="conv2d_hwcn.partial_conv",
            )
        else:
            raise RuntimeError(f"Unsupported conv2d layout {layout}")
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(_compute_depthwise_conv2d.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.partial_conv",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.partial_conv",
            )
        else:
            raise RuntimeError(f"Unsupported depthwise_conv2d layout {layout}")
    else:  # group_conv2d
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.partial_conv",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nhwc, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nhwc),
                name="group_conv2d_nhwc.partial_conv",
            )
        else:
            raise RuntimeError(f"Unsupported group_conv2d layout {layout}")
    return strategy
