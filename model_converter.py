import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
import tvm.micro
import tvm.micro.testing
from tvm import relay
import tvm.contrib.utils
from tvm.micro import export_model_library_format
from tvm.driver import tvmc

RIOT_BOARD_TO_TARGET = {
    'stm32f746g-disco': tvm.target.target.stm32('stm32F7xx'),

    'iotlab-m3' : tvm.target.target.stm32('stm32F1xx'),

    'samr21-xpro' : tvm.target.target.stm32('stm32L0xx'),
    'samr30-xpro' : tvm.target.target.stm32('stm32L0xx'),
    'samr34-xpro' : tvm.target.target.stm32('stm32L0xx'),

    'arduino-zero' : tvm.target.target.stm32('stm32L0xx'),

    'firefly': tvm.target.target.stm32('stm32F2xx'),

    'b-l072z-lrwan1' : tvm.target.target.stm32('stm32L0xx'),
    'b-l475e-iot01a' : tvm.target.target.stm32('stm32L4xx'),

    'nrf52dk' : tvm.target.target.micro('nrf52840'),
    'nrf52840dk' : tvm.target.target.micro('nrf52840'),

    'nucleo-wl55jc' : tvm.target.target.stm32('stm32L0xx'),
    'microbit' : tvm.target.target.stm32('stm32F0xx'),
    'openmote-b' : tvm.target.target.stm32('stm32F2xx'),
    'dwm1001' : tvm.target.target.micro('nrf52840'),

    'hifive1b' : 'c -keys=arm_cpu,cpu -device=arm_cpu -mcpu=sifive-e31 -model=sifive-e31',

}

def load_from_tflite(model_path : str):
    
    tflite_model_buf = open(model_path, "rb").read()

    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)

    return mod, params
    
def compile_per_model_eval(relay_mod, params, riot_board=None, mlf_path=None):
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
    TARGET = RIOT_BOARD_TO_TARGET.get(riot_board) or tvm.target.target.micro('host')
    with tvm.transform.PassContext(opt_level=3, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True
                                                    }): # what is usmp? -> Enable Unified Static Memory Planning
        module = relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params, executor=EXECUTOR)
    if mlf_path is not None:
        export_model_library_format(module, mlf_path)
    return module



def load_from_pytorch(model_path : str):
    pass

def load_from_onxx(model_path : str):
    pass

def load_from_mxnet(model_path : str):
    pass

def load_model(model_path: str):
    model = tvmc.load(model_path)
    return model.mod, model.params
