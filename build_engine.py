#!/usr/bin/env python3
# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import argparse

def build_profile(builder, network, profile_shapes, default_shape_value=1):
    """
    Build optimization profile for the builder and configure the min, opt, max shapes appropriately.
    """
    def is_dimension_dynamic(dim):
        return dim is None or dim <= 0

    def override_shape(shape):
        return tuple([1 if is_dimension_dynamic(dim) else dim for dim in shape])

    profile = builder.create_optimization_profile()
    for idx in range(network.num_inputs):
        inp = network.get_input(idx)

        def get_profile_shape(name):
            if name not in profile_shapes:
                return None
            shapes = profile_shapes[name]
            if not isinstance(shapes, list) or len(shapes) != 3:
                G_LOGGER.critical("Profile values must be a list containing exactly 3 shapes (tuples or Dims), but received shapes: {:} for input: {:}.\nNote: profile was: {:}.\nNote: Network inputs were: {:}".format(shapes, name, profile_shapes, get_network_inputs(network)))
            return shapes

        if inp.is_shape_tensor:
            shapes = get_profile_shape(inp.name)
            if not shapes:
                rank = inp.shape[0]
                shapes = [(default_shape_value, ) * rank] * 3
                print("Setting shape input to {:}. If this is incorrect, for shape input: {:}, please provide tuples for min, opt, and max shapes containing {:} elements".format(shapes[0], inp.name, rank))
            min, opt, max = shapes
            profile.set_shape_input(inp.name, min, opt, max)
            print("Setting shape input: {:} values to min: {:}, opt: {:}, max: {:}".format(inp.name, min, opt, max))
        elif -1 in inp.shape:
            shapes = get_profile_shape(inp.name)
            if not shapes:
                shapes = [override_shape(inp.shape)] * 3
                print("Overriding dynamic input shape {:} to {:}. If this is incorrect, for input tensor: {:}, please provide tuples for min, opt, and max shapes containing values: {:} with dynamic dimensions replaced,".format(inp.shape, shapes[0], inp.name, inp.shape))
            min, opt, max = shapes
            profile.set_shape(inp.name, min, opt, max)
            print("Setting input: {:} shape to min: {:}, opt: {:}, max: {:}".format(inp.name, min, opt, max))
    if not profile:
        print("Profile is not valid, please provide profile data. Note: profile was: {:}".format(profile_shapes))
    return profile

def preprocess_network(network):
    """
    Add quantize and dequantize nodes after the input placeholder.
    The scale values are currently picked on emperical basis. Ideally,
    you need to add these nodes during quantization aware training and 
    learn the dynamic ranges of input node.
    """
    quant_scale = np.array([1.0/127.0], dtype=np.float32)
    dequant_scale = np.array([127.0/1.0], dtype=np.float32)
    # Zero point is always zero for quantization in TensorRT.
    zeros = np.zeros(shape=(1, ), dtype=np.float32)
 
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        # Find layer consuming input tensor
        found = False
        for layer in network:
            if found:
                break;

            for k in range(layer.num_inputs):
                if (inp == layer.get_input(k)):
                    mode = trt.ScaleMode.UNIFORM
                    quantize = network.add_scale(inp, mode, scale=quant_scale, shift=zeros)
                    quantize.set_output_type(0, trt.int8)
                    quantize.name = "InputQuantizeNode"
                    quantize.get_output(0).name = "QuantizedInput"
                    dequantize = network.add_scale(quantize.get_output(0), mode, scale=dequant_scale, shift=zeros)
                    dequantize.set_output_type(0, trt.float32)
                    dequantize.name = "InputDequantizeNode"
                    dequantize.get_output(0).name = "DequantizedInput"
                    layer.set_input(k, dequantize.get_output(0))
                    found = True
                    break

def build_engine_onnx(model_file, verbose=False):
    """
    Parse the model file through TensorRT, build TRT engine and run inference
    """
    # Create builder and network
    if verbose:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=network_flags) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
                
        # Add quantize and dequantize nodes for input of the network
        preprocess_network(network)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.flags = config.flags | 1 << int(trt.BuilderFlag.INT8)
        # Setting the (min, opt, max) batch sizes to be 1. Users need to configure this according to their requirements.
        config.add_optimization_profile(build_profile(builder, network, profile_shapes={'input' : [(1, 3, 224, 224),(1, 3, 224, 224),(1, 3, 224, 224)]}))

        return builder.build_engine(network, config)

def main(args):

    model_file = args.onnx
    # Parse the ONNX graph through TensorRT and build the engine
    trt_engine = build_engine_onnx(model_file, args.verbose)
    # Serialize the engine and save to file
    with open(args.engine, "wb") as file:
        file.write(trt_engine.serialize())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default='rn50.onnx', help="Path to RN50 ONNX graph")
    parser.add_argument("--engine", type=str,  default='rn50_trt.engine', help="output path to TensorRT engine")
    parser.add_argument('-v', '--verbose', action='store_true', help="Flag to enable verbose logging")
    args = parser.parse_args()
    main(args)
