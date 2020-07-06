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

import os
import argparse
import PIL.Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import image_processing

TRT_DYNAMIC_DIM = -1

def load_normalized_test_case(test_image, pagelocked_buffer, preprocess_func):
    # Expected input dimensions
    C, H, W = (3, 224, 224)
    # Normalize the images, concatenate them and copy to pagelocked memory.
    data = np.asarray([preprocess_func(PIL.Image.open(test_image).convert('RGB'), C, H, W)]).flatten()
    np.copyto(pagelocked_buffer, data)

class HostDeviceMem(object):
    r""" Simple helper data class that's a little nicer to use than a 2-tuple.
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine, batch_size: int):
    print('Allocating buffers ...')

    inputs = []
    outputs = []
    dbindings = []

    stream = cuda.Stream()

    for binding in engine:
        size = batch_size * abs(trt.volume(engine.get_binding_shape(binding)))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        dbindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, dbindings, stream

def infer(engine_path, preprocess_func, batch_size, input_image, labels=[], verbose=False):
    
    if verbose:
        logger = trt.Logger(trt.Logger.VERBOSE)
    else:
        logger = trt.Logger(trt.Logger.INFO)
        
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        
        def override_shape(shape, batch_size):
            return tuple([batch_size if dim==TRT_DYNAMIC_DIM else dim for dim in shape])

        # Allocate buffers and create a CUDA stream.
        inputs, outputs, dbindings, stream = allocate_buffers(engine, batch_size)

        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Resolve dynamic shapes in the context
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                shape = engine.get_binding_shape(binding_idx)
                if engine.binding_is_input(binding_idx):
                    if TRT_DYNAMIC_DIM in shape:
                        shape = override_shape(shape, batch_size)
                    context.set_binding_shape(binding_idx, shape)

            # Load the test images and preprocess them
            load_normalized_test_case(input_image, inputs[0].host, preprocess_func)
                
            # Transfer input data to the GPU.
            cuda.memcpy_htod(inputs[0].device, inputs[0].host)
            # Run inference.
            context.execute(batch_size, dbindings)
            # Transfer predictions back to host from GPU
            out = outputs[0]
            cuda.memcpy_dtoh(out.host, out.device)

            softmax_output = np.array(out.host)
            top1_idx = np.argmax(softmax_output)
            output_class = labels[top1_idx+1]
            output_confidence = softmax_output[top1_idx]
            
            print ("Output class of the image: {} Confidence: {}".format(output_class, output_confidence))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on TensorRT engines for Imagenet-based Classification models.')
    parser.add_argument('-e', '--engine', type=str, required=True,
                        help='Path to RN50 TensorRT engine')
    parser.add_argument('-i', '--image', required=True, type=str,
                        help="Path to input image.")
    parser.add_argument("-l", "--labels", type=str, default=os.path.join("labels", "class_labels.txt"),
                        help="Path to file which has imagenet 1k labels.")
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        help="Batch size of inputs")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Flag to enable verbose loggin")
    args = parser.parse_args()
    
    # Class 0 is not used and is treated as background class. Renaming it to "background"
    with open(args.labels, "r") as f:
        background_class = ["background"]
        imagenet_synsets = f.read().splitlines()
        imagenet_classes=[]
        for synset in imagenet_synsets:
            class_name = synset.strip()
            imagenet_classes.append(class_name)
        all_classes = background_class + imagenet_classes
        labels = np.array(all_classes)
        
    # Preprocessing for input images
    preprocess_func = image_processing.preprocess_resnet50
    
    # Run inference on the test image
    infer(args.engine, preprocess_func, args.batch_size, args.image, labels, args.verbose)
    
