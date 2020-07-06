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

import onnx_graphsurgeon as gs
import argparse
import onnx
import numpy as np

def process_transpose_nodes(graph):
    """
    This is a workaround to manually transpose the conv weights and remove
    the existing transpose nodes. Currently TRT has a limitation when there is
    a transpose node as an input to the weights of the conv layer. This utility 
    would be removed in future releases.
    """
    # Find all the transposes before the convolutional nodes
    conv_nodes = [node for node in graph.nodes if node.op == "Conv"]
    for node in conv_nodes:
        # Transpose the convolutional weights and reset them to the weights
        conv_weights_tensor = node.i(1).i().i().inputs[0]
        conv_weights_transposed = np.transpose(conv_weights_tensor.values, [3, 2, 0, 1])
        conv_weights_tensor.values = conv_weights_transposed
        
        # Remove the transpose nodes after the dequant node. TensorRT does not support transpose nodes after QDQ nodes.
        dequant_node_output = node.i(1).i(0).outputs[0]
        node.inputs[1] = dequant_node_output

    # Remove unused nodes, and topologically sort the graph.
    return graph.cleanup().toposort()

if __name__=='__main__':
    parser = argparse.ArgumentParser("Post process ONNX graph by removing transpose nodes")
    parser.add_argument("--input", required=True, help="Input onnx graph")
    parser.add_argument("--output", default='postprocessed_rn50.onnx', help="Name of post processed onnx graph")
    args = parser.parse_args()
    
    # Load the rn50 graph
    graph = gs.import_onnx(onnx.load(args.input))

    # Remove the transpose nodes and reshape the convolution weights 
    graph = process_transpose_nodes(graph)
    
    # Export the onnx graph from graphsurgeon
    onnx_model = gs.export_onnx(graph)
    print("Output ONNX graph generated: ", args.output)
    onnx.save_model(onnx_model, args.output)