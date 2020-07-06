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

import argparse
from tensorflow.core.protobuf import config_pb2, rewriter_config_pb2, meta_graph_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer, ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver


def constfold(graphdef, output_name):
    graph = ops.Graph()
    with graph.as_default():
        outputs = output_name.split(',')
        output_collection = meta_graph_pb2.CollectionDef()
        output_list = output_collection.node_list.value
        for output in outputs:
            output_list.append(output)
        importer.import_graph_def(graphdef, name="")
        metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
        metagraph.collection_def["train_op"].CopyFrom(output_collection)
        
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.optimizers.extend(["constfold"])
    rewriter_config.meta_optimizer_iterations = (rewriter_config_pb2.RewriterConfig.ONE)
    session_config = config_pb2.ConfigProto()
    session_config.graph_options.rewrite_options.CopyFrom(rewriter_config)
    
    return tf_optimizer.OptimizeGraph(session_config, metagraph)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Folds constants in the provided frozen model")
    parser.add_argument("-i", "--input", help="The input frozen model to be constant folded.")
    parser.add_argument("--output_node", default="resnet50/output/softmax_1", help="Output node names separated by commas")
    parser.add_argument("-o", "--output", default="folded_rn50.pb", help="Path to constant folded output graph")
    args, _ = parser.parse_known_args()

    with open(args.input, 'rb') as f:
        graphdef = graph_pb2.GraphDef()
        graphdef.ParseFromString(f.read())

    folded_graph = constfold(graphdef, args.output_node)
    print("Writing output to {:}".format(args.output))
    with open(args.output, "wb") as f:
        f.write(folded_graph.SerializeToString())
