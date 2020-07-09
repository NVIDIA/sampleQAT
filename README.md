# TensorRT inference of Resnet-50 trained with QAT.

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
    * [Step 1: Quantization Aware Training](#step-1-quantization-aware-training)
    * [Step 2: Export frozen graph of RN50 QAT](#step-2-export-frozen-graph-of-rn50-qat)
    * [Step 3: Constant folding](#step-3-constant-folding)
	* [Step 4: TF2ONNX conversion](#step-4-tf2onnx-conversion)
    * [Step 5: Post processing ONNX](#step-5-post-processing-onnx)
    * [Step 6: Build TensorRT engine from ONNX graph](#step-6-build-tensorrt-engine-from-onnx-graph)
    * [Step 7: TensorRT Inference](#step-7-tensorrt-inference)
- [Additional resources](#additional-resources)
- [Changelog](#changelog)
- [Known issues](#known-issues)
- [License](#license)

## Description

This sample demonstrates workflow for training and inference of Resnet-50 model trained using Quantization Aware Training.
The inference implementation is experimental prototype and is provided with no guarantee of support.

## How does this sample work?

This sample demonstrates

* Training a Resnet-50 model using quantization aware training.
* Post processing and conversion to ONNX graph to ensure it is successfully parsed by TensorRT.
* Inference of Resnet-50 QAT graph with TensorRT.

## Prerequisites

Dependencies required for this sample

1.  <a href="https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow">TensorFlow NGC containers</a> (20.01-tf1-py3 NGC container or above for Steps 1-4. Please use `tf1` variants which have TF 1.15.2 version installed.
    This sample does not work with public version of Tensorflow 1.15.2 library) 
        
2.  Install the dependencies for Python3 inside the NGC container.
	-   For Python 3 users, from the root directory, run:
		`python3 -m pip install -r requirements.txt`

3. TensorRT-7.1

4. <a href="https://github.com/NVIDIA/TensorRT/tree/release/7.1/tools/onnx-graphsurgeon">ONNX-Graphsurgeon 0.2.1</a>

## Running the sample

***NOTE: Steps 1-4 require <a href="https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow">NGC containers</a> (TensorFlow 20.01-tf1-py3 NGC container or above). Steps 5-7 can be executed within or outside the NGC container***

### Step 1: Quantization Aware Training

Please follow detailed instructions on how to <a href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quantization-aware-training">finetune a RN50 model using QAT</a>. 

This stage involoves 

* Finetune a RN50 model with quantization nodes and save the final checkpoint.
* Post process the above RN50 QAT checkpoint by reshaping the weights of final FC layer into a 1x1 conv layer.

### Step 2: Export frozen graph of RN50 QAT

Export the RN50 QAT graph replacing the final FC layer with a 1x1 conv layer. 
Please follow these <a href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#exporting-frozen-graphs">instructions</a> to generate a frozen graph in desired data formats.

### Step 3: Constant folding

Once we have the frozen graph from Step 2, run the following command to perform constant folding on TF graph
```
python fold_constants.py --input <input_pb> --output <output_pb_name>
```

Arguments:
* `--input` : Input Tensorflow graph
* `--output_node` : Output node name of the RN50 graph (Default: `resnet50_v1.5/output/softmax_1`)
* `--output` : Output name of constant folded TF graph.

### Step 4: TF2ONNX conversion

TF2ONNX converter is used to convert the constant folded tensorflow frozen graph into ONNX graph. For RN50 QAT, `tf.quantization.quantize_and_dequantize` operation (QDQ) is converted into `QuantizeLinear` and `DequantizeLinear` operations.
Support for converting QDQ operations has been added in `1.6.1` version of TF2ONNX.

Command to convert RN50 QAT TF graph to ONNX
```
python3 -m tf2onnx.convert --input <path_to_rn50_qat_graph> --output <output_file_name> --inputs input:0 --outputs resnet50/output/softmax_1:0 --opset 11
```

Arguments:
* `--input` : Name of TF input graph
* `--output` : Name of ONNX output graph
* `--inputs` : Name of input tensors
* `--outputs` : Name of output tensors
* `--opset` : ONNX opset version

### Step 5: Post processing ONNX

Run the following command to postprocess the ONNX graph using ONNX-Graphsurgeon API. This step removes the `transpose` nodes after `Dequantize` nodes. 
```
python postprocess_onnx.py --input <input_onnx_file> --output <output_onnx_file>
```

Arguments:
* `--input` : Input ONNX graph
* `--output` : Output name of postprocessed ONNX graph.

### Step 6: Build TensorRT engine from ONNX graph
```
python build_engine.py --onnx <input_onnx_graph>
```

Arguments:
* `--onnx` : Path to RN50 QAT onnx graph 
* `--engine` : Output file name of TensorRT engine.
* `--verbose` : Flag to enable verbose logging

Sample output log of this step is as follows. `ERROR: Tensor input:0 cannot be both input and output` can be ignored during TensorRT engine generation.
```
[TensorRT] ERROR: Tensor input:0 cannot be both input and output
[TensorRT] INFO: [EXPLICIT_PRECISION] Setting tensor scales of all tensors of explicit precision network to 1.0f
[TensorRT] WARNING: No implementation of layer InputQuantizeNode obeys the requested constraints in strict mode. No conforming implementation was found i.e. requested layer computation precision and output precision types are ignored, using the fastest implementation.
[TensorRT] INFO: Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
[TensorRT] INFO: Detected 1 inputs and 1 output network tensors.
```

### Step 7: TensorRT Inference

Command to run inference on a sample image

```
python infer.py --engine <input_trt_engine>
```

Arguments:
* `--engine` : Path to input RN50 TensorRT engine. 
* `--labels` : Path to imagenet 1k labels text file provided.
* `--image` : Path to the sample image
* `--verbose` : Flag to enable verbose logging

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
usage: <python <filename>.py> [-h]
```

# Additional resources

The following resources provide a deeper understanding about Quantization aware training, TF2ONNX and importing a model into TensorRT using Python:

**Quantization Aware Training**
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf)
- [Quantization Aware Training guide](https://www.tensorflow.org/model_optimization/guide/quantization/training)
- [Resnet-50 Deep Learning Example](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Classification/ConvNets/resnet50v1.5/README.md)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

**Parsers**
- [TF2ONNX Converter](https://github.com/onnx/tensorflow-onnx)
- [ONNX Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Importing A Model Using A Parser In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# Changelog

June 2020: Initial release of this sample

# Known issues

Tensorflow operation `tf.quantization.quantize_and_dequantize` is used for quantization during training. The gradient of this operation is not clipped based on input range.

# License

The sampleQAT license can be found in the LICENSE file.

