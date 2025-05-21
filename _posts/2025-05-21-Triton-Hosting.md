---
layout: post
title: Hosting Custom Models on NVIDIA Triton Inference Server
category: inference
date: 2025-05-21
---

## Understanding NVIDIA's Inference Stack

NVIDIA has split its inference-serving stack into two closely related but distinct layers. **NVIDIA Dynamo** is the new *distributed* inference-orchestration framework that spans multiple GPU nodes, while **NVIDIA Dynamo-Triton** is simply the well-known Triton Inference Server re-branded as the execution engine within—and still usable outside—the Dynamo platform.

### NVIDIA Dynamo

* **Purpose** – Built for *data-center-scale* generative-AI and "reasoning" models, Dynamo coordinates many Triton (or other) workers across racks or clusters. It handles request routing, dynamic load balancing, elastic scaling, and model-specific parallelism strategies (tensor, pipeline, MoE, KV-cache sharding, etc.).
* **Architecture** – A modular control-plane plus pluggable workers; it disaggregates *prefill* and *decode* phases onto different GPU pools to maximize throughput and token-per-second revenue.
* **Engine-agnostic** – Workers can run TensorRT-LLM, vLLM, FasterTransformer, Python, or Triton backends, making Dynamo an umbrella scheduler rather than a model runtime itself.
* **Open Source & Brand-New** – Released March 2025 on GitHub under Apache-2.0 and positioned as the successor and "operating system for AI factories."

### NVIDIA Dynamo-Triton

* **Re-branding of Triton Inference Server** – Triton keeps its code base, APIs, and single-node focus, but is now marketed as **Dynamo-Triton** to signal that it is the default worker engine inside Dynamo.
* **Scope** – Executes models on one machine (multiple GPUs/CPUs) and already supports real-time, batch, streaming, ensembles, and 20+ backends (TensorFlow, PyTorch, ONNX, XGBoost, etc.).
* **Standalone or Integrated** – You can still run Triton by itself for edge or on-prem workloads; when deployed under Dynamo, it becomes a managed "worker" receiving requests from the Dynamo router.

### Key Differences at a Glance

| Aspect            | NVIDIA Dynamo                                                             | NVIDIA Dynamo-Triton                                                 |
| ----------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Role**          | Control-plane / scheduler across nodes                                    | Model execution engine on a single node                              |
| **Focus**         | Large-scale distributed LLM & multimodel pipelines                        | Per-node inference with many backends                                |
| **Architecture**  | Router + planner + worker pools; disaggregated prefill/decode             | Microservice exposing gRPC/HTTP, CUDA-pinned backend plugins         |
| **Scaling**       | Horizontal (thousands of GPUs) with dynamic resource allocation           | Vertical (multiple GPUs per server) via instances & dynamic batching |
| **Backends**      | Any engine that implements the worker API (TRT-LLM, vLLM, Triton, custom) | Built-in support for \~20 frameworks via backend plugins             |
| **Brand History** | New in 2025, "successor" platform to Triton                               | Formerly Triton Inference Server (2018-2024)                         |
| **Typical Use**   | AI factories, SaaS LLM services, multi-region clusters                    | Edge devices, single-server micro-services, on-prem deployments      |

### When to Use Which?

#### Choose **Dynamo-Triton Alone** if…

* You only need to serve models from one server or small GPU pod.
* Your workloads already fit Triton's feature set (dynamic batching, ensembles, GPU/CPU selection).
* You prefer maturity and minimal moving parts for on-prem or edge deployment.

#### Choose **Full Dynamo (with Dynamo-Triton Workers)** if…

* You must elastically scale an LLM or multimodel pipeline across many servers.
* Token throughput and GPU-to-GPU load balancing are your bottlenecks.
* You want a single control plane that can mix Triton, TensorRT-LLM, and specialized LLM runtimes under one scheduler.

### Roadmap and Support

NVIDIA states it "remains committed" to Triton; future releases will appear under the Dynamo-Triton label but keep API compatibility, while new distributed features (e.g., KV-cache migration, cross-node batching) land only in Dynamo. Enterprise support is provided via NVIDIA AI Enterprise, and both projects are Apache-2.0 on GitHub.

#### TL;DR

*Dynamo* = **data-center-scale orchestration layer** for generative-AI inference.
*Dynamo-Triton* = **the classic Triton Inference Server**, now one of the worker engines within the broader Dynamo platform, and still usable stand-alone.

## NVIDIA Triton Inference Server

Triton Inference Server enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more.

Triton Inference Server delivers optimized performance for many query types, including real time, batched, ensembles and audio/video streaming. Triton inference Server is part of NVIDIA AI Enterprise, a software platform that accelerates the data science pipeline and streamlines the development and deployment of production AI.

#### [Frameworks Overview]({{ site.baseurl }}/random/model-frameworks/)

### Triton Architecture

Triton Inference Server's architecture consists of several key components:

#### Model Repository
- File-system based storage for model artifacts
- Contains model configurations, weights, and metadata

#### Request Handling
- Multiple protocol support:
  - HTTP/REST API
  - gRPC API
  - C API
- Request routing to appropriate model schedulers

#### Model Schedulers
- Per-model scheduling algorithms
- Dynamic batching capabilities
- Configurable scheduling policies

#### Backend System
- Framework-specific inference engines
- Support for multiple backends:
  - TensorRT
  - PyTorch
  - TensorFlow
  - ONNX Runtime
  - Custom backends via C API

#### Model Management
- Dedicated management API
- Protocol support:
  - HTTP/REST
  - gRPC
  - C API

### Monitoring & Health
- Health endpoints:
  - Readiness probes
  - Liveness checks
- Performance metrics:
  - Throughput
  - Latency
  - Resource utilization

#### Extension System
- Backend C API for custom functionality
- Support for:
  - Custom preprocessing
  - Custom postprocessing
  - New framework integration

![Triton Architecture]({{ site.baseurl }}/assets/images/triton.png)

#### [Transport Protocols for Triton]({{ site.baseurl }}/random/transport-protocol/)


## Guide: Hosting a Transformer Model on Triton Inference Server

This guide provides a comprehensive approach for deploying transformer models on NVIDIA Triton Inference Server with custom processing capabilities using gRPC.

## Introduction to Triton Inference Server

NVIDIA Triton Inference Server is a highly optimized system for deploying machine learning models in production. Key advantages include:

- Support for multiple frameworks (TensorRT, PyTorch, ONNX, TensorFlow)
- Dynamic batching for improved throughput
- Concurrent model execution
- Model versioning and A/B testing capabilities
- gRPC and REST API interfaces
- Custom preprocessing and postprocessing via C++ or Python backends

## Architecture Overview

The proposed architecture consists of:

```
                           ┌─────────────────────────────────────┐
                           │        Triton Inference Server      │
                           │                                     │
Client ─── gRPC/HTTP ────► │ ┌─────────┐  ┌─────────┐  ┌──────┐ │
                           │ │ Input   │  │         │  │Output│ │
                           │ │ Process │─►│ Model   │─►│Process│ │
                           │ │ (Python)│  │(PyTorch)│  │(Python)│ │
                           │ └─────────┘  └─────────┘  └──────┘ │
                           │                                     │
                           └─────────────────────────────────────┘
```

1. **Client**: Sends data via gRPC
2. **Input Processing**: Converts raw data to model-ready tokens
3. **Model Execution**: Runs inference on transformer
4. **Output Processing**: Formats model outputs for client consumption

## Model Preparation

### Export PyTorch Model for Triton

```python
import torch
from transformers import AutoModelForCausalLM

def export_model():
    # Load the model
    model_path = "your_model_path"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Export the model to TorchScript
    example_input = torch.zeros((1, 256), dtype=torch.int64)  # Example batch of tokens
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the model
    torch.jit.save(traced_model, "model.pt")
```

Alternatively, you can use ONNX:

```python
import torch
from transformers import AutoModelForCausalLM

def export_to_onnx():
    # Load model
    model_path = "your_model_path"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    
    # Example input for tracing
    dummy_input = torch.zeros(1, 256, dtype=torch.int64)
    
    # Export to ONNX
    torch.onnx.export(
        model,                     # Model being exported
        dummy_input,               # Model input
        "model.onnx",              # Output file
        opset_version=13,          # ONNX opset version
        input_names=["input_ids"], # Input names
        output_names=["logits"],   # Output names
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )
```

## Custom Processing Implementation

Triton supports custom preprocessing and postprocessing through its Python backend. These components will handle converting raw data to model inputs and model outputs to client responses.

### Input Preprocessing

Create a file `1/model.py` for input processing:

```python
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Load tokenizer for special tokens
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("your_model_path")
    
    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get input tensors
            input_data = pb_utils.get_input_tensor_by_name(request, "input_data").as_numpy()[0]
            
            # Process input data to tokens
            tokens = self.process_input(input_data)
            
            # Create output tensor
            token_tensor = pb_utils.Tensor("input_ids", 
                                          np.array([tokens], dtype=np.int64))
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[token_tensor]
            )
            responses.append(inference_response)
            
        return responses
    
    def process_input(self, input_data):
        """Convert input data to model tokens."""
        # Implement your custom preprocessing logic here
        tokens = self.tokenizer.encode(input_data)
        return tokens
```

### Output Postprocessing

Create a file `3/model.py` for output processing:

```python
import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Load tokenizer if needed
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("your_model_path")
    
    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get input tensors
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            
            # Process logits
            output = self.process_output(logits)
            
            # Create response JSON
            response_data = {
                "status": "success",
                "result": output
            }
            
            # Create output tensor
            response_tensor = pb_utils.Tensor("response", 
                                            np.array([json.dumps(response_data)], dtype=np.object_))
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[response_tensor]
            )
            responses.append(inference_response)
            
        return responses
    
    def process_output(self, logits):
        """Process model output logits."""
        # Implement your custom postprocessing logic here
        predictions = np.argmax(logits, axis=-1)
        result = self.tokenizer.decode(predictions[0])
        return result
```

## Triton Model Configuration

You'll need to configure each component of your pipeline:

### Input Processor Configuration

Create `model_repository/input_processor/config.pbtxt`:

```protobuf
name: "input_processor"
backend: "python"
max_batch_size: 8

input [
  {
    name: "input_data"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

### Model Configuration

Create `model_repository/transformer_model/config.pbtxt`:

```protobuf
name: "transformer_model"
platform: "pytorch_libtorch"
max_batch_size: 8

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]  # [sequence_length, vocab_size]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### Output Processor Configuration

Create `model_repository/output_processor/config.pbtxt`:

```protobuf
name: "output_processor"
backend: "python"
max_batch_size: 8

input [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }
]

output [
  {
    name: "response"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

### Ensemble Configuration

Create `model_repository/transformer_ensemble/config.pbtxt`:

```protobuf
name: "transformer_ensemble"
platform: "ensemble"
max_batch_size: 8

input [
  {
    name: "input_data"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "response"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "input_processor"
      model_version: -1
      input_map {
        key: "input_data"
        value: "input_data"
      }
      output_map {
        key: "input_ids"
        value: "input_ids"
      }
    },
    {
      model_name: "transformer_model"
      model_version: -1
      input_map {
        key: "input_ids"
        value: "input_ids"
      }
      output_map {
        key: "logits"
        value: "logits"
      }
    },
    {
      model_name: "output_processor"
      model_version: -1
      input_map {
        key: "logits"
        value: "logits"
      }
      output_map {
        key: "response"
        value: "response"
      }
    }
  ]
}
```

## Client Implementation

### Python gRPC Client

```python
import grpc
import numpy as np
import json
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

class TritonClient:
    def __init__(self, url="localhost:8001"):
        self.url = url
        self.client = InferenceServerClient(url=url)
    
    def predict(self, input_text):
        """Send text data to Triton server for prediction."""
        
        # Prepare inputs
        input_data = InferInput(name="input_data", shape=[1], datatype="BYTES")
        input_data.set_data_from_numpy(np.array([input_text.encode('utf-8')], dtype=np.object_))
        
        # Set up inference
        outputs = [InferRequestedOutput("response")]
        
        # Run inference
        response = self.client.infer(
            model_name="transformer_ensemble",
            inputs=[input_data],
            outputs=outputs
        )
        
        # Process the results
        result = response.as_numpy("response")[0].decode('utf-8')
        result_json = json.loads(result)
        
        return result_json

# Example usage
if __name__ == "__main__":
    client = TritonClient()
    result = client.predict("Hello, how are you?")
    print(json.dumps(result, indent=2))
```

## Deployment and Scaling

### Docker Deployment

The easiest way to deploy Triton is using Docker:

```bash
# Pull Triton server image
docker pull nvcr.io/nvidia/tritonserver:22.12-py3

# Start Triton server with your model repository
docker run --gpus all -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v ${PWD}/model_repository:/models \
    nvcr.io/nvidia/tritonserver:22.12-py3 \
    tritonserver --model-repository=/models \
    --log-verbose=1 \
    --strict-model-config=false
```

## Performance Tuning

### Dynamic Batching

Triton's dynamic batching can significantly improve throughput. Adjust these settings:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 5000
}
```

### Model Optimization with TensorRT

For even better performance, consider converting your model to TensorRT:

```python
import tensorrt as trt
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("your_model_path")

# Export to ONNX first
dummy_input = torch.zeros(1, 256, dtype=torch.int64)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=13,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }
)

# Convert ONNX to TensorRT
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", 'rb') as model_file:
    parser.parse(model_file.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

# Build TensorRT engine
serialized_engine = builder.build_serialized_network(network, config)

with open("model.plan", "wb") as f:
    f.write(serialized_engine)
```

## Conclusion

This guide provides a comprehensive approach for deploying transformer models on NVIDIA Triton Inference Server with custom processing. By following these steps, you can achieve:

1. Higher throughput with dynamic batching
2. Lower latency with GPU optimization
3. Better scalability with containerized deployment
4. Efficient handling of custom inputs and outputs
5. Robust client-server communication via gRPC

This setup provides a production-ready system for hosting transformer models that can handle high-throughput, low-latency requirements.

[1]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/docs/benchmarking.html "Benchmarking Triton via HTTP or gRPC endpoint — NVIDIA Triton Inference Server"
[2]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html "Inference Protocols and APIs — NVIDIA Triton Inference Server"
[3]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_binary_data.html?utm_source=chatgpt.com "Binary Tensor Data Extension — NVIDIA Triton Inference Server"
[4]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html?utm_source=chatgpt.com "Shared-Memory Extension — NVIDIA Triton Inference Server"
[5]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inprocess_c_api.html?utm_source=chatgpt.com "C API Description — NVIDIA Triton Inference Server"
[6]: https://github.com/triton-inference-server/client?utm_source=chatgpt.com "Triton Client Libraries and Examples - GitHub"
[7]: https://github.com/triton-inference-server/server/issues/2306?utm_source=chatgpt.com "Why is there such a big performance difference between using http and grpc?"
[8]: https://github.com/triton-inference-server/server/issues/1821?utm_source=chatgpt.com "gRPC communication extremely slow · Issue #1821 · triton ... - GitHub"
[9]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/docs/benchmarking.html?utm_source=chatgpt.com "Benchmarking Triton via HTTP or gRPC endpoint"
[10]: https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md?utm_source=chatgpt.com "server/docs/protocol/extension_binary_data.md at main · triton ..."
[11]: https://www.njordy.com/2023/02/25/triton_shared_memory/?utm_source=chatgpt.com "Triton shared memory and pinned memory - Njord tech blog"
</rewritten_file>