---
layout: page
title: "Transport Protocol Comparison"
permalink: /random/transport-protocol/
---

**Fastest path in practice:**
For raw-audio (or any large binary) payloads, the lowest end-to-end latency comes from **embedding Triton with the in-process C API and passing the audio in shared memory**. When you must go over the network, **gRPC slightly out-runs HTTP/REST** because it is a binary protocol, supports bidirectional streaming, and avoids the JSON/base-64 tax; using HTTP with the *binary tensor data* extension narrows—but rarely erases—that gap. The table below details the trade-offs and why most production deployments pick **gRPC (+ shared-memory)** for audio pipelines, reserving the C API for same-host inference and HTTP for browser or firewall-restricted clients.

### Protocol Options at a Glance

| Use case                                      | API choice                                                           | Why it is (or isn't) fastest                                                                                                                                                                               |
| --------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Same process / same host**                  | **C API**                                                            | No network stack; your app links `libtritonserver.so`, so only a memcpy separates you from the backend. NVIDIA's docs state HTTP/gRPC "introduce additional latency… C API avoids that" ([NVIDIA Docs][1]) |
| **Remote, latency-sensitive audio streaming** | **gRPC**                                                             | Binary Protobuf frames, persistent HTTP/2 connection, optional bidirectional *streaming* RPC that keeps every chunk on the same server ([NVIDIA Docs][2])                                                  |
| Remote, firewall-friendly / browsers          | HTTP/REST                                                            | Works everywhere; use *binary tensor* extension to skip base-64 ([NVIDIA Docs][3])                                                                                                                         |
| Any local client that can mmap                | **Shared-memory extension** (system / CUDA) with either gRPC or HTTP | Payload lives in a POSIX shm segment; the RPC transmits only a handle—huge win for multi-MB audio blocks ([NVIDIA Docs][4])                                                                                |

### How the Bytes Travel

#### gRPC

* One Protobuf message per request; audio enters a `BYTES` tensor—already raw, no intermediate encoding.
* The *streaming* variant keeps a single socket open so your 50 × 10 ms chunks land on the same GPU-worker with no re-handshake cost ([NVIDIA Docs][2]).

#### HTTP/REST

* Default payload is JSON + base-64, which inflates size ≈ 33 %.
* Activate **`binary_tensor_data`**; Triton will accept raw little-endian tensor bytes appended to the JSON header ([NVIDIA Docs][3]). This cuts transfer time but you still pay HTTP header parsing and one TCP handshake per connection (unless you reuse HTTP/1.1 keep-alive or HTTP/2 multiplexing).

#### In-process C API

* Your code calls `TRITONSERVER_InferAsync()` on a handle returned by `TRITONSERVER_ServerNew()` ([NVIDIA Docs][5]).
* No sockets, no serialization; the only copies are whatever you perform before `TRITONSERVER_InferenceRequestSetSharedMemory()`.

#### Shared Memory Extras

* Register a region once (`RegisterSystemSharedMemory`) then send its name to Triton for every inference ([GitHub][6]).
* Works with HTTP or gRPC, but is especially useful when client and server share the same node or GPU (CUDA-shm).

### Quantitative Evidence

* **C API avoids transport cost:** NVIDIA's perf-analyzer example shows the *same* ResNet-50 model yielding **only 20 µs transport overhead** in C-API mode, while HTTP/gRPC add dozens of microseconds ([NVIDIA Docs][1]).
* **HTTP CPU overhead at scale:** Users report high `sy` CPU and latency growth when concurrency rises under HTTP, not under gRPC ([GitHub][7]).
* **gRPC improvements closed the gap:** By mid-2024 a PR eliminated earlier gRPC slow-start issues; maintainers now see "no noticeable difference between gRPC and HTTP on the server" for most models, leaving serialization cost as the main delta ([GitHub][8]).
* **Docs still flag extra latency:** The official benchmarking guide reiterates that both remote protocols add latency the C API doesn't have ([NVIDIA Docs][9]).
* **Streaming advantages:** Because gRPC streams keep connection affinity, they're the recommended path where chunk order must be preserved (common in VAD or incremental ASR) ([NVIDIA Docs][2]).

### Putting It Together for Audio Pipelines

1. **Same-host micro-service?**
   *Embed Triton via C API and place your PCM/Opus bytes in system/CUDA shared memory*—fastest possible (tens of µs overhead).

2. **Micro-service across machines or Kubernetes pods?**
   *Use gRPC.* Keep a persistent channel and send either:

   * a single request containing the whole clip as a `BYTES` tensor, or
   * a streaming RPC where every 10–20 ms chunk is an individual message (ideal for real-time).

3. **Browser or firewalled corporate environment?**
   *Use HTTP/REST with `application/vnd.triton.binary+json`* so your WAV/FLAC bytes stay raw; overhead is \~5-15 % higher than gRPC but universally routable ([GitHub][10]).

4. **Large on-prem batch jobs?**
   Combine gRPC with the **system shared-memory** extension so workers push pointers, not payloads—often doubling throughput on 10 Gb E links ([njordy.com][11]).

### Recommendation

* **Fastest overall:** **C API + shared memory** (requires embedding Triton; same host only).
* **Fastest over the wire:** **gRPC** (binary, streaming, lower CPU).
* **Acceptable fallback:** HTTP/REST with *binary tensor*; add keep-alive and gzip if the network is slow.
  In all cases, register a shared-memory region or enable gRPC streaming when you control both ends—especially important when shipping high-rate audio bytes to your model.

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
