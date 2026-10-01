# vllmd

**Distributed vLLM without the distributed-systems ceremony.**

Turn a bunch of GPU machines into a single OpenAI-compatible inference endpoint.

```bash
pip install vllmd

# machine 1
vllmd head

# machine 2
vllmd join 192.168.1.10

# machine 3
vllmd join 192.168.1.10

# deploy
vllmd run Qwen/Qwen3-32B
```

That's it.

No Kubernetes required.  
No CRDs.  
No Helm charts.  
No twenty-seven YAML files describing one GPU server.

Just vLLM, distributed.

---

## Why?

[vLLM](https://github.com/vllm-project/vllm) is extremely good at running LLM inference.

But once you have more than one machine, things get annoying surprisingly fast.

You suddenly need to think about:

- worker discovery
- health checks
- model placement
- request routing
- GPU availability
- replicas
- failures
- load balancing
- process supervision
- deployment state
- metrics
- cluster configuration

There are excellent systems for large Kubernetes deployments.

`vllmd` is for everything before that.

Think:

```text
Ollama     → easy local inference
vLLM       → fast production inference
vllmd      → easy distributed vLLM
llm-d      → large-scale Kubernetes inference platform
```

`vllmd` aims to make a pile of GPU servers feel like one machine.

---

# What does it do?

```text
                         ┌─────────────────┐
                         │     Client      │
                         │ OpenAI SDK/curl │
                         └────────┬────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │        vllmd          │
                      │                       │
                      │  discovery            │
                      │  routing              │
                      │  scheduling           │
                      │  health               │
                      │  model placement      │
                      └───────────┬───────────┘
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
               ▼                  ▼                  ▼
        ┌────────────┐     ┌────────────┐     ┌────────────┐
        │   node-a   │     │   node-b   │     │   node-c   │
        │            │     │            │     │            │
        │   vLLM     │     │   vLLM     │     │   vLLM     │
        │ RTX 4090×2 │     │ RTX 5090×1 │     │ A6000×2    │
        └────────────┘     └────────────┘     └────────────┘
```

Every worker runs stock vLLM.

`vllmd` does not replace vLLM.

It coordinates it.

---

# Goals

The project has one very simple goal:

> **Make distributed vLLM boring.**

Running inference across multiple machines should feel approximately like running Docker Compose.

A cluster should be something you can create in minutes:

```bash
vllmd head
```

```text
vllmd coordinator started

Cluster address:
192.168.1.20:8000

Join command:

    vllmd join 192.168.1.20
```

Then on other GPU machines:

```bash
vllmd join 192.168.1.20
```

And finally:

```bash
vllmd run meta-llama/Llama-3.3-70B
```

Example output:

```text
vllmd cluster

NODE      GPU                VRAM     STATUS
node-a    RTX 4090 ×2        48 GB    ready
node-b    RTX 4090 ×2        48 GB    ready
node-c    RTX 3090 ×2        48 GB    ready

Total GPUs: 6
Total VRAM: 144 GB

Deploying:
meta-llama/Llama-3.3-70B

Plan:
3 replicas
tensor-parallel-size: 2

Endpoint:
http://192.168.1.20:8000/v1
```

---

# Installation

Coordinator only:

```bash
pip install vllmd
```

GPU worker:

```bash
pip install "vllmd[worker]"
```

Development install:

```bash
git clone https://github.com/YOURNAME/vllmd
cd vllmd

pip install -e .
```

Worker development install:

```bash
pip install -e ".[worker]"
```

## Requirements

Coordinator:

- Python 3.11+
- Linux or macOS for development
- no GPU required

Workers:

- Linux
- NVIDIA GPU
- compatible NVIDIA driver
- CUDA-supported vLLM environment
- vLLM

`vllmd` intentionally keeps vLLM as an optional dependency so the coordinator does not need to install the entire GPU inference stack.

---

# Quick start

## 1. Start the coordinator

```bash
vllmd head
```

Output:

```text
vllmd head running

API:
http://0.0.0.0:8000

Workers:
0

Waiting for nodes...
```

---

## 2. Join GPU workers

On another machine:

```bash
vllmd join 192.168.1.20
```

The worker automatically reports:

```text
hostname
GPU model
GPU count
VRAM
CUDA version
vLLM version
available memory
```

Example:

```text
Connected to vllmd cluster.

Node:
gpu-box-02

GPUs:
2 × NVIDIA RTX 4090

VRAM:
48 GB

Status:
ready
```

---

## 3. Deploy a model

```bash
vllmd run Qwen/Qwen3-32B
```

`vllmd` selects an appropriate deployment strategy and starts vLLM workers.

```text
Analyzing cluster...

Available:
4 GPUs
96 GB VRAM

Model:
Qwen/Qwen3-32B

Deployment plan:

replicas: 2
tensor parallel: 2

Starting workers...

✓ node-a
✓ node-b
✓ node-c
✓ node-d

Model ready.

Endpoint:
http://192.168.1.20:8000/v1
```

---

# OpenAI-compatible API

Use the normal OpenAI client.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.20:8000/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[
        {
            "role": "user",
            "content": "Explain distributed inference like I'm five."
        }
    ],
)

print(response.choices[0].message.content)
```

Or curl:

```bash
curl http://192.168.1.20:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [
      {
        "role": "user",
        "content": "Hello."
      }
    ]
  }'
```

---

# Architecture

`vllmd` consists of two lightweight components.

## Coordinator

The coordinator owns cluster state.

It handles:

```text
worker registry
request routing
health checking
model placement
deployment lifecycle
load balancing
metrics
```

## Worker agent

Each GPU node runs a tiny worker agent.

The worker:

```text
detects GPUs
registers with coordinator
reports health
starts/stops vLLM
reports queue/load information
supervises the vLLM process
```

Underneath:

```text
vllmd worker
      │
      ├── GPU discovery
      ├── heartbeat
      ├── process supervisor
      │
      └── vLLM
           │
           └── CUDA / GPU
```

---

# Design philosophy

## Don't fork vLLM

`vllmd` uses stock vLLM.

Inference belongs in vLLM.

Cluster orchestration belongs in vllmd.

```text
vLLM
├── CUDA execution
├── attention
├── batching
├── KV cache
├── tensor parallelism
└── inference

vllmd
├── discovery
├── scheduling
├── health
├── deployment
├── routing
└── cluster lifecycle
```

If something should exist in vLLM itself, it should be contributed upstream.

---

## No Kubernetes requirement

Kubernetes support may exist eventually.

Kubernetes dependency will not.

A three-machine GPU lab should not require becoming a platform engineering department.

---

## Sensible defaults

You should not need to understand every vLLM flag to run a model.

This:

```bash
vllmd run Qwen/Qwen3-32B
```

should automatically determine reasonable values for:

```text
tensor parallel size
replica count
GPU placement
memory utilization
worker selection
```

Advanced users can override everything.

```bash
vllmd run Qwen/Qwen3-32B \
  --replicas 4 \
  --tensor-parallel 2 \
  --gpu-memory-utilization 0.92
```

---

# Scheduler

The initial scheduler is intentionally boring.

```python
worker = min(
    healthy_workers,
    key=lambda worker: worker.active_requests,
)
```

Future versions can account for:

```text
queue depth
GPU utilization
VRAM pressure
estimated latency
prefix cache locality
model affinity
network topology
KV cache locality
```

Schedulers should eventually be pluggable.

```python
class Scheduler:
    async def select_worker(self, request, workers):
        ...
```

---

# Model placement

Eventually:

```bash
vllmd run Qwen/Qwen3-32B
```

can produce something like:

```text
Cluster:

node-a
RTX 4090 ×2
48 GB VRAM

node-b
RTX 4090 ×2
48 GB VRAM

node-c
RTX 3090 ×2
48 GB VRAM


Recommended deployment:

Replica 1
node-a
tensor-parallel=2

Replica 2
node-b
tensor-parallel=2

Replica 3
node-c
tensor-parallel=2
```

The important bit:

**vllmd decides placement. vLLM runs inference.**

---

# Failure handling

Workers send periodic heartbeats.

```json
{
  "status": "healthy",
  "gpu_utilization": 61,
  "queue_depth": 4,
  "available_vram": 18.2
}
```

If a node disappears:

```text
node-b heartbeat timeout
        │
        ▼
mark unhealthy
        │
        ▼
stop routing requests
        │
        ▼
recalculate deployment
```

Eventually vllmd may automatically recreate lost replicas elsewhere.

---

# Planned CLI

```bash
vllmd head
```

Start a coordinator.

```bash
vllmd join HOST
```

Join a cluster.

```bash
vllmd run MODEL
```

Deploy a model.

```bash
vllmd ps
```

Show running models.

```bash
vllmd nodes
```

Show cluster nodes.

```bash
vllmd stop MODEL
```

Stop a model.

```bash
vllmd logs MODEL
```

Stream worker logs.

```bash
vllmd inspect MODEL
```

Show deployment topology.

---

# Project structure

```text
vllmd/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── vllmd/
    ├── __init__.py
    ├── main.py
    ├── worker.py
    ├── scheduler.py
    ├── registry.py
    └── models.py
```

For the MVP, the codebase is intentionally tiny.

The goal is to validate the control-plane model before introducing unnecessary infrastructure.

---

# Development

Install local dependencies:

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

Run coordinator:

```bash
python -m vllmd.main
```

Run a worker:

```bash
python -m vllmd.worker \
  --coordinator http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B
```

For early local testing, multiple workers can be simulated on a single machine using different ports.

---

# MVP scope

The first version deliberately does **not** attempt to implement distributed model execution itself.

That means no custom:

```text
NCCL orchestration
RDMA layer
tensor transport
cross-node KV cache
CUDA runtime
distributed executor
```

vLLM already solves the hard inference problems.

The initial `vllmd` layer focuses on:

```text
worker discovery
worker lifecycle
health checks
request routing
GPU inventory
model deployment
simple scheduling
unified OpenAI endpoint
```

This keeps the first implementation small, understandable, and actually shippable.

---

# Roadmap

## v0.1

- [ ] coordinator
- [ ] worker registration
- [ ] worker heartbeat
- [ ] GPU discovery
- [ ] vLLM process supervision
- [ ] OpenAI API proxy
- [ ] basic least-load routing
- [ ] multi-replica inference
- [ ] CLI

## v0.2

- [ ] automatic model placement
- [ ] GPU-aware scheduler
- [ ] worker recovery
- [ ] metrics
- [ ] model lifecycle management
- [ ] authentication
- [ ] streaming improvements

## v0.3

- [ ] prefix-aware routing
- [ ] topology-aware scheduling
- [ ] heterogeneous GPU placement
- [ ] distributed KV-cache awareness
- [ ] prefill/decode-aware routing

## Eventually

- [ ] autoscaling
- [ ] multi-model clusters
- [ ] cluster federation
- [ ] Kubernetes integration
- [ ] Ray integration
- [ ] RDMA-aware topology
- [ ] prefill/decode disaggregation
- [ ] web dashboard

---

# What vllmd is not

It is not another inference engine.

It is not a vLLM fork.

It is not trying to replace Kubernetes.

It is not trying to replace llm-d.

It is not trying to invent a new model API.

It is glue.

Hopefully very good glue.

---

# Status

Very early.

Expect:

```text
bugs
bad ideas
API changes
questionable decisions
GPUs doing GPU things
```

The architecture is intentionally small while the basic model is validated.

---

# Contributing

PRs are welcome.

Especially around:

- scheduling
- GPU discovery
- vLLM integration
- networking
- observability
- failure recovery
- heterogeneous GPU clusters

If you're running a weird collection of GPUs in someone's basement, garage, office, lab, datacenter or suspiciously powerful bedroom PC, that is exactly the kind of setup this project wants to support.

---

# Principles

```text
simple > clever

stock vLLM > forked vLLM

automatic > manual

one command > twelve YAML files

boring infrastructure > exciting outages
```

---

# License

Apache-2.0

---

<p align="center">

**vllmd**

*one endpoint, however many GPUs you found lying around*

</p>
