# Distributed LLM Serving

Local Kubernetes services:

- vLLM CPU: `facebook/opt-125m`
- Triton CPU: ONNX echo model
- Gateway: health checks, retries, timeouts, and metrics
- Prometheus and Grafana

## Requirements

- Docker Desktop with Kubernetes enabled
- `kubectl`
- Python 3.10+
- `envsubst`

## Deploy

```bash
./scripts/build-onnx.sh
kubectl apply -f k8s/namespace.yaml
export VLLM_IMAGE=vllm/vllm-openai-cpu:latest
envsubst < k8s/vllm-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/triton-model.yaml
kubectl apply -f k8s/triton-deployment.yaml
kubectl apply -f k8s/gateway-deployment.yaml
kubectl apply -f k8s/monitoring.yaml
```

Check status:

```bash
kubectl -n llm-serving get pods
```

Local access:

```bash
kubectl -n llm-serving port-forward svc/gateway 8080:8080
kubectl -n llm-serving port-forward svc/grafana 3000:3000
```

Grafana: `http://127.0.0.1:3000`

## Benchmark

```bash
python -m pip install -r benchmark/requirements.txt
python benchmark/run.py --url http://127.0.0.1:8080 \
  --concurrency 4 --requests 100
```

Pod metrics require Metrics Server:

```bash
kubectl top pods -n llm-serving
```

## Endpoints

| Path | Use |
| --- | --- |
| `/health/live` | Liveness |
| `/health/ready` | Dependency readiness |
| `/v1/chat/completions` | vLLM |
| `/triton/infer` | Triton |
| `/metrics` | Prometheus |

## ONNX

```bash
python -m pip install onnx
python scripts/export_onnx.py
```

The model returns `input * 2`.
