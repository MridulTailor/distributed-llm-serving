# Performance Report

## Test

- Platform: Docker Desktop Kubernetes on Apple Silicon
- Model: `facebook/opt-125m`
- Requests per test: 12
- Maximum output: 8 tokens
- State: warm model
- Resource metrics: unavailable; Metrics Server was not installed

## Results

| Concurrency | Success | Throughput (req/s) | p95 (s) |
| ---: | ---: | ---: | ---: |
| 1 | 12/12 | 1.499 | 1.017 |
| 2 | 12/12 | 3.029 | 0.800 |
| 4 | 12/12 | 4.767 | 0.892 |
| 8 | 12/12 | 6.024 | 1.137 |

Host CPU reached about 45%. Pod CPU and memory were not measured.

## Triton

Gateway test:

- Input: `3`
- Output: `6`
- Status: passed

## Metrics

- `gateway_requests_total`
- `gateway_request_duration_seconds`
- `gateway_backend_failures_total`
