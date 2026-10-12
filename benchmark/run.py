import argparse
import asyncio
import statistics
import time

import httpx
import psutil

async def one(client: httpx.AsyncClient, url: str, index: int) -> tuple[float, bool]:
    payload = {
        "model": "facebook/opt-125m",
        "messages": [{"role": "user", "content": f"Reply with the number {index}."}],
        "max_tokens": 8,
        "temperature": 0,
    }
    started = time.perf_counter()
    try:
        response = await client.post(f"{url}/v1/chat/completions", json=payload)
        return time.perf_counter() - started, response.is_success
    except httpx.HTTPError:
        return time.perf_counter() - started, False

async def main(args: argparse.Namespace) -> None:
    process = psutil.Process()
    cpu_before = process.cpu_times()
    host_cpu_before = psutil.cpu_percent(interval=None)
    host_memory_before = psutil.virtual_memory().percent
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def limited(index: int) -> tuple[float, bool]:
            async with semaphore:
                return await one(client, args.url.rstrip("/"), index)

        results = await asyncio.gather(*(limited(i) for i in range(args.requests)))
    elapsed = time.perf_counter() - started
    latencies = sorted(latency for latency, _ in results)
    successful = sum(ok for _, ok in results)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    cpu_after = process.cpu_times()
    rss_mb = process.memory_info().rss / 1024**2
    host_cpu_after = psutil.cpu_percent(interval=None)
    host_memory_after = psutil.virtual_memory().percent
    print(f"requests={args.requests} success={successful} elapsed_s={elapsed:.3f}")
    print(f"throughput_rps={successful / elapsed:.3f} p95_latency_s={p95:.3f}")
    print(
        f"client_cpu_s={(cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system):.3f} "
        f"client_rss_mb={rss_mb:.1f} mean_latency_s={statistics.mean(latencies):.3f}"
    )
    print(
        f"host_cpu_percent={host_cpu_before:.1f}->{host_cpu_after:.1f} "
        f"host_memory_percent={host_memory_before:.1f}->{host_memory_after:.1f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    asyncio.run(main(parser.parse_args()))
