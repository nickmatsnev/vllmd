import argparse
import asyncio
import socket
import subprocess
import uuid

import httpx
from fastapi import FastAPI


app = FastAPI()

worker_id = str(uuid.uuid4())

COORDINATOR = None
VLLM_PORT = 8001


@app.get("/health")
async def health():
    return {
        "id": worker_id,
        "status": "ok",
    }


async def register():
    hostname = socket.gethostname()

    payload = {
        "id": worker_id,
        "host": hostname,
        "port": VLLM_PORT,
        "gpu_count": 1,
        "gpu_memory_gb": 24,
    }

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{COORDINATOR}/workers/register",
            json=payload,
        )


async def heartbeat_loop():
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{COORDINATOR}/workers/{worker_id}/heartbeat"
                )
        except Exception:
            pass

        await asyncio.sleep(5)


def start_vllm(model: str):
    return subprocess.Popen([
        "vllm",
        "serve",
        model,
        "--port",
        str(VLLM_PORT),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    COORDINATOR = args.coordinator

    start_vllm(args.model)

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
    )
