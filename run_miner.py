"""
run_miner.py — цикл майнинга: задание от gateway -> GPU-попытки -> PlainProof -> проверка -> submitPlainProof.

  .venv\\Scripts\\python run_miner.py --port 8337 --m 4096 --n 4096 --k 2048

Требует запущенный pearl-gateway на localhost. Параметры m, n, k, rank должны удовлетворять правилам протокола
(k >= 16*rank, k % 64 == 0, rank >= 128, периоды паттернов делят m и n).
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import pearl_pipeline as pp
import pearl_proof as pr
import pearlhash_ref as ref
from gateway_client import GatewayClient, MiningPaused, job_bound


def run(client: GatewayClient, m: int, n: int, k: int, rank: int = 128, *, max_solutions: int | None = None,
        poll_s: float = 1.0, stop: threading.Event | None = None, log=print,
        batch: int = 16, depth: int = 3) -> dict:
    stop = stop or threading.Event()
    cfg = ref.default_config(k, rank)
    stats = {"attempts": 0, "found": 0, "submitted": 0, "rejected_locally": 0}
    job, last_poll, t0 = None, 0.0, time.time()
    while not stop.is_set() and (max_solutions is None or stats["submitted"] < max_solutions):
        if job is None or time.time() - last_poll > poll_s:
            try:
                new = client.get_mining_info()
            except MiningPaused as e:
                log(f"майнинг на паузе: {e}")
                job = None
                stop.wait(1.0)
                continue
            if new != job:
                log(f"новое задание: cert_version={new.cert_version}")
            job, last_poll = new, time.time()
        bound = job_bound(job, cfg)
        if bound is None:
            raise RuntimeError("цель задания неприменима к этой конфигурации (штраф ранга не влезает в 256 бит)")
        header = job.header()
        base = int.from_bytes(os.urandom(6), "little")
        for _seed, sol in pp.mine_random_pipelined(m, n, k, header, cfg, job.cert_version,
                                                   range(base, base + batch), bound=bound, depth=depth):
            stats["attempts"] += 1
            if sol is None:
                continue
            stats["found"] += 1
            proof = pr.build_plain_proof(sol, header, cfg)
            ok, msg = pr.verify_before_submit(job.cert_version, header, proof)
            if not ok:
                stats["rejected_locally"] += 1
                log(f"решение не прошло локальную проверку, не отправляю: {msg}")
                continue
            res = client.submit_plain_proof(pr.proof_to_base64(proof), job)
            stats["submitted"] += 1
            log(f"решение отправлено ({res}), попыток: {stats['attempts']}, {time.time() - t0:.1f} c")
            if max_solutions is not None and stats["submitted"] >= max_solutions:
                break
            if stop.is_set():
                break
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Pearl GPU miner -> pearl-gateway (localhost)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8337)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=128)
    args = ap.parse_args()
    with GatewayClient(args.host, args.port) as client:
        try:
            print(run(client, args.m, args.n, args.k, args.rank))
        except KeyboardInterrupt:
            print("остановлено")


if __name__ == "__main__":
    main()
