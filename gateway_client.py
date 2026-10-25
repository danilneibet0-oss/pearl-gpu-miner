"""
gateway_client.py — клиент к pearl-gateway (miner <-> gateway).

Протокол (miner/pearl-gateway/src/pearl_gateway/miner_rpc): JSON-RPC 2.0, одна запись на строку, TCP.
  getMiningInfo   params={}  -> {"incomplete_header_bytes": b64, "target": int, "cert_version": int}
  submitPlainProof params={"plain_proof": b64, "mining_job": <то же задание>} -> "submitted"
  ошибка «майнинг на паузе»: code -32001.

cert_version всегда берётся из задания gateway, не хардкодится.
Gateway слушает только 127.0.0.1; клиент по умолчанию тоже отказывается подключаться к нерфейсным адресам.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import socket
import threading
from dataclasses import dataclass

import pearlhash_ref as ref

DEFAULT_PORT = 8337
READ_LIMIT = 2**20                  # как READER_BUFFER_LIMIT на стороне gateway
MINING_PAUSED_CODE = -32001
PENALTY_BASE_RANK = 128


class GatewayError(Exception):
    def __init__(self, code: int, message: str, data=None):
        super().__init__(f"[{code}] {message}" + (f" ({data})" if data else ""))
        self.code, self.message, self.data = code, message, data


class MiningPaused(GatewayError):
    pass


@dataclass(frozen=True)
class MiningJob:
    incomplete_header_bytes: bytes
    target: int
    cert_version: int

    @classmethod
    def from_dict(cls, d: dict) -> "MiningJob":
        return cls(base64.b64decode(d["incomplete_header_bytes"], validate=True),
                   int(d["target"]), int(d["cert_version"]))

    def to_dict(self) -> dict:
        return {"incomplete_header_bytes": base64.b64encode(self.incomplete_header_bytes).decode(),
                "target": self.target, "cert_version": self.cert_version}

    def header(self) -> ref.IncompleteBlockHeader:
        return ref.IncompleteBlockHeader.from_bytes(self.incomplete_header_bytes)


def job_bound(job: MiningJob, cfg: ref.MiningConfiguration) -> int | None:
    """
    Граница для jackpot_hash с штрафом за ранг — как MiningJob.adjust_target / penalized_target_bound:
    target * tile_size * (dot_product_length // rank) * 128. None, если не влезает в 256 бит.
    """
    if cfg.rank < PENALTY_BASE_RANK:
        raise ValueError(f"rank {cfg.rank} < {PENALTY_BASE_RANK}: блок будет отклонён консенсусом")
    factor = cfg.tile_size() * (cfg.dot_product_length() // cfg.rank) * PENALTY_BASE_RANK
    if factor == 0 or job.target > ref.U256_MAX // factor:
        return None
    return job.target * factor


class GatewayClient:
    """Блокирующий клиент; один запрос за раз (защищён локом)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 10.0,
                 allow_remote: bool = False):
        if not allow_remote and not _is_loopback(host):
            raise ValueError(f"gateway слушает только localhost; {host!r} запрещён (allow_remote=True, если уверены)")
        self.host, self.port, self.timeout = host, port, timeout
        self._sock: socket.socket | None = None
        self._buf = b""
        self._next_id = 1
        self._lock = threading.Lock()

    # --- соединение ---
    def connect(self) -> "GatewayClient":
        self.close()
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._buf = b""
        return self

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # --- JSON-RPC ---
    def _readline(self) -> bytes:
        while b"\n" not in self._buf:
            if len(self._buf) > READ_LIMIT:
                raise GatewayError(-32000, "ответ gateway слишком длинный")
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("gateway закрыл соединение")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def call(self, method: str, params: dict | None = None):
        with self._lock:
            if self._sock is None:
                self.connect()
            rid = self._next_id
            self._next_id += 1
            req = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": rid}
            try:
                self._sock.sendall((json.dumps(req) + "\n").encode())
                resp = json.loads(self._readline())
            except (OSError, ConnectionError, ValueError):
                self.close()                      # соединение в неизвестном состоянии — переподключимся в следующий раз
                raise
        if resp.get("id") != rid and resp.get("id") is not None:
            raise GatewayError(-32000, f"id ответа {resp.get('id')} != {rid}")
        if "error" in resp:
            e = resp["error"]
            cls = MiningPaused if e.get("code") == MINING_PAUSED_CODE else GatewayError
            raise cls(e.get("code", -32000), e.get("message", ""), e.get("data"))
        return resp.get("result")

    # --- методы gateway ---
    def get_mining_info(self) -> MiningJob:
        return MiningJob.from_dict(self.call("getMiningInfo"))

    def submit_plain_proof(self, plain_proof_b64: str, job: MiningJob) -> str:
        """plain_proof_b64 = PlainProof.to_base64(); job — задание, по которому майнили."""
        return self.call("submitPlainProof", {"plain_proof": plain_proof_b64, "mining_job": job.to_dict()})


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
