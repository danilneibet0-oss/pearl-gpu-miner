"""
stratum_client.py — клиент пулового Stratum-диалекта Pearl по ПУБЛИЧНОЙ спецификации Suprnova:
https://prl.suprnova.cc/stratum-spec.html  (в спецификации сказано, что диалект общий для экосистемы Pearl,
но независимых подтверждений нет — для конкретного пула это нужно проверять).

Кратко (по спецификации):
  * TCP (или TLS на отдельном порту), JSON по одной записи на строку, строка до 4 МБ;
  * params — всегда ОБЪЕКТ (массив даёт ошибку 20); mining.subscribe в этом диалекте НЕТ и не отправляется;
  * клиент -> сервер: mining.authorize {wallet, worker, pass, agent}, mining.submit {job_id, plain_proof};
  * сервер -> клиент: mining.notify {header(76 байт hex), height, job_id, target(U256 hex)} — может прийти ДО ack;
  * ответ: {"error": null|{"code","msg"}, "id", "result"}; коды 20..27, неизвестные поля игнорируются.
"""

from __future__ import annotations

import base64
import gzip
import json
import socket
import ssl
import threading
from dataclasses import dataclass

import pearlhash_ref as ref

MAX_LINE = 4 * 1024 * 1024

ERRORS = {20: "method not supported / params not an object", 21: "stale job", 22: "duplicate share",
          23: "low-difficulty share", 24: "wallet missing", 25: "invalid wallet / bad proof encoding",
          26: "invalid proof", 27: "unauthorized"}


class StratumError(Exception):
    def __init__(self, code: int, msg: str = ""):
        super().__init__(f"[{code}] {msg or ERRORS.get(code, '')}")
        self.code, self.msg = code, msg


@dataclass(frozen=True)
class StratumJob:
    job_id: str
    header_bytes: bytes            # 76 байт без ProofCommitment
    height: int
    target: int                    # U256 из hex-строки как обычное целое
    cert_version: int | None = None   # в спецификации Suprnova нет, но Kryptex присылает его в notify

    def header(self) -> ref.IncompleteBlockHeader:
        return ref.IncompleteBlockHeader.from_bytes(self.header_bytes)


def parse_job(params: dict) -> StratumJob:
    header = bytes.fromhex(params["header"])
    if len(header) != 76:
        raise ValueError(f"header: ожидалось 76 байт, получено {len(header)}")
    cv = params.get("cert_version")
    return StratumJob(str(params["job_id"]), header, int(params["height"]), int(params["target"], 16),
                      int(cv) if cv is not None else None)


def parse_login(login: str) -> tuple[str, str]:
    """'wallet/worker', 'wallet.worker', 'solo:wallet.worker' или 'wallet' -> (wallet, worker)."""
    login = login.strip()
    for sep in ("/", "."):
        if sep in login:
            wallet, worker = login.split(sep, 1)
            return wallet, worker or "default"
    return login, "default"


class StratumClient:
    def __init__(self, host: str, port: int, tls: bool = False, timeout: float = 15.0):
        self.host, self.port, self.tls, self.timeout = host, port, tls, timeout
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._cond = threading.Condition()
        self._pending: dict[int, dict | None] = {}
        self._next_id = 1
        self._job: StratumJob | None = None
        self._closed = True
        self._reader: threading.Thread | None = None
        self.gzip = False            # пул подтвердил протокол v2: plain_proof отправляется в gzip

    # --- соединение ---
    def connect(self) -> "StratumClient":
        self.close()
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.tls:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
        sock.settimeout(None)
        self._sock, self._closed = sock, False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        return self

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __enter__(self):
        return self if not self._closed else self.connect()

    def __exit__(self, *exc):
        self.close()

    # --- приём ---
    def _read_loop(self) -> None:
        sock, buf = self._sock, b""
        try:
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._on_line(line)
                if len(buf) > MAX_LINE:
                    break
        except OSError:
            pass
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def _on_line(self, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            return                                   # мусор игнорируем
        if not isinstance(msg, dict):
            return
        with self._cond:
            if msg.get("method") == "mining.notify" and isinstance(msg.get("params"), dict):
                try:
                    self._job = parse_job(msg["params"])
                except (KeyError, ValueError, TypeError):
                    return
            elif msg.get("id") in self._pending:
                self._pending[msg["id"]] = msg
            # прочие методы и поля — игнорируются (спецификация: клиент обязан игнорировать неизвестное)
            self._cond.notify_all()

    # --- вызовы ---
    def call(self, method: str, params: dict, full: bool = False):
        with self._cond:
            if self._closed:
                raise ConnectionError("нет соединения с пулом")
            rid = self._next_id
            self._next_id += 1
            self._pending[rid] = None
        line = (json.dumps({"id": rid, "method": method, "params": params}) + "\n").encode()
        try:
            with self._send_lock:
                self._sock.sendall(line)
        except OSError as e:
            self.close()
            raise ConnectionError(str(e)) from e
        with self._cond:
            ok = self._cond.wait_for(lambda: self._pending[rid] is not None or self._closed, self.timeout)
            resp = self._pending.pop(rid)
        if resp is None:
            raise ConnectionError("соединение закрыто") if self._closed else TimeoutError(f"{method}: нет ответа")
        err = resp.get("error")
        if err:
            raise StratumError(int(err.get("code", 0)), str(err.get("msg", "")))
        return resp if full else resp.get("result")

    def authorize(self, login: str, password: str | None = None, agent: str = "pearl-gpu-miner/0.1",
                  v2: bool = True) -> bool:
        """
        v2=True: в параметрах "type": "v2" — протокол Kryptex со сжатием; если пул вернул "type": "v2", то plain_proof
        отправляется в gzip (self.gzip). Имя воркера передаётся склейкой wallet.worker (так его берёт Kryptex);
        явное поле worker остаётся для диалекта Suprnova.
        """
        wallet, worker = parse_login(login)
        glued = wallet if worker == "default" and "/" not in login and "." not in login else f"{wallet}.{worker}"
        params = {"wallet": glued, "worker": worker, "agent": agent}
        if password:
            params["pass"] = password
        if v2:
            params["type"] = "v2"
        resp = self.call("mining.authorize", params, full=True)
        result = resp.get("result")
        confirmed = resp.get("type") == "v2" or (isinstance(result, dict) and result.get("type") == "v2")
        self.gzip = bool(v2 and confirmed)
        return bool(result)

    def submit(self, job_id: str, plain_proof_b64: str) -> bool:
        """plain_proof_b64 — base64 от бинарного PlainProof; при self.gzip уходит base64(gzip(байты))."""
        if self.gzip:
            plain_proof_b64 = base64.b64encode(gzip.compress(base64.b64decode(plain_proof_b64), 6)).decode()
        return bool(self.call("mining.submit", {"job_id": job_id, "plain_proof": plain_proof_b64}))

    # --- задания ---
    def current_job(self) -> StratumJob | None:
        with self._cond:
            return self._job

    def wait_job(self, timeout: float, newer_than: str | None = None) -> StratumJob:
        """Ждёт задание (если newer_than задан — с другим job_id)."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: (self._job is not None and self._job.job_id != newer_than) or self._closed, timeout)
            if self._job is None or self._job.job_id == newer_than:
                raise ConnectionError("соединение закрыто") if self._closed else TimeoutError("нет задания от пула")
            return self._job
