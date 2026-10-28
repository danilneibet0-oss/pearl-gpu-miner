"""
Тесты stratum-клиента Pearl на локальном фейковом сервере (127.0.0.1, без внешней сети).
Фейковый сервер реализует диалект из публичной спецификации Suprnova (https://prl.suprnova.cc/stratum-spec.html):
JSON по строкам, params — объект, методы mining.authorize / mining.submit, push mining.notify (до ack),
коды ошибок 20–27, plain_proof = base64 (raw / gzip / zlib), реальная проверка официальным верификатором.
"""

import asyncio
import base64
import gzip
import json
import threading
import zlib

import pytest

import pearlhash_ref as ref

pm = pytest.importorskip("pearl_mining")
pytest.importorskip("cupy")
import pearl_proof as pr  # noqa: E402
import run_stratum  # noqa: E402
from stratum_client import StratumClient, StratumError, StratumJob, parse_login  # noqa: E402

WALLET = "prl1p" + "q" * 20
CERT = 3
NBITS = 0x1E00FFFF


def make_job(job_id="deadbeef_1", height=100, nbits=NBITS, prev=0):
    h = ref.IncompleteBlockHeader(1, bytes([prev]) * 32, bytes(range(32, 64)), 1_700_000_000, nbits)
    return {"header": h.to_bytes().hex(), "height": height, "job_id": job_id,
            "target": format(ref.nbits_to_target(nbits), "064x")}


class FakePool:
    """Один клиент за раз; сообщения строго по спецификации."""

    def __init__(self, notify_before_ack=True, v2=False):
        self.job = make_job()
        self.notify_before_ack = notify_before_ack
        self.v2 = v2                     # пул поддерживает gzip-протокол (type v2 в authorize/ответе)
        self.gzip_on = False
        self.authorized = False
        self.shares = []                 # {"valid", "msg", "job_id"}
        self.seen_proofs = set()
        self.raw_requests = []
        self.connections = 0
        self.drop_after = None
        self._writer = None
        self._loop = None
        self._ready = threading.Event()
        self.port = None

    # ---- логика ----
    def _notify(self):
        return {"id": None, "method": "mining.notify", "params": self.job}

    error_style = "dict"                 # как пул кодирует ошибки: dict / list / str / message

    @staticmethod
    def _err(rid, code, msg):
        style = FakePool.error_style
        if style == "list":              # классический Stratum v1: [код, сообщение, данные]
            err = [code, msg, None]
        elif style == "str":
            err = f"{code}: {msg}"
        elif style == "message":
            err = {"code": code, "message": msg}
        else:
            err = {"code": code, "msg": msg}
        return {"error": err, "id": rid, "result": None}

    def _handle(self, req):
        self.raw_requests.append(req)
        rid, method, params = req.get("id"), req.get("method"), req.get("params")
        if not isinstance(params, dict):
            return [self._err(rid, 20, "params must be an object")]
        if method == "mining.authorize":
            wallet = params.get("wallet")
            if not wallet:
                return [self._err(rid, 24, "wallet missing")]
            if not wallet.startswith("prl1"):
                return [self._err(rid, 25, "invalid wallet")]
            self.authorized = True
            worker = params.get("worker")
            if "." in wallet:                                  # склейка wallet.worker (делится по первой точке)
                wallet, glued = wallet.split(".", 1)
                worker = worker or glued
            self.login = (wallet, worker or "default", params.get("pass"))
            ack = {"error": None, "id": rid, "result": True}
            self.gzip_on = bool(self.v2 and params.get("type") == "v2")
            if self.gzip_on:
                ack["type"] = "v2"
            return [self._notify(), ack] if self.notify_before_ack else [ack, self._notify()]
        if method == "mining.submit":
            if not self.authorized:
                return [self._err(rid, 27, "unauthorized")]
            if params.get("job_id") != self.job["job_id"]:
                return [self._err(rid, 21, "stale job")]
            key = params.get("plain_proof")
            if key in self.seen_proofs:
                return [self._err(rid, 22, "duplicate share")]
            try:
                raw = base64.b64decode(key, validate=True)
                wire = len(raw)
                if self.gzip_on and raw[:2] != b"\x1f\x8b":
                    return [self._err(rid, 26, "v2: plain_proof must be gzip")]
                was_gzip = raw[:2] == b"\x1f\x8b"
                if raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                elif raw[:1] == b"\x78":
                    raw = zlib.decompress(raw)
                proof = pm.PlainProof.from_base64(base64.b64encode(raw).decode())
                header = pm.IncompleteBlockHeader.from_bytes(bytes.fromhex(self.job["header"]))
                ok, msg = pm.verify_plain_proof_for_cert_version(CERT, header, proof)
            except Exception as e:      # noqa: BLE001
                return [self._err(rid, 26, f"invalid proof: {e}")]
            self.shares.append({"valid": ok, "msg": msg, "job_id": params["job_id"],
                                "gzip": was_gzip, "wire": wire, "raw": len(raw)})
            if not ok:
                return [self._err(rid, 23 if "target" in msg.lower() or "difficulty" in msg.lower() else 26, msg)]
            self.seen_proofs.add(key)
            return [{"error": None, "id": rid, "result": True}]
        return [self._err(rid, 20, f"method {method} not supported")]

    async def _client(self, reader, writer):
        self._writer = writer
        self.connections += 1
        self.authorized = False
        buf = b""
        while True:
            data = await reader.read(65536)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                for msg in self._handle(json.loads(line)):
                    writer.write((json.dumps(msg) + "\n").encode())
                await writer.drain()
                if self.drop_after is not None and len([x for x in self.shares if x["valid"]]) >= self.drop_after:
                    self.drop_after = None                      # один обрыв после N принятых шаров
                    writer.close()
                    return
        writer.close()

    def push_job(self, job):
        self.job = job

        async def go():
            self._writer.write((json.dumps(self._notify()) + "\n").encode())
            await self._writer.drain()

        asyncio.run_coroutine_threadsafe(go(), self._loop).result(5)

    def drop_client(self):
        async def go():
            self._writer.close()

        asyncio.run_coroutine_threadsafe(go(), self._loop).result(5)

    def send_raw(self, data: bytes):
        async def go():
            self._writer.write(data)
            await self._writer.drain()

        asyncio.run_coroutine_threadsafe(go(), self._loop).result(5)

    def start(self):
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._server = self._loop.run_until_complete(asyncio.start_server(self._client, "127.0.0.1", 0))
            self.port = self._server.sockets[0].getsockname()[1]
            self._ready.set()
            self._loop.run_forever()
            self._loop.run_until_complete(asyncio.sleep(0.05))
            self._loop.close()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert self._ready.wait(10)
        return self

    def stop(self):
        async def shutdown():
            self._server.close()
            await self._server.wait_closed()
            self._loop.stop()

        asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
        self._thread.join(5)


@pytest.fixture
def pool():
    p = FakePool().start()
    yield p
    p.stop()


def wait_for(pred, timeout=5.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def connect(pool):
    return StratumClient("127.0.0.1", pool.port, timeout=10).connect()


# ---------------------------------------------------------------------------
def test_parse_login_formats():
    assert parse_login("krxABCDEFG/worker1") == ("krxABCDEFG", "worker1")
    assert parse_login("prl1pabc.rig") == ("prl1pabc", "rig")
    assert parse_login("solo:prl1pabc.rig") == ("solo:prl1pabc", "rig")
    assert parse_login("prl1pabc") == ("prl1pabc", "default")
    assert parse_login("solo:wallet/w") == ("solo:wallet", "w")


@pytest.mark.parametrize("before_ack", [True, False])
def test_authorize_and_job(before_ack):
    p = FakePool(notify_before_ack=before_ack).start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig1", "d=100000")
            job = c.wait_job(5)
            assert isinstance(job, StratumJob)
            assert job.job_id == p.job["job_id"] and job.height == 100
            assert job.target == ref.nbits_to_target(NBITS)
            assert job.header_bytes.hex() == p.job["header"]
        assert p.login == (WALLET, "rig1", "d=100000")
    finally:
        p.stop()


def test_params_are_objects_never_arrays(pool):
    with connect(pool) as c:
        c.authorize(f"{WALLET}.rig")
        assert all(isinstance(r["params"], dict) for r in pool.raw_requests)
        assert pool.raw_requests[0]["method"] == "mining.authorize"      # subscribe не отправляется никогда
        assert not any(r["method"] == "mining.subscribe" for r in pool.raw_requests)


def test_authorize_errors(pool):
    with connect(pool) as c:
        with pytest.raises(StratumError) as e:
            c.authorize("notaprlwallet/rig")
        assert e.value.code == 25
        with pytest.raises(StratumError) as e:
            c.call("mining.authorize", {})
        assert e.value.code == 24
        with pytest.raises(StratumError) as e:
            c.call("mining.subscribe", {})
        assert e.value.code == 20


def test_submit_before_authorize_is_27(pool):
    with connect(pool) as c:
        with pytest.raises(StratumError) as e:
            c.submit("deadbeef_1", "AAAA")
        assert e.value.code == 27


def _share(job: StratumJob, seed=0):
    import pearl_pipeline as pp
    k = 2048
    cfg = ref.default_config(k)
    header = job.header()
    bound = ref.difficulty_bound(header.nbits, cfg)
    for s in range(seed, seed + 200):
        sol = pp.mine_random(256, 256, k, header, cfg, CERT, seed=s, bound=bound)
        if sol:
            return pr.proof_to_base64(pr.build_plain_proof(sol, header, cfg))
    raise AssertionError("не нашли решение для теста")


def test_valid_share_accepted_then_duplicate_and_stale(pool):
    with connect(pool) as c:
        c.authorize(f"{WALLET}/rig")
        job = c.wait_job(5)
        proof = _share(job)
        assert c.submit(job.job_id, proof) is True
        with pytest.raises(StratumError) as e:
            c.submit(job.job_id, proof)
        assert e.value.code == 22
        with pytest.raises(StratumError) as e:
            c.submit("00000000_9", proof)
        assert e.value.code == 21
        with pytest.raises(StratumError) as e:
            c.submit(job.job_id, base64.b64encode(b"garbage").decode())
        assert e.value.code == 26
    assert [s["valid"] for s in pool.shares] == [True]


def test_gzip_and_zlib_proofs_accepted(pool):
    with connect(pool) as c:
        c.authorize(f"{WALLET}/rig")
        job = c.wait_job(5)
        raw = base64.b64decode(_share(job, seed=0))
        assert c.submit(job.job_id, base64.b64encode(gzip.compress(raw)).decode()) is True
        raw2 = base64.b64decode(_share(job, seed=1000))
        assert c.submit(job.job_id, base64.b64encode(zlib.compress(raw2)).decode()) is True


def test_job_push_replaces_current_job(pool):
    with connect(pool) as c:
        c.authorize(f"{WALLET}/rig")
        j1 = c.wait_job(5)
        pool.push_job(make_job("cafebabe_2", height=101, prev=7))
        j2 = c.wait_job(5, newer_than=j1.job_id)
        assert j2.job_id == "cafebabe_2" and j2.height == 101 and j2.header_bytes != j1.header_bytes
        assert c.current_job().job_id == "cafebabe_2"


def test_partial_and_coalesced_lines(pool):
    with connect(pool) as c:
        c.authorize(f"{WALLET}/rig")
        c.wait_job(5)
        n1 = json.dumps({"id": None, "method": "mining.notify", "params": make_job("aaaaaaaa_1", 5)}) + "\n"
        n2 = json.dumps({"id": None, "method": "mining.notify", "params": make_job("bbbbbbbb_2", 6)}) + "\n"
        data = (n1 + n2).encode()
        pool.send_raw(data[:20])                    # строка пришла кусками
        pool.send_raw(data[20:len(n1) + 10])
        pool.send_raw(data[len(n1) + 10:])          # вторая строка хвостом вместе с первой
        assert wait_for(lambda: c.current_job().job_id == "bbbbbbbb_2")


def test_unknown_messages_and_extra_fields_ignored(pool):
    with connect(pool) as c:
        c.authorize(f"{WALLET}/rig")
        c.wait_job(5)
        job = dict(make_job("cccccccc_3", 7), future_field=123)
        pool.send_raw((json.dumps({"id": None, "method": "mining.notify", "params": job, "extra": 1}) + "\n").encode())
        pool.send_raw(b'{"id": null, "method": "mining.something_new", "params": {}}\n')
        pool.send_raw(b"not json at all\n")
        assert wait_for(lambda: c.current_job().job_id == "cccccccc_3")
        pool.send_raw(b'{"id": null, "method": "mining.notify", "params": {"job_id": "broken"}}\n')
        assert c.current_job().job_id == "cccccccc_3"   # битое задание не затирает текущее


def test_server_close_raises(pool):
    c = connect(pool)
    c.authorize(f"{WALLET}/rig")
    pool.drop_client()
    with pytest.raises((ConnectionError, OSError)):
        for _ in range(3):
            c.call("mining.authorize", {"wallet": WALLET})
    with pytest.raises(ConnectionError):
        c.wait_job(1, newer_than=c.current_job().job_id)
    c.close()


def test_miner_loop_submits_valid_shares(pool):
    with connect(pool) as c:
        c.authorize(f"{WALLET}/rig")
        stats = run_stratum.run(c, 256, 256, 2048, cert_version=CERT, max_shares=2, log=lambda *_: None)
    assert stats["accepted"] == 2 and stats["rejected"] == 0
    assert [s["valid"] for s in pool.shares] == [True, True]
    assert len({s["job_id"] for s in pool.shares}) == 1


def test_cert_version_from_notify_wins_and_fallback_is_explicit():
    """Kryptex присылает cert_version в notify — он главнее параметра; без него и без параметра майнер не гадает."""
    j = dict(make_job(), cert_version=2)
    from stratum_client import parse_job
    assert parse_job(j).cert_version == 2
    assert parse_job(make_job()).cert_version is None
    p = FakePool().start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            c.wait_job(5)
            with pytest.raises(RuntimeError, match="cert_version"):
                run_stratum.run(c, 256, 256, 2048, max_shares=1, log=lambda *_: None)
    finally:
        p.stop()


def test_miner_uses_cert_version_from_notify():
    p = FakePool().start()
    p.job = dict(make_job(), cert_version=CERT)
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            stats = run_stratum.run(c, 256, 256, 2048, cert_version=2, max_shares=1, log=lambda *_: None)   # параметр 2 игнорируется
        assert stats["accepted"] == 1 and p.shares[0]["valid"]
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# Длительная работа: переподключение, остановка на фатальных ошибках, отчёты
# ---------------------------------------------------------------------------
def test_mine_forever_reconnects_after_drop():
    p = FakePool().start()
    p.drop_after = 1
    logs = []
    try:
        stats = run_stratum.mine_forever("127.0.0.1", p.port, f"{WALLET}/rig", 256, 256, 2048, cert_version=CERT, max_shares=3,
                                         base_delay=0.05, log=logs.append)
        assert stats["accepted"] == 3 and stats["reconnects"] >= 1
        assert p.connections >= 2
        assert all(s["valid"] for s in p.shares)
    finally:
        p.stop()
    assert any("переподключ" in str(m) for m in logs)


def test_mine_forever_fatal_authorize_error_does_not_retry():
    p = FakePool().start()
    try:
        with pytest.raises(StratumError) as e:
            run_stratum.mine_forever("127.0.0.1", p.port, "notaprlwallet/rig", 256, 256, 2048,
                                     base_delay=0.01, log=lambda *_: None)
        assert e.value.code == 25
        assert p.connections == 1
    finally:
        p.stop()


def test_mine_forever_gives_up_when_pool_unreachable():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                                   # порт заведомо закрыт
    with pytest.raises((ConnectionError, OSError)):
        run_stratum.mine_forever("127.0.0.1", port, f"{WALLET}/rig", 256, 256, 2048, max_reconnects=2,
                                 base_delay=0.01, log=lambda *_: None)


def test_mine_forever_stop_event_returns_stats(pool):
    import threading
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    stats = run_stratum.mine_forever("127.0.0.1", pool.port, f"{WALLET}/rig", 256, 256, 2048, cert_version=CERT, stop=stop,
                                     base_delay=0.01, log=lambda *_: None)
    assert stats["attempts"] > 0


def test_periodic_report_has_rate_and_shares(pool):
    logs = []
    with connect(pool) as c:
        c.authorize(f"{WALLET}/rig")
        run_stratum.run(c, 256, 256, 2048, cert_version=CERT, max_shares=1, report_s=0.0, log=logs.append)
    text = "\n".join(map(str, logs))
    assert "попыток/с" in text and "TMAC/s" in text and "принято" in text


def test_console_output_never_crashes_on_legacy_codepage(monkeypatch):
    """Регрессия: при stdout в cp1251 печать «≈» в отчёте скорости роняла майнер через 30 секунд."""
    import io
    import sys

    import miner
    raw = io.BytesIO()
    fake = io.TextIOWrapper(raw, encoding="cp1251", errors="strict")
    monkeypatch.setattr(sys, "stdout", fake)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1251", errors="strict"))
    miner._setup_console()
    print("736 попыток/с ≈ 25.0 TMAC/s")
    sys.stdout.flush()
    assert "≈".encode("utf-8") in raw.getvalue()


# ---------------------------------------------------------------------------
# Протокол Kryptex «v2»: gzip для plain_proof, имя воркера wallet.worker, разреженные A и B
# ---------------------------------------------------------------------------
def test_authorize_sends_type_v2_and_glued_worker_name():
    p = FakePool(v2=True).start()
    try:
        with connect(p) as c:
            c.authorize("prl1pabc/rig7")
            req = p.raw_requests[0]["params"]
            assert req["type"] == "v2"
            assert req["wallet"] == "prl1pabc.rig7"                 # Kryptex берёт имя воркера из «wallet.worker»
            assert req["worker"] == "rig7"                          # явное поле тоже остаётся (спецификация Suprnova)
        assert p.login[:2] == ("prl1pabc", "rig7")
        p2 = FakePool(v2=True).start()
        with connect(p2) as c:
            c.authorize("prl1pabc")                                 # без воркера — без «.default»
            assert p2.raw_requests[0]["params"]["wallet"] == "prl1pabc"
        p2.stop()
    finally:
        p.stop()


def test_gzip_enabled_only_when_pool_confirms_v2():
    on, off = FakePool(v2=True).start(), FakePool(v2=False).start()
    try:
        with connect(on) as c1, connect(off) as c2:
            c1.authorize(f"{WALLET}/rig")
            c2.authorize(f"{WALLET}/rig")
            assert c1.gzip is True and c2.gzip is False
    finally:
        on.stop()
        off.stop()


@pytest.mark.parametrize("v2", [True, False])
def test_submit_uses_gzip_iff_negotiated(v2):
    p = FakePool(v2=v2).start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            job = c.wait_job(5)
            assert c.submit(job.job_id, _share(job)) is True
        assert p.shares[0]["valid"] and p.shares[0]["gzip"] is v2
    finally:
        p.stop()


def test_v2_pool_rejects_raw_proof_so_client_must_gzip():
    p = FakePool(v2=True).start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            job = c.wait_job(5)
            c.gzip = False                                          # имитация клиента без gzip
            with pytest.raises(StratumError) as e:
                c.submit(job.job_id, _share(job))
            assert e.value.code == 26
    finally:
        p.stop()


def test_sparse_generator_properties():
    import cupy as cp
    import pearl_pipeline as pp
    x = cp.asnumpy(pp.gen_sparse((512, 2048), seed=7, nnz=32))
    assert x.shape == (512, 2048) and x.dtype.name == "int8"
    nz = int((x != 0).sum())
    assert 1 <= nz <= 32 and x.min() >= -64 and x.max() <= 64
    assert (cp.asnumpy(pp.gen_sparse((512, 2048), seed=7, nnz=32)) == x).all()
    assert not (cp.asnumpy(pp.gen_sparse((512, 2048), seed=8, nnz=32)) == x).all()
    assert int((cp.asnumpy(pp.gen_sparse((4, 64), seed=1, nnz=1000)) != 0).sum()) <= 4 * 64     # nnz > размера не ломает


def test_sparse_proofs_compress_dense_proofs_do_not():
    import cupy as cp
    import pearl_pipeline as pp
    cfg = ref.default_config(2048)
    header = make_job_header()
    sizes = {}
    for name, gen in (("sparse", lambda s, sd: pp.gen_sparse(s, sd, 32)), ("dense", pp.gen_signal)):
        for sd in range(1, 200):
            a, bt = gen((256, 2048), sd * 2), gen((256, 2048), sd * 2 + 1)
            sol = pp.mine_attempt(a, bt, header, cfg, CERT)
            if sol:
                proof = pr.build_plain_proof(sol, header, cfg)
                raw = base64.b64decode(proof.to_base64())
                sizes[name] = len(gzip.compress(raw)) / len(raw)
                ok, msg = pr.verify_before_submit(CERT, header, proof)
                assert ok, msg
                break
    assert sizes["sparse"] < 0.15 and sizes["dense"] > 0.8, sizes


def make_job_header():
    return ref.IncompleteBlockHeader.from_bytes(bytes.fromhex(make_job()["header"]))


def test_sparse_matrices_find_shares_at_the_same_rate_as_dense():
    """Разреженный сигнал не должен менять вероятность находки: шум и хеши делают джекпот псевдослучайным."""
    import pearl_pipeline as pp
    cfg = ref.default_config(2048)
    header = make_job_header()
    bound = ref.U256_MAX // 2048                    # ≈ 39% попыток находят тайл
    n = 400
    hits = {}
    for name, gen in (("sparse", lambda s, sd: pp.gen_sparse(s, sd, 32)), ("dense", pp.gen_signal)):
        hits[name] = sum(
            pp.mine_attempt(gen((256, 2048), 2 * i + 1000), gen((256, 2048), 2 * i + 1001), header, cfg, CERT,
                            bound) is not None for i in range(n))
    assert abs(hits["sparse"] - hits["dense"]) < 0.12 * n, hits            # ~5σ при p≈0.39
    assert 0.2 * n < hits["sparse"] < 0.6 * n, hits


def test_miner_uses_gzip_and_sparse_matrices_when_pool_confirms_v2():
    p = FakePool(v2=True).start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            stats = run_stratum.run(c, 256, 256, 2048, cert_version=CERT, max_shares=3, log=lambda *_: None)
        assert stats["accepted"] == 3
        assert all(s["valid"] and s["gzip"] for s in p.shares)
        assert all(s["wire"] < 0.15 * s["raw"] for s in p.shares), [(s["wire"], s["raw"]) for s in p.shares]
    finally:
        p.stop()


def test_dense_flag_keeps_random_matrices():
    p = FakePool(v2=True).start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            run_stratum.run(c, 256, 256, 2048, cert_version=CERT, max_shares=2, dense=True, log=lambda *_: None)
        assert all(s["valid"] and s["gzip"] for s in p.shares)
        assert all(s["wire"] > 0.8 * s["raw"] for s in p.shares)
    finally:
        p.stop()


def test_friendly_errors_for_broken_bat_values(capsys):
    """Регрессия: в батнике вместо переменных вписали значения внутрь %...% — должно быть понятное сообщение."""
    import miner
    with pytest.raises(SystemExit) as e:
        miner.main(["--stratum", "--pool", "7048krxABCDEFG/worker1", "--wallet", "x"])
    assert "%" not in str(e.value) or "без процентов" in str(e.value)
    with pytest.raises(SystemExit) as e:
        miner.main(["--stratum", "--pool", "%prl-ru.kryptex.network:7048%", "--wallet", "x"])
    assert "без процентов" in str(e.value)
    assert miner.main(["--stratum", "--pool", "prl-ru.kryptex.network:7048", "--wallet", "%krxABCDEFG/worker1%"]) == 2
    assert "без знаков %" in capsys.readouterr().out
    assert miner.main(["--stratum", "--pool", "prl-ru.kryptex.network:7048", "--wallet", "ВАШ_USERNAME/worker1"]) == 2
    assert "заглушка" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Регрессия ночного падения: Kryptex ответил на stale-шар ошибкой-СПИСКОМ, и майнер упал
# с AttributeError: 'list' object has no attribute 'get'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("style,expected_code", [("dict", 21), ("list", 21), ("message", 21), ("str", 21)])
def test_error_formats_from_pool_are_all_understood(style, expected_code, monkeypatch):
    monkeypatch.setattr(FakePool, "error_style", style)
    p = FakePool(v2=True).start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            c.wait_job(5)
            with pytest.raises(StratumError) as e:
                c.submit("00000000_9", "AAAA")
            assert e.value.code == expected_code, style
    finally:
        p.stop()


def test_error_parser_edge_cases():
    from stratum_client import parse_error
    assert parse_error([21, "Job not found", None]) == (21, "Job not found")
    assert parse_error([21]) == (21, "")
    assert parse_error(["21", "x"]) == (21, "x")
    assert parse_error({"code": "23", "message": "low diff"}) == (23, "low diff")
    assert parse_error({"msg": "no code"}) == (0, "no code")
    assert parse_error("boom") == (0, "boom")
    assert parse_error(26) == (26, "")
    assert parse_error(["junk", "m"]) == (0, "m")
    assert parse_error([]) == (0, "")
    assert parse_error(True)[0] == 0


def test_miner_survives_stale_error_in_list_form(monkeypatch):
    monkeypatch.setattr(FakePool, "error_style", "list")
    p = FakePool(v2=True).start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            job = c.wait_job(5)
            real_submit = c.submit
            calls = {"n": 0}

            def flaky(job_id, proof):                      # первый шар «устарел» — как у Kryptex
                calls["n"] += 1
                return real_submit("00000000_9" if calls["n"] == 1 else job_id, proof)

            c.submit = flaky
            stats = run_stratum.run(c, 256, 256, 2048, cert_version=CERT, max_shares=3, log=lambda *_: None)
        assert stats["stale"] >= 1 and stats["accepted"] == 3
    finally:
        p.stop()


def test_submit_returning_false_is_not_counted_as_accepted():
    p = FakePool(v2=True).start()
    try:
        with connect(p) as c:
            c.authorize(f"{WALLET}/rig")
            c.wait_job(5)
            c.submit = lambda *_: False                    # пул ответил result=false без ошибки
            with pytest.raises(run_stratum.FatalMinerError):
                run_stratum.run(c, 256, 256, 2048, cert_version=CERT, max_shares=1, log=lambda *_: None)
            assert True
    finally:
        p.stop()


def test_mine_forever_restarts_after_unexpected_exception(monkeypatch):
    """Любое неожиданное исключение внутри цикла не должно останавливать ночной майнинг."""
    p = FakePool(v2=True).start()
    real_run = run_stratum.run
    state = {"n": 0}

    def exploding_run(*a, **kw):
        state["n"] += 1
        if state["n"] == 1:
            raise AttributeError("'list' object has no attribute 'get'")
        return real_run(*a, **kw)

    monkeypatch.setattr(run_stratum, "run", exploding_run)
    logs = []
    try:
        stats = run_stratum.mine_forever("127.0.0.1", p.port, f"{WALLET}/rig", 256, 256, 2048, cert_version=CERT,
                                         max_shares=2, base_delay=0.05, log=logs.append)
        assert stats["accepted"] == 2 and stats["reconnects"] >= 1
        assert any("Неожиданная ошибка" in str(m) for m in logs)
    finally:
        p.stop()


def test_mine_forever_gives_up_after_repeated_crashes(monkeypatch):
    p = FakePool(v2=True).start()
    monkeypatch.setattr(run_stratum, "run", lambda *a, **kw: (_ for _ in ()).throw(ValueError("всегда")))
    try:
        with pytest.raises(ValueError):
            run_stratum.mine_forever("127.0.0.1", p.port, f"{WALLET}/rig", 256, 256, 2048, cert_version=CERT,
                                     base_delay=0.01, max_crashes=3, log=lambda *_: None)
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# Автовыбор размера матриц: большие попытки лучше амортизируют накладные расходы (замер: 26 -> 41 TMAC/s)
# ---------------------------------------------------------------------------
def test_auto_matrix_size_follows_free_gpu_memory():
    gib = 1 << 30
    assert run_stratum.auto_size(2048, depth=3, free_bytes=6 * gib) == 16384
    assert run_stratum.auto_size(2048, depth=3, free_bytes=3 * gib) == 8192
    assert run_stratum.auto_size(2048, depth=3, free_bytes=int(1.2 * gib)) == 4096
    assert run_stratum.auto_size(2048, depth=3, free_bytes=64 << 20) == 4096        # минимум, дальше не уменьшаем
    assert run_stratum.auto_size(4096, depth=3, free_bytes=6 * gib) == 8192          # k вдвое больше -> попытка вдвое тяжелее
    assert run_stratum.auto_size(2048, depth=1, free_bytes=2 * gib) == 16384


def test_cli_auto_size_zero_means_auto(monkeypatch):
    import miner
    seen = {}

    def fake_forever(host, port, login, m, n, k, rank=128, **kw):
        seen.update(m=m, n=n, k=k, depth=kw.get("depth"))
        return {"accepted": 0}

    monkeypatch.setattr(run_stratum, "mine_forever", fake_forever)
    monkeypatch.setattr(run_stratum, "auto_size", lambda k, depth=3, free_bytes=None: 8192)
    monkeypatch.setattr(run_stratum, "auto_config", lambda free_bytes=None: (8192, 4096, 2))
    assert miner.main(["--stratum", "--pool", "127.0.0.1:1", "--wallet", f"{WALLET}/rig"]) == 0
    assert (seen["m"], seen["n"], seen["k"], seen["depth"]) == (8192, 8192, 4096, 2)
    seen.clear()
    assert miner.main(["--stratum", "--pool", "127.0.0.1:1", "--wallet", f"{WALLET}/rig", "--m", "4096", "--n", "2048"]) == 0
    assert (seen["m"], seen["n"], seen["k"]) == (4096, 2048, 2048)                    # явные значения главнее автовыбора
    seen.clear()
    assert miner.main(["--stratum", "--pool", "127.0.0.1:1", "--wallet", f"{WALLET}/rig", "--k", "2048"]) == 0
    assert (seen["m"], seen["n"], seen["k"], seen["depth"]) == (8192, 8192, 2048, 3)  # k задан вручную -> размер по auto_size


def test_auto_config_picks_k4096_only_when_it_fits():
    gib = 1 << 30
    assert run_stratum.auto_config(free_bytes=7 * gib) == (16384, 4096, 2)
    assert run_stratum.auto_config(free_bytes=int(5.1 * gib)) == (16384, 4096, 2)
    assert run_stratum.auto_config(free_bytes=int(4.9 * gib)) == (16384, 2048, 3)    # k=4096 не помещается -> прежний режим
    assert run_stratum.auto_config(free_bytes=3 * gib) == (8192, 2048, 3)
