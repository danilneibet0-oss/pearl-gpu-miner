"""
Тесты клиента gateway на локальном фейковом сервере (127.0.0.1, эфемерный порт, без внешней сети).

Фейковый сервер повторяет поведение miner_rpc/server.py: те же JSON-схемы (загружаются из официального
schemas.py), те же коды ошибок, ответ "submitted" и реальная проверка PlainProof официальным верификатором.
"""

import asyncio
import importlib.util
import json
import pathlib
import threading

import pytest

import pearlhash_ref as ref

pm = pytest.importorskip("pearl_mining")
pytest.importorskip("cupy")
fastjsonschema = pytest.importorskip("fastjsonschema")
import pearl_proof as pr  # noqa: E402
import run_miner  # noqa: E402
from gateway_client import GatewayClient, GatewayError, MiningJob, MiningPaused, job_bound  # noqa: E402

_SCHEMAS = pathlib.Path(__file__).parent / "pearl/miner/pearl-gateway/src/pearl_gateway/miner_rpc/schemas.py"
_spec = importlib.util.spec_from_file_location("gw_schemas", _SCHEMAS)
schemas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(schemas)


class FakeGateway:
    def __init__(self, job: MiningJob | None):
        self.job = job
        self.submissions: list[dict] = []     # {"valid": bool, "msg": str, "stale": bool}
        self.port = None
        self._ready = threading.Event()
        self._loop = None

    # --- логика, как в MinerRpcServer._process_request ---
    def _process(self, line: str):
        rid = None
        try:
            body = json.loads(line)
            try:
                schemas.validate_jsonrpc(body)
            except fastjsonschema.JsonSchemaException as e:
                return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request", "data": str(e)},
                        "id": body.get("id") if isinstance(body, dict) else None}
            method, params, rid = body["method"], body.get("params", {}), body["id"]
            if method == "getMiningInfo":
                try:
                    schemas.validate_get_mining_info(params)
                except fastjsonschema.JsonSchemaException as e:
                    return {"jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params", "data": str(e)}, "id": rid}
                if self.job is None:
                    return {"jsonrpc": "2.0", "error": {"code": -32001, "message": "mining_paused: no block template"}, "id": rid}
                return {"jsonrpc": "2.0", "result": self.job.to_dict(), "id": rid}
            if method == "submitPlainProof":
                try:
                    schemas.validate_submit_plain_proof(params)
                except fastjsonschema.JsonSchemaException as e:
                    return {"jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params", "data": str(e)}, "id": rid}
                proof = pm.PlainProof.from_base64(params["plain_proof"])
                mj = MiningJob.from_dict(params["mining_job"])
                stale = mj.incomplete_header_bytes != self.job.incomplete_header_bytes
                ok, msg = pm.verify_plain_proof_for_cert_version(
                    mj.cert_version, pm.IncompleteBlockHeader.from_bytes(mj.incomplete_header_bytes), proof)
                self.submissions.append({"valid": ok, "msg": msg, "stale": stale, "job": mj})
                return {"jsonrpc": "2.0", "result": "submitted", "id": rid}
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method {method} not found"}, "id": rid}
        except json.JSONDecodeError as e:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error", "data": str(e)}, "id": None}

    async def _handle(self, reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            writer.write((json.dumps(self._process(line.decode().strip())) + "\n").encode())
            await writer.drain()
        writer.close()

    def start(self):
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._server = self._loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", 0))
            self.port = self._server.sockets[0].getsockname()[1]
            self._ready.set()
            self._loop.run_forever()
            self._loop.run_until_complete(asyncio.sleep(0.05))      # дать proactor доделать отмену accept
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


def _job(nbits=0x1E00FFFF, cert=3):
    h = ref.IncompleteBlockHeader(1, bytes(range(32)), bytes(range(32, 64)), 1_700_000_000, nbits)
    return MiningJob(h.to_bytes(), ref.nbits_to_target(nbits), cert)


@pytest.fixture
def gateway():
    gw = FakeGateway(_job()).start()
    yield gw
    gw.stop()


def test_get_mining_info(gateway):
    with GatewayClient(port=gateway.port) as c:
        assert c.get_mining_info() == gateway.job
        assert c.get_mining_info() == gateway.job           # id растут, соединение живёт


def test_mining_paused_is_distinct_error():
    gw = FakeGateway(None).start()
    try:
        with GatewayClient(port=gw.port) as c:
            with pytest.raises(MiningPaused):
                c.get_mining_info()
            gw.job = _job()
            assert c.get_mining_info() == gw.job            # после паузы работает на том же соединении
    finally:
        gw.stop()


def test_unknown_method_and_bad_params(gateway):
    with GatewayClient(port=gateway.port) as c:
        with pytest.raises(GatewayError) as e:
            c.call("nope")
        assert e.value.code == -32601
        with pytest.raises(GatewayError) as e:
            c.call("getMiningInfo", {"x": 1})
        assert e.value.code == -32602
        with pytest.raises(GatewayError) as e:
            c.call("submitPlainProof", {"plain_proof": "###", "mining_job": {}})
        assert e.value.code == -32602


def test_client_refuses_remote_hosts_by_default():
    with pytest.raises(ValueError):
        GatewayClient(host="8.8.8.8")
    with pytest.raises(ValueError):
        GatewayClient(host="example.com")
    GatewayClient(host="localhost")
    GatewayClient(host="::1")


def test_connection_closed_then_reconnect(gateway):
    c = GatewayClient(port=gateway.port)
    c.connect()
    assert c.get_mining_info() == gateway.job
    c._sock.close()                                          # имитация обрыва
    with pytest.raises(OSError):
        c.get_mining_info()
    assert c.get_mining_info() == gateway.job                # следующий вызов переподключается сам
    c.close()


def test_job_bound_matches_official_penalized_bound():
    for k in (2048, 4096):
        cfg = ref.default_config(k)
        job = _job()
        official_cfg = pm.MiningConfiguration(
            k, 128, pm.MMAType.Int7xInt7ToInt32,
            pm.PeriodicPattern.from_list([0, 8, 64, 72]),
            pm.PeriodicPattern.from_list([0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105]))
        assert job_bound(job, cfg) == pm.penalized_target_bound(job.target, official_cfg)
    # для rank 128 штрафа нет: совпадает с границей консенсуса, которой пользуется эталон
    cfg = ref.default_config(2048)
    j = _job(0x1E00FFFF)
    assert job_bound(j, cfg) == ref.difficulty_bound(0x1E00FFFF, cfg)
    with pytest.raises(ValueError):
        job_bound(j, ref.default_config(2048, rank=64))


def test_end_to_end_miner_submits_valid_proof(gateway):
    with GatewayClient(port=gateway.port) as c:
        stats = run_miner.run(c, 256, 256, 2048, max_solutions=2, log=lambda *_: None)
    assert stats["submitted"] == 2 and stats["rejected_locally"] == 0
    assert len(gateway.submissions) == 2
    assert all(s["valid"] and not s["stale"] for s in gateway.submissions), gateway.submissions
    assert all(s["job"] == gateway.job for s in gateway.submissions)         # cert_version взят из задания


@pytest.mark.parametrize("cert", [2, 3])
def test_cert_version_comes_from_job(cert):
    gw = FakeGateway(_job(cert=cert)).start()
    try:
        with GatewayClient(port=gw.port) as c:
            run_miner.run(c, 256, 256, 2048, max_solutions=1, log=lambda *_: None)
        assert gw.submissions[0]["job"].cert_version == cert and gw.submissions[0]["valid"]
    finally:
        gw.stop()
