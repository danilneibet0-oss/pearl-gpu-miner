"""
Тесты упаковки решения в PlainProof и официальной проверки (pearl_mining).
Решения ищет GPU-конвейер; проверяет официальный Rust-верификатор.
Запуск:  .venv\\Scripts\\python -m pytest test_pearlhash_proof.py -q
"""

import numpy as np
import pytest

import pearlhash_ref as ref

pm = pytest.importorskip("pearl_mining")
cp = pytest.importorskip("cupy")
import pearl_pipeline as pp  # noqa: E402
import pearl_proof as pr  # noqa: E402


def _header(nbits):
    return ref.IncompleteBlockHeader(1, bytes(range(32)), bytes(range(32, 64)), 1_700_000_000, nbits)


def _mine(cert, nbits=0x1E00FFFF, shape=(256, 256, 2048), seed=0):
    m, n, k = shape
    cfg = ref.default_config(k)
    header = _header(nbits)
    sol = pp.mine_random(m, n, k, header, cfg, cert, seed=seed)
    assert sol is not None
    return sol, header, cfg


def test_header_bytes_match_official():
    h = _header(0x1E00FFFF)
    assert pr.to_pm_header(h).to_bytes() == h.to_bytes()
    assert pr.header_from_job(h.to_bytes()).to_bytes() == h.to_bytes()


def test_mining_config_bytes_match_official():
    cfg = ref.default_config(2048)
    official = pm.MiningConfiguration(
        2048, 128, pm.MMAType.Int7xInt7ToInt32,
        pm.PeriodicPattern.from_list([0, 8, 64, 72]),
        pm.PeriodicPattern.from_list([0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105]))
    assert official.to_bytes() == cfg.to_bytes()


@pytest.mark.parametrize("cert", [2, 3])
@pytest.mark.parametrize("shape", [(256, 256, 2048), (512, 384, 2176)])
def test_gpu_solution_verifies_officially(cert, shape):
    sol, header, cfg = _mine(cert, shape=shape)
    proof = pr.build_plain_proof(sol, header, cfg)
    ok, msg = pr.verify_before_submit(cert, header, proof)
    assert ok, msg


@pytest.mark.parametrize("cert", [2, 3])
def test_base64_roundtrip_still_verifies(cert):
    sol, header, cfg = _mine(cert, seed=3)
    proof = pr.build_plain_proof(sol, header, cfg)
    again = pm.PlainProof.from_base64(pr.proof_to_base64(proof))
    ok, msg = pr.verify_before_submit(cert, header, again)
    assert ok, msg


def test_wrong_cert_version_rejected():
    """Решение, найденное с солёными сидами (cert 3), не проходит как cert 2 — и наоборот."""
    for mined, checked in ((3, 2), (2, 3)):
        sol, header, cfg = _mine(mined)
        proof = pr.build_plain_proof(sol, header, cfg)
        ok, _ = pr.verify_before_submit(checked, header, proof)
        assert not ok


def test_harder_target_rejected():
    sol, header, cfg = _mine(3)
    proof = pr.build_plain_proof(sol, header, cfg)
    harder = ref.IncompleteBlockHeader(1, header.prev_block, header.merkle_root, header.timestamp, 0x1B00FFFF)
    ok, _ = pr.verify_before_submit(3, harder, proof)
    assert not ok


def test_tampered_matrix_rejected():
    sol, header, cfg = _mine(3)
    sol.a = sol.a.copy()
    sol.a[sol.a_rows[0], 0] ^= 1             # доказательство строится по изменённым данным -> другой корень
    proof = pr.build_plain_proof(sol, header, cfg)
    ok, _ = pr.verify_before_submit(3, header, proof)
    assert not ok


def test_official_cpu_miner_agrees_with_our_verifier_path():
    """Контроль самого теста: официальный mine() даёт доказательство, проходящее ту же проверку."""
    header = _header(0x2100FFFF)
    cfg = pm.MiningConfiguration(
        2048, 128, pm.MMAType.Int7xInt7ToInt32,
        pm.PeriodicPattern.from_list([0, 8, 64, 72]),
        pm.PeriodicPattern.from_list([0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105]))
    plain = pm.mine(256, 256, 2048, pr.to_pm_header(header), cfg, cert_version=3)
    assert plain is not None
    ok, msg = pm.verify_plain_proof_for_cert_version(3, pr.to_pm_header(header), plain)
    assert ok, msg


@pytest.mark.parametrize("sparse", [None, 32])
def test_large_attempt_verifies_officially(sparse):
    """Крупные попытки (как при автоподборе размера) тоже дают доказательство, проходящее официальную проверку."""
    m, n, k = 8192, 4096, 2048
    cfg = ref.default_config(k)
    header = _header(0x1E00FFFF)
    sol = None
    for seed in range(20):
        sol = pp.mine_random(m, n, k, header, cfg, 3, seed=seed, sparse_nnz=sparse)
        if sol:
            break
    assert sol is not None and sol.a.shape == (m, k) and sol.b.shape == (k, n)
    proof = pr.build_plain_proof(sol, header, cfg)
    ok, msg = pr.verify_before_submit(3, header, proof)
    assert ok, msg
