"""
run_stratum.py — цикл майнинга для пула (stratum-диалект Suprnova/Pearl, см. stratum_client.py).

Проверено на реальном пуле Kryptex (prl-ru.kryptex.network:7048): авторизация, задания и принятый шар.

Места, где спецификация молчит, — явные параметры, а не догадки:
  * cert_version — берётся из mining.notify, если пул его присылает (Kryptex присылает; в спецификации Suprnova поля нет).
    Иначе обязателен параметр cert_version. Определяет сиды шума: 1/2 — старые, 3 — «солёные».
  * target — трактуется как цель ДО штрафа ранга (как target в задании gateway): граница джекпот-хеша =
    target × размер_тайла × (dot_length/rank) × 128. Если пул считает иначе, он ответит кодом 23 (низкая сложность);
    после нескольких таких ответов майнер останавливается, а не спамит пул.

mine_forever — длительная работа: переподключение с экспоненциальной паузой, остановка на фатальных ошибках
(неверный кошелёк, серия отказов), периодические отчёты о скорости и шарах.
"""

from __future__ import annotations

import os
import threading
import time

import pearl_pipeline as pp
import pearl_proof as pr
import pearlhash_ref as ref
from gateway_client import job_bound
from stratum_client import StratumClient, StratumError

EASIEST_NBITS = 0x2100FFFF       # локальная проверка структуры доказательства (сложность проверяем сами по границе)
MAX_BAD = 3                      # столько отказов 23/26 подряд -> остановка
SPARSE_NNZ = 32                  # ненулевых ячеек в A и в B^T при разреженном режиме
FATAL_AUTH = {24, 25}            # нет кошелька / неверный кошелёк: повторять бессмысленно


def new_stats() -> dict:
    return {"attempts": 0, "found": 0, "accepted": 0, "rejected": 0, "stale": 0, "duplicate": 0, "reconnects": 0}


def _report(stats: dict, m: int, n: int, k: int, since: float, attempts0: int, log) -> None:
    dt = max(time.time() - since, 1e-9)
    rate = (stats["attempts"] - attempts0) / dt
    tmacs = rate * m * n * k / 1e12
    log(f"{rate:.0f} попыток/с ≈ {tmacs:.1f} TMAC/s | шаров: принято {stats['accepted']}, "
        f"отклонено {stats['rejected']}, устарело {stats['stale']}, дубликатов {stats['duplicate']} | "
        f"попыток всего {stats['attempts']}")


def run(client: StratumClient, m: int, n: int, k: int, rank: int = 128, *, cert_version: int | None = None,
        max_shares: int | None = None, batch: int = 16, depth: int = 3,
        stop: threading.Event | None = None, log=print, stats: dict | None = None,
        report_s: float = 30.0, dense: bool = False, sparse_nnz: int = SPARSE_NNZ) -> dict:
    stop = stop or threading.Event()
    cfg = ref.default_config(k, rank)
    stats = stats if stats is not None else new_stats()
    bad_in_row, t0 = 0, time.time()
    last_rep, rep_attempts = time.time(), stats["attempts"]
    job = client.wait_job(60)
    log(f"задание {job.job_id}, высота {job.height}")
    while not stop.is_set() and (max_shares is None or stats["accepted"] < max_shares):
        cur = client.current_job()
        if cur is not None and cur.job_id != job.job_id:
            job = cur
            log(f"новое задание {job.job_id}, высота {job.height}")
        bound = job_bound(job, cfg)
        if bound is None:
            raise RuntimeError("цель задания неприменима к этой конфигурации")
        header = job.header()
        cert = job.cert_version if job.cert_version is not None else cert_version
        if cert is None:
            raise RuntimeError("пул не прислал cert_version, а параметр cert_version не задан")
        base = int.from_bytes(os.urandom(6), "little")
        for _seed, sol in pp.mine_random_pipelined(m, n, k, header, cfg, cert,
                                                   range(base, base + batch), bound=bound, depth=depth,
                                                   sparse_nnz=None if (dense or not client.gzip) else sparse_nnz):
            stats["attempts"] += 1
            if time.time() - last_rep >= report_s:
                _report(stats, m, n, k, last_rep, rep_attempts, log)
                last_rep, rep_attempts = time.time(), stats["attempts"]
            now = client.current_job()
            if now is not None and now.job_id != job.job_id:
                break                                   # задание сменилось — решения по старому будут stale
            if sol is None:
                continue
            stats["found"] += 1
            proof = pr.build_plain_proof(sol, header, cfg)
            ok, msg = pr.verify_before_submit(cert, header, proof, nbits_override=EASIEST_NBITS)
            if not ok:
                log(f"решение не прошло локальную проверку структуры, не отправляю: {msg}")
                stats["rejected"] += 1
                continue
            try:
                client.submit(job.job_id, pr.proof_to_base64(proof))
            except StratumError as e:
                if e.code == 21:
                    stats["stale"] += 1
                    log("шар устарел (21)")
                    break
                if e.code == 22:
                    stats["duplicate"] += 1
                    continue
                stats["rejected"] += 1
                bad_in_row += 1
                log(f"шар отклонён: {e}")
                if bad_in_row >= MAX_BAD:
                    raise RuntimeError(f"{MAX_BAD} отказа подряд ({e}); останавливаюсь, чтобы не спамить пул. "
                                       "Проверьте cert_version и трактовку target (см. docstring run_stratum).") from e
                continue
            bad_in_row = 0
            stats["accepted"] += 1
            log(f"шар принят ({stats['accepted']}), попыток {stats['attempts']}, {time.time() - t0:.1f} c")
            if max_shares is not None and stats["accepted"] >= max_shares:
                break
            if stop.is_set():
                break
    return stats


def mine_forever(host: str, port: int, login: str, m: int, n: int, k: int, rank: int = 128, *,
                 password: str | None = None, tls: bool = False, cert_version: int | None = None,
                 max_shares: int | None = None, max_reconnects: int | None = None,
                 base_delay: float = 2.0, max_delay: float = 60.0,
                 stop: threading.Event | None = None, log=print, report_s: float = 30.0,
                 dense: bool = False) -> dict:
    """
    Майнинг «пока не остановят»: при обрыве/таймауте — переподключение с паузой base_delay·2^n (до max_delay),
    пауза сбрасывается после успешной авторизации. Фатальные ошибки (неверный кошелёк, серия отказов шаров)
    пробрасываются наружу без повторов. max_reconnects=None — бесконечно (для тестов можно ограничить).
    """
    stop = stop or threading.Event()
    stats = new_stats()
    fails = 0
    while not stop.is_set() and (max_shares is None or stats["accepted"] < max_shares):
        try:
            with StratumClient(host, port, tls=tls, timeout=30) as client:
                client.authorize(login, password)
                fails = 0
                log(f"подключено к {host}:{port}, авторизация принята; gzip (v2): {'да' if client.gzip else 'нет'}, "
                    f"матрицы: {'случайные' if (dense or not client.gzip) else 'разреженные'}")
                run(client, m, n, k, rank, cert_version=cert_version, max_shares=max_shares, stop=stop,
                    log=log, stats=stats, report_s=report_s, dense=dense)
        except StratumError as e:
            if e.code in FATAL_AUTH:
                raise
            log(f"ошибка пула: {e}")
        except (ConnectionError, TimeoutError, OSError) as e:
            log(f"связь с пулом потеряна: {e}")
            if max_reconnects is not None and fails >= max_reconnects:
                raise
        else:
            continue                                       # run завершился штатно (stop / max_shares)
        fails += 1
        stats["reconnects"] += 1
        delay = min(max_delay, base_delay * 2 ** (fails - 1))
        log(f"переподключение через {delay:.1f} с (попытка {fails})")
        if stop.wait(delay):
            break
    return stats
