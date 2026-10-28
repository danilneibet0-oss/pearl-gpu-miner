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


MATRIX_SIZES = (16384, 8192, 4096)        # кандидаты для m = n (кратны 128, период паттернов)


def auto_size(k: int, depth: int = 3, free_bytes: int | None = None) -> int:
    """
    Размер m = n для попытки. Чем больше матрицы, тем лучше накладные расходы (хеши, шум) делятся на работу ядра:
    замер на RTX 3060 Ti — 26 TMAC/s при 4096 и 41 TMAC/s при 16384. Берём наибольший размер, для которого
    depth одновременных попыток займут не больше 60% свободной памяти GPU (с запасом на кэш аллокатора: ×6).
    """
    if free_bytes is None:
        import cupy as cp
        free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
    for size in MATRIX_SIZES:
        need = depth * 6 * 4 * size * k          # A, B, A_шум, B_шум на попытку
        if need <= 0.6 * free_bytes:
            return size
    return MATRIX_SIZES[-1]


AUTO_K_SIZE, AUTO_K, AUTO_K_DEPTH = 16384, 4096, 2      # единственная конфигурация, где k=4096 выигрывает у 2048


def auto_config(free_bytes: int | None = None) -> tuple[int, int, int]:
    """
    (m = n, k, depth) по свободной памяти GPU. Замер на RTX 3060 Ti: 16384x16384x4096 при двух попытках в полёте —
    43.7 TMAC/s против 41.9 при k=2048; на меньших матрицах k=4096 не лучше, поэтому берём его только если эта
    конфигурация помещается в 60% свободной памяти, иначе k=2048 и размер по auto_size.
    """
    if free_bytes is None:
        import cupy as cp
        free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
    if AUTO_K_DEPTH * 6 * 4 * AUTO_K_SIZE * AUTO_K <= 0.6 * free_bytes:
        return AUTO_K_SIZE, AUTO_K, AUTO_K_DEPTH
    return auto_size(2048, 3, free_bytes), 2048, 3


class FatalMinerError(RuntimeError):
    """Намеренная остановка майнера (повторять бессмысленно): серия отказов, неприменимая цель, нет cert_version."""


def new_stats() -> dict:
    return {"attempts": 0, "found": 0, "accepted": 0, "rejected": 0, "stale": 0, "duplicate": 0, "reconnects": 0,
            "expected": 0.0}


def _report(stats: dict, m: int, n: int, k: int, since: float, attempts0: int, log) -> None:
    dt = max(time.time() - since, 1e-9)
    rate = (stats["attempts"] - attempts0) / dt
    tmacs = rate * m * n * k / 1e12
    log(f"{rate:.0f} попыток/с ≈ {tmacs:.1f} TMAC/s | шаров: принято {stats['accepted']}, "
        f"отклонено {stats['rejected']}, устарело {stats['stale']}, дубликатов {stats['duplicate']} | "
        f"попыток всего {stats['attempts']}, ожидалось шаров по целям заданий {stats['expected']:.1f}")


def run(client: StratumClient, m: int, n: int, k: int, rank: int = 128, *, cert_version: int | None = None,
        max_shares: int | None = None, batch: int = 16, depth: int = 3,
        stop: threading.Event | None = None, log=print, stats: dict | None = None,
        report_s: float = 30.0, dense: bool = False, sparse_nnz: int = SPARSE_NNZ) -> dict:
    stop = stop or threading.Event()
    cfg = ref.default_config(k, rank)
    rt, ct = pp.tiles_for(cfg, m, n)
    n_tiles, last_target = len(rt) * len(ct), None
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
            raise FatalMinerError("цель задания неприменима к этой конфигурации")
        p_hit = n_tiles * bound / (ref.U256_MAX + 1)        # вероятность шара за попытку (тайлы независимы)
        if job.target != last_target:
            last_target = job.target
            log(f"цель задания {job.target:064x}, шаров на попытку ≈ {p_hit:.3g} (≈ {1 / p_hit:.0f} попыток на шар)")
        header = job.header()
        cert = job.cert_version if job.cert_version is not None else cert_version
        if cert is None:
            raise FatalMinerError("пул не прислал cert_version, а параметр cert_version не задан")
        base = int.from_bytes(os.urandom(6), "little")
        for _seed, sol in pp.mine_random_pipelined(m, n, k, header, cfg, cert,
                                                   range(base, base + batch), bound=bound, depth=depth,
                                                   sparse_nnz=None if (dense or not client.gzip) else sparse_nnz):
            stats["attempts"] += 1
            stats["expected"] += p_hit
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
                accepted = client.submit(job.job_id, pr.proof_to_base64(proof))
                if not accepted:                          # result=false без объекта ошибки — это отказ, а не принятый шар
                    raise StratumError(0, "пул вернул result=false")
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
                    raise FatalMinerError(f"{MAX_BAD} отказа подряд ({e}); останавливаюсь, чтобы не спамить пул. "
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
                 dense: bool = False, max_crashes: int = 5, depth: int = 3) -> dict:
    """
    Майнинг «пока не остановят»: при обрыве/таймауте — переподключение с паузой base_delay·2^n (до max_delay),
    пауза сбрасывается после успешной авторизации. Фатальные ошибки (неверный кошелёк, серия отказов шаров)
    пробрасываются наружу без повторов. max_reconnects=None — бесконечно (для тестов можно ограничить).
    Неожиданное исключение (ошибка в самом майнере или нестандартный ответ пула) не убивает ночной майнинг:
    traceback пишется в лог, соединение пересоздаётся; но max_crashes подряд без единого принятого шара — остановка.
    """
    stop = stop or threading.Event()
    stats = new_stats()
    fails = 0
    crashes, accepted_at_crash = 0, -1
    while not stop.is_set() and (max_shares is None or stats["accepted"] < max_shares):
        try:
            with StratumClient(host, port, tls=tls, timeout=30) as client:
                client.authorize(login, password)
                fails = 0
                log(f"подключено к {host}:{port}, авторизация принята; gzip (v2): {'да' if client.gzip else 'нет'}, "
                    f"матрицы: {'случайные' if (dense or not client.gzip) else 'разреженные'}")
                run(client, m, n, k, rank, cert_version=cert_version, max_shares=max_shares, stop=stop,
                    log=log, stats=stats, report_s=report_s, dense=dense, depth=depth)
        except FatalMinerError:
            raise
        except StratumError as e:
            if e.code in FATAL_AUTH:
                raise
            log(f"ошибка пула: {e}")
        except (ConnectionError, TimeoutError, OSError) as e:
            log(f"связь с пулом потеряна: {e}")
            if max_reconnects is not None and fails >= max_reconnects:
                raise
        except Exception as e:                              # noqa: BLE001 — намеренно: майнер не должен падать молча
            import traceback
            log("Неожиданная ошибка (майнер продолжит работу): " + "".join(traceback.format_exception(e)).strip())
            crashes = crashes + 1 if stats["accepted"] == accepted_at_crash else 1
            accepted_at_crash = stats["accepted"]
            if crashes >= max_crashes:
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
