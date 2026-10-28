"""
miner.py — командная строка в стиле готовых майнеров:

  miner.py --disable-cpu --algorithm pearlhash --pool 127.0.0.1:8337 --wallet <wallet>

Два режима:
  * локальный pearl-gateway (127.0.0.1 / localhost) — по умолчанию; gateway сам хранит кошелёк;
  * пул по Stratum (диалект из публичной спецификации Suprnova, см. stratum_client.py) — ТОЛЬКО с флагом --stratum,
    чтобы случайно не подключиться к внешнему серверу. Диалект проверен на Kryptex (prl-ru.kryptex.network:7048);
    на других пулах не гарантирован.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys

if getattr(sys, "frozen", False):
    # Miner.exe (PyInstaller): CUDA-библиотеки (NVRTC, cudart) лежат в <папка_приложения>\_internalin
    os.environ["CUDA_PATH"] = sys._MEIPASS

LOCAL = {"localhost"}


def _is_local(host: str) -> bool:
    if host in LOCAL:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


PLACEHOLDERS = ("ВАШ_USERNAME", "YOUR_USERNAME", "<WALLET>")        # сравниваются с wallet в верхнем регистре


def parse_pool(pool: str) -> tuple[str, int]:
    pool = pool.split("://", 1)[-1]
    host, _, port = pool.rpartition(":")
    if not host or not port.isdigit():
        hint = " В строке остались знаки % — впишите значение прямо, без процентов." if "%" in pool else ""
        raise SystemExit(f"Неверный --pool {pool!r}: ожидается host:port, например prl-ru.kryptex.network:7048.{hint}")
    return host, int(port)


def _sizes(args) -> tuple[int, int]:
    """m, n попытки: заданные вручную или автоматически по свободной памяти GPU (0 = авто)."""
    import run_stratum
    m, n = args.m, args.n
    if not m or not n:
        try:
            auto = run_stratum.auto_size(args.k)
        except Exception as e:      # noqa: BLE001 — нет GPU/CUDA: честно сообщаем, а не падаем с трассировкой
            print(f"Не удалось определить память GPU ({e}); ставлю размер 4096. Укажите --m и --n вручную при необходимости.")
            auto = 4096
        m, n = m or auto, n or auto
    return m, n


def run_pool(args, host: str, port: int) -> int:
    import threading
    from stratum_client import StratumError
    import run_stratum

    m, n = _sizes(args)
    print(f"Майнинг в пул {host}:{port} ({'TLS' if args.tls else 'TCP'}), логин {args.wallet}, матрицы {m}x{n}x{args.k}. "
          "Остановка: Ctrl+C.")
    stop = threading.Event()
    stats = None
    try:
        stats = run_stratum.mine_forever(host, port, args.wallet, m, n, args.k, args.rank,
                                         password=args.password or None, tls=args.tls,
                                         cert_version=args.cert_version, stop=stop, dense=args.dense)
    except KeyboardInterrupt:
        stop.set()
        print("Остановлено пользователем.")
        return 0
    except (StratumError, RuntimeError, ConnectionError, TimeoutError, OSError, ValueError, AttributeError) as e:
        print(f"Остановлено: {e}")
        return 1
    print(stats)
    return 0


def _setup_console() -> None:
    """Вывод всегда в UTF-8 (с заменой, а не исключением): Miner.exe игнорирует PYTHONIOENCODING, а при перенаправлении
    вывода Windows берёт cp1251/cp866 и падает на символах вроде «≈»."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def _run_as_python_dash_m() -> None:
    """В Miner.exe sys.executable — сам exe: библиотека cuda.pathfinder запускает дочерний `exe -m модуль ...`."""
    import runpy
    module = sys.argv[2]
    sys.argv = [module] + sys.argv[3:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    raise SystemExit(0)


def main(argv=None) -> int:
    _setup_console()
    if getattr(sys, "frozen", False) and argv is None and len(sys.argv) >= 3 and sys.argv[1] == "-m":
        _run_as_python_dash_m()
    ap = argparse.ArgumentParser(description="Pearl (pearlhash) GPU miner")
    ap.add_argument("--disable-cpu", action="store_true", help="совместимость: CPU не используется никогда")
    ap.add_argument("--algorithm", default="pearlhash", help="только pearlhash")
    ap.add_argument("--pool", default="", help="host:port pearl-gateway (локально), напр. 127.0.0.1:8337")
    ap.add_argument("--wallet", default="", help="кошелёк (в режиме gateway хранится в самом gateway)")
    ap.add_argument("--selftest", action="store_true", help="самопроверка GPU и ядер (ничего не подключает)")
    ap.add_argument("--stratum", action="store_true", help="разрешить подключение к внешнему пулу по Stratum")
    ap.add_argument("--tls", action="store_true", help="TLS (для пула: SSL-порт)")
    ap.add_argument("--cert-version", type=int, default=None,
                    help="версия сертификата, если пул не присылает её в mining.notify (Kryptex присылает)")
    ap.add_argument("--dense", action="store_true",
                    help="всегда случайные (плотные) матрицы; по умолчанию при gzip-протоколе пула — разреженные")
    ap.add_argument("--password", default="", help="пароль пула (например d=500000 для стартовой сложности)")
    ap.add_argument("--m", type=int, default=0, help="строк A в попытке; 0 — автоматически по памяти GPU (до 16384)")
    ap.add_argument("--n", type=int, default=0, help="столбцов B в попытке; 0 — автоматически")
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=128)
    args = ap.parse_args(argv)
    if args.selftest:
        import selftest
        return 0 if selftest.run() else 1
    if not args.pool:
        ap.error("укажите --pool host:port (или --selftest)")

    if args.algorithm.lower() != "pearlhash":
        print(f"Алгоритм {args.algorithm!r} не поддерживается, только pearlhash.")
        return 2
    host, port = parse_pool(args.pool)
    if args.stratum:
        if not args.wallet:
            print("Для пула нужен --wallet (адрес или username/worker).")
            return 2
        if "%" in args.wallet or args.wallet.strip().upper().startswith(PLACEHOLDERS):
            print(f"Вместо логина стоит заглушка или остатки переменной: {args.wallet!r}.\n"
                  "Впишите свой логин в батнике в виде username/имя_воркера (например krxABCDEFG/rig1), без знаков %.")
            return 2
        return run_pool(args, host, port)
    if not _is_local(host):
        print(f"""
{host}:{port} — не локальный gateway. Чтобы подключиться к внешнему пулу по Stratum, добавьте флаг --stratum.
Диалект (спецификация Suprnova) проверен на Kryptex, на других пулах не гарантирован. Ничего не отправлено.""")
        return 2
    if args.wallet:
        print("Примечание: в режиме gateway кошелёк задаётся в настройках самого gateway; --wallet здесь не используется.")

    from gateway_client import GatewayClient
    import run_miner

    m, n = _sizes(args)
    print(f"GPU-майнер pearlhash -> gateway {host}:{port}, матрицы {m}x{n}x{args.k}, rank {args.rank}")
    try:
        with GatewayClient(host, port) as client:
            stats = run_miner.run(client, m, n, args.k, args.rank)
        print(stats)
    except KeyboardInterrupt:
        print("Остановлено.")
    except ConnectionRefusedError:
        print(f"Не удалось подключиться к gateway {host}:{port}: он не запущен? (см. README)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
