"""
build_exe.py — сборка Miner.exe (PyInstaller, режим onedir) и пакета для раздачи.

  .venv\\Scripts\\python build_exe.py

Итог: release\\PearlMiner\\ (Miner.exe, _internal\\, start_miner.bat, README.txt) и release\\PearlMiner.zip.
Пользователю нужны только драйвер NVIDIA (RTX 30xx и новее); Python, CUDA Toolkit и Rust не нужны.
"""

import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
SITE = pathlib.Path(sys.prefix) / "Lib" / "site-packages"
NVBIN = SITE / "nvidia" / "cu13" / "bin" / "x86_64"
# nvJitLink (93 МБ) не нужен: NVRTC собирает cubin сразу под архитектуру карты (проверено selftest)
NEEDED = ["cudart64_13.dll", "nvrtc64_130_0.dll", "nvrtc-builtins64_134.dll"]
# крупные библиотеки CUDA, которые майнеру не нужны (нет cuBLAS/cuFFT/... в нашем коде)
UNUSED_PREFIXES = ("cublas", "cublasLt", "cufft", "curand", "cusolver", "cusparse", "nvblas", "cutensor")


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "release"          # можно собрать в другую папку, пока работает старая сборка
    build, dist, rel = ROOT / "build_exe", ROOT / "dist_exe", ROOT / out
    for d in (build, dist, rel):
        shutil.rmtree(d, ignore_errors=True)
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--console", "--name", "Miner",
           "--distpath", str(dist), "--workpath", str(build), "--specpath", str(build), "--paths", str(ROOT),
           "--collect-all", "cupy", "--collect-all", "cupy_backends", "--collect-all", "cuda",
           "--collect-all", "pearl_mining", "--collect-submodules", "blake3",
           "--hidden-import", "graphlib", "--hidden-import", "selftest",
           "--exclude-module", "torch", "--exclude-module", "matplotlib", "--exclude-module", "scipy",
           "--exclude-module", "pandas", "--exclude-module", "tkinter", "--exclude-module", "pytest",
           "--exclude-module", "IPython"]
    for dll in NEEDED:
        cmd += ["--add-binary", f"{NVBIN / dll};bin"]
    cmd += ["--add-data", f"{SITE / 'nvidia' / 'cu13' / 'include'};include"]       # заголовки CUDA для NVRTC (CuPy)
    cmd.append(str(ROOT / "miner.py"))
    subprocess.run(cmd, check=True, cwd=ROOT)

    app = dist / "Miner"
    internal = app / "_internal"
    # убрать лишние CUDA-библиотеки (экономия > 1 ГБ)
    for f in list(internal.rglob("*.dll")):
        if f.name.lower().startswith(UNUSED_PREFIXES):
            f.unlink()
    shutil.rmtree(internal / "nvidia", ignore_errors=True)
    # заголовки cub/thrust и xsf для встроенных ядер CuPy; наши ядра на чистом CUDA C++ их не используют (-19 МБ)
    for sub in ("_cccl", "xsf"):
        shutil.rmtree(internal / "cupy" / "_core" / "include" / "cupy" / sub, ignore_errors=True)       # дубликаты: нужные DLL лежат рядом с exe

    pkg = rel / "PearlMiner"
    shutil.copytree(app, pkg, dirs_exist_ok=True)      # папка может быть занята проводником и не удалиться
    shutil.copy(ROOT / "dist_files" / "start_miner.bat", pkg / "start_miner.bat")
    shutil.copy(ROOT / "dist_files" / "README.txt", pkg / "README.txt")
    zpath = rel / "PearlMiner.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in pkg.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(rel))
    size = lambda p: sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1e6      # noqa: E731
    print(f"\nГотово: {pkg} ({size(pkg):.0f} МБ), {zpath} ({zpath.stat().st_size / 1e6:.0f} МБ)")


if __name__ == "__main__":
    main()
