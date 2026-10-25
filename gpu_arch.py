"""gpu_arch.py — архитектура GPU для NVRTC (вместо жёстко зашитой sm_86)."""

MIN_CC = 80      # нужны mma.sync m16n8k32 для int8 и cp.async: Ampere (RTX 30xx) и новее


def nvrtc_arch() -> str:
    import cupy as cp
    cc = int(cp.cuda.Device().compute_capability)
    if cc < MIN_CC:
        raise RuntimeError(f"Видеокарта с compute capability {cc / 10:.1f} не поддерживается: "
                           f"нужна архитектура Ampere (RTX 30xx) или новее (>= 8.0).")
    return f"-arch=sm_{cc}"
