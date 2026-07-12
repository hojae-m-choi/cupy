"""course_utils.py — CuPy 2일 집중 실습 공통 유틸리티

설계 원칙: 노트북이 진행되며 **필요한 함수만 하나씩 추가**합니다.
(불필요하게 큰 유틸을 처음부터 만들지 않습니다.)

현재 포함 함수 (노트북 00_intro_env 기준)
- bytes_human(n)         : 바이트를 사람이 읽기 쉬운 단위로
- print_env()            : numpy/cupy 버전 + GPU 정보 출력
- bench(fn, ...)         : cupyx.profiler.benchmark 래퍼 (올바른 GPU 타이밍)
- gpu_ms(result)/cpu_ms(result) : bench 결과에서 평균 시간(ms) 추출

추가 이력
- 00_intro_env       : bytes_human, print_env, bench, gpu_ms, cpu_ms
- 01_benchmark_basics: print_bench, compare
- 03_numpy_routines  : allclose
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except Exception:  # CuPy 미설치/미가용 환경에서도 import 자체는 통과
    cp = None  # type: ignore


# ---------------------------------------------------------------------------
# 00_intro_env
# ---------------------------------------------------------------------------
def bytes_human(n: int) -> str:
    """바이트 수를 KB/MB/GB 등 사람이 읽기 쉬운 문자열로 변환."""
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for s in suffixes:
        if x < 1024.0 or s == suffixes[-1]:
            return f"{x:.2f} {s}"
        x /= 1024.0
    return f"{x:.2f} TB"


def print_env() -> None:
    """실습 환경 요약 출력: numpy/cupy 버전, GPU 이름, VRAM."""
    print("=== Environment ===")
    print("numpy:", np.__version__)
    if cp is None:
        print("cupy : (사용 불가) — cupy-cuda12x 또는 cupy-cuda11x 설치 필요")
        print("===================")
        return
    print("cupy :", cp.__version__)
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"]
        name = name.decode() if isinstance(name, (bytes, bytearray)) else name
        cc = f"{props['major']}.{props['minor']}"  # compute capability
        total_mem = int(props.get("totalGlobalMem", 0))
        print("device:", f"{dev.id} - {name} (compute capability {cc})")
        print("VRAM  :", bytes_human(total_mem))
    except Exception as e:  # GPU 쿼리 실패 시에도 죽지 않게
        print("device: (조회 실패)", e)
    print("===================")


def bench(fn, *, n_repeat: int = 20, n_warmup: int = 3, name: str | None = None):
    """올바른 GPU 타이밍 헬퍼.

    GPU 연산은 비동기라서 time.perf_counter()로는 정확히 잴 수 없습니다.
    cupyx.profiler.benchmark는 CUDA 이벤트로 동기화하여 CPU/GPU 시간을 함께 측정합니다.

    Returns: cupyx.profiler._time._PerfCaseResult (gpu_ms/cpu_ms로 평균 추출)
    """
    if cp is None:
        raise RuntimeError("CuPy를 사용할 수 없습니다. cupy-cuda12x/11x를 설치하세요.")
    from cupyx.profiler import benchmark
    return benchmark(
        fn, (), n_repeat=n_repeat, n_warmup=n_warmup,
        name=name or getattr(fn, "__name__", "fn"),
    )


def gpu_ms(result) -> float:
    """bench() 결과에서 GPU 평균 시간(ms)."""
    return float(np.asarray(result.gpu_times).mean()) * 1e3


def cpu_ms(result) -> float:
    """bench() 결과에서 CPU(런치 포함) 평균 시간(ms)."""
    return float(np.asarray(result.cpu_times).mean()) * 1e3


# ---------------------------------------------------------------------------
# 01_benchmark_basics
# ---------------------------------------------------------------------------
def print_bench(result) -> None:
    """bench() 결과를 한 줄로 보기 좋게 출력 (CPU wall-clock / GPU 커널)."""
    c = cpu_ms(result)
    g = gpu_ms(result)
    print(f"{result.name:>20} | CPU(wall) {c:9.3f} ms | GPU(kernel) {g:9.3f} ms")


def compare(name, cpu_fn, gpu_fn, *, n_repeat: int = 10, n_warmup: int = 2):
    """CPU 함수와 GPU 함수를 같은 조건으로 측정해 wall-clock으로 공정 비교.

    GPU도 CPU(wall-clock) 시간으로 비교합니다(전송·동기화 포함, end-to-end 관점).
    Returns: (cpu_ms, gpu_ms, speedup)
    """
    rc = bench(cpu_fn, n_repeat=n_repeat, n_warmup=n_warmup, name=f"{name}-cpu")
    rg = bench(gpu_fn, n_repeat=n_repeat, n_warmup=n_warmup, name=f"{name}-gpu")
    c = cpu_ms(rc)
    g = cpu_ms(rg)
    speedup = c / g if g > 0 else float("inf")
    print(f"{name:>16} | CPU {c:9.3f} ms | GPU {g:9.3f} ms | {speedup:6.2f}x")
    return c, g, speedup


# ---------------------------------------------------------------------------
# 03_linalg_fft
# ---------------------------------------------------------------------------
def allclose(a, b, *, rtol: float = 1e-5, atol: float = 1e-6, name: str = "") -> None:
    """CPU/GPU 결과 정확성 검증. cupy 배열은 자동으로 host로 옮겨 비교한다.

    float32 GPU 연산은 CPU와 bit 단위로 같지 않으므로 rtol/atol을 적절히 키워 사용한다.
    """
    if cp is not None:
        if isinstance(a, cp.ndarray):
            a = cp.asnumpy(a)
        if isinstance(b, cp.ndarray):
            b = cp.asnumpy(b)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)
    print(f"[allclose OK] {name}".rstrip())
