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


def bytes_human(n: int) -> str:
    """바이트 수를 KB/MB/GB 등 사람이 읽기 쉬운 문자열로 변환.

    예: bytes_human(1536) -> "1.50 KB"
    예: bytes_human(2**30) -> "1.00 GB"
    """
    # 1024배씩 커지는 단위 목록. GPU/OS 메모리 표기 관례에 맞춰 1000이 아닌 1024(2^10) 기준.
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)  # 정수 나눗셈으로 인한 반올림 손실을 피하기 위해 float로 변환
    for s in suffixes:
        # x가 1024 미만이면 더 이상 올릴 단위가 없으므로 현재 단위(s)로 확정해 반환
        # (또는 마지막 단위인 TB까지 왔다면 그 이상은 없으므로 무조건 반환)
        if x < 1024.0 or s == suffixes[-1]:
            return f"{x:.2f} {s}"
        # 아직 1024 이상이면 한 단계 큰 단위로 넘어가기 위해 1024로 나눔 (B->KB->MB->GB->TB)
        x /= 1024.0
    # 방어적 코드: 위 루프는 suffixes[-1]("TB")에서 항상 return하므로 실제로는 도달하지 않음
    return f"{x:.2f} TB"


def print_env() -> None:
    """실습 환경 요약 출력: numpy/cupy 버전, GPU 이름, VRAM."""
    print("=== Environment ===")
    print("numpy:", np.__version__)  # numpy는 필수 의존성이므로 항상 설치되어 있다고 가정
    if cp is None:
        # 이 모듈 상단의 `try: import cupy as cp / except: cp = None`에 의해
        # CuPy 미설치·미가용 환경에서는 cp가 None이 됨 -> 여기서 안내만 하고 조용히 종료
        print("cupy : (사용 불가) — cupy-cuda12x 또는 cupy-cuda11x 설치 필요")
        print("===================")
        return  # GPU가 없으므로 아래 디바이스 조회 로직은 의미가 없어 함수 종료
    print("cupy :", cp.__version__)
    try:
        dev = cp.cuda.Device()  # 현재 활성(디폴트) CUDA 디바이스 핸들(보통 0번)
        # CUDA 런타임 API로 하드웨어 속성(이름, compute capability, 총 VRAM 등)을 조회
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"]
        # CUDA 버전/플랫폼에 따라 이름이 bytes로 오는 경우가 있어 문자열로 디코딩 처리
        name = name.decode() if isinstance(name, (bytes, bytearray)) else name
        # compute capability: GPU 아키텍처 세대를 나타내는 버전(예: 8.0 = Ampere, 9.0 = Hopper)
        cc = f"{props['major']}.{props['minor']}"
        total_mem = int(props.get("totalGlobalMem", 0))  # GPU 전체 VRAM 용량(바이트)
        print("device:", f"{dev.id} - {name} (compute capability {cc})")
        print("VRAM  :", bytes_human(total_mem))  # 위에서 정의한 bytes_human으로 가독성 좋게 변환
    except Exception as e:
        # 드라이버 미설치/권한 문제 등으로 조회가 실패해도 노트북 실행이 멈추지 않도록 예외 처리
        print("device: (조회 실패)", e)
    print("===================")


def bench(fn, *, n_repeat: int = 20, n_warmup: int = 3, name: str | None = None):
    """올바른 GPU 타이밍 헬퍼.

    GPU 연산은 비동기라서 time.perf_counter()로는 정확히 잴 수 없습니다.
    cupyx.profiler.benchmark는 CUDA 이벤트로 동기화하여 CPU/GPU 시간을 함께 측정합니다.

    Parameters
    ----------
    fn : Callable
        인자 없이 호출 가능한 벤치마크 대상 함수 (예: `lambda: cp.sin(x)`).
    n_repeat : int
        실제로 측정에 반영되는 반복 횟수. 이 반복들의 시간으로 평균/표준편차를 낸다.
    n_warmup : int
        측정에서 제외되는 예열(warm-up) 반복 횟수. CUDA 컨텍스트 초기화, 커널
        최초 컴파일/캐시 적재 등 '일회성 오버헤드'가 측정값을 왜곡하지 않도록 미리 소모한다.
    name : str, optional
        결과에 표시할 이름. 지정하지 않으면 fn.__name__을 사용(람다는 이름이 없어 "fn"이 됨).

    Returns
    -------
    cupyx.profiler._time._PerfCaseResult
        gpu_times / cpu_times (각 n_repeat 길이의 배열, 단위: 초)를 담고 있으며,
        이 모듈의 gpu_ms()/cpu_ms()로 평균값(ms)을 뽑아 쓴다.
    """
    if cp is None:
        # CuPy가 없으면 GPU 벤치마크 자체가 불가능하므로 조용히 넘어가지 않고 명확히 에러 발생
        raise RuntimeError("CuPy를 사용할 수 없습니다. cupy-cuda12x/11x를 설치하세요.")
    # benchmark는 각 반복마다 CUDA 이벤트를 커널 앞뒤로 기록하고, 이벤트를 동기화(sync)해
    # '커널이 실제로 끝난 시점'까지 기다린 뒤 시간을 재기 때문에 비동기 문제 없이 정확하다.
    from cupyx.profiler import benchmark
    return benchmark(
        fn, (),                                        # fn을 인자 없이(()) 반복 호출
        n_repeat=n_repeat,
        n_warmup=n_warmup,
        name=name or getattr(fn, "__name__", "fn"),     # 이름 미지정 시 함수명 사용
    )


def gpu_ms(result) -> float:
    """bench() 결과에서 GPU 평균 시간(ms).

    result.gpu_times: 각 반복에서 GPU가 실제로 커널을 실행한 시간(단위: 초)의 배열.
    CPU가 커널을 GPU에 '제출'만 하고 바로 다음 줄로 넘어가는 launch overhead는
    포함하지 않으므로, 순수 커널 연산 시간(GPU 관점)을 보고 싶을 때 사용한다.
    """
    # 여러 반복(n_repeat)의 평균을 낸 뒤, 초(s) 단위를 밀리초(ms)로 변환(x1e3)
    return float(np.asarray(result.gpu_times).mean()) * 1e3


def cpu_ms(result) -> float:
    """bench() 결과에서 CPU(런치 포함) 평균 시간(ms).

    result.cpu_times: 각 반복에서 '호출 시작~반환'까지 CPU 입장에서 흐른
    wall-clock 시간(단위: 초)의 배열. 커널 launch overhead와 파이썬 오버헤드,
    (필요 시) 동기화 대기까지 포함하므로 사용자가 체감하는 end-to-end 시간에 더 가깝다.
    """
    return float(np.asarray(result.cpu_times).mean()) * 1e3


def print_bench(result) -> None:
    """bench() 결과를 한 줄로 보기 좋게 출력 (CPU wall-clock / GPU 커널)."""
    c = cpu_ms(result)  # CPU 관점 wall-clock 평균(ms): launch overhead·파이썬 오버헤드 포함
    g = gpu_ms(result)  # GPU 관점 커널 실행 평균(ms): 순수 커널 실행 시간만
    # result.name: bench() 호출 시 지정한 name(또는 함수명). >20/9.3f로 폭을 고정해
    # 여러 줄을 연달아 출력해도 표처럼 열이 맞춰져 보이게 함
    print(f"{result.name:>20} | CPU(wall) {c:9.3f} ms | GPU(kernel) {g:9.3f} ms")


def compare(name, cpu_fn, gpu_fn, *, n_repeat: int = 10, n_warmup: int = 2):
    """CPU 함수와 GPU 함수를 같은 조건으로 측정해 wall-clock으로 공정 비교.

    GPU도 CPU(wall-clock) 시간으로 비교합니다(전송·동기화 포함, end-to-end 관점).
    Returns: (cpu_ms, gpu_ms, speedup)
    """
    # 두 함수(cpu_fn/gpu_fn)를 동일한 n_repeat/n_warmup 조건으로 각각 측정.
    # name에 "-cpu"/"-gpu" 접미사를 붙여 어느 쪽 결과인지 구분되게 함
    rc = bench(cpu_fn, n_repeat=n_repeat, n_warmup=n_warmup, name=f"{name}-cpu")
    rg = bench(gpu_fn, n_repeat=n_repeat, n_warmup=n_warmup, name=f"{name}-gpu")
    # 주의: GPU 쪽도 gpu_ms가 아닌 cpu_ms(wall-clock)를 사용한다.
    # 커널 시간(gpu_ms)만 비교하면 host<->device 전송·동기화 오버헤드가 감춰져 불공정한
    # 비교가 되므로, 사용자가 실제 체감하는 end-to-end(호출~반환) 시간으로 맞춰 비교한다.
    c = cpu_ms(rc)
    g = cpu_ms(rg)
    # GPU가 더 빠르면 speedup > 1. g가 0에 가까워 나눗셈이 위험한 경우 무한대로 대체
    speedup = c / g if g > 0 else float("inf")
    print(f"{name:>16} | CPU {c:9.3f} ms | GPU {g:9.3f} ms | {speedup:6.2f}x")
    return c, g, speedup


def allclose(a, b, *, rtol: float = 1e-5, atol: float = 1e-6, name: str = "") -> None:
    """CPU/GPU 결과 정확성 검증. cupy 배열은 자동으로 host로 옮겨 비교한다.

    float32 GPU 연산은 CPU와 bit 단위로 같지 않으므로 rtol/atol을 적절히 키워 사용한다.
    """
    if cp is not None:
        # a, b가 각각 cupy 배열이면 cp.asnumpy()로 device->host 복사해
        # np.testing.assert_allclose가 numpy 배열끼리 비교하도록 맞춘다.
        # (cp가 None인 환경, 즉 CuPy 미설치 시에는 cupy 배열 자체가 존재할 수 없으므로
        #  이 분기를 건너뛰고 바로 아래에서 numpy 배열끼리 비교한다.)
        if isinstance(a, cp.ndarray):
            a = cp.asnumpy(a)
        if isinstance(b, cp.ndarray):
            b = cp.asnumpy(b)
    # rtol(상대 허용오차)/atol(절대 허용오차): |a-b| <= atol + rtol*|b| 를 만족해야 통과.
    # GPU는 리덕션·행렬곱 등에서 연산 순서가 CPU와 달라 부동소수점 마지막 비트가 다를 수
    # 있으므로, 완전 일치(==)가 아니라 허용오차 이내인지를 검사한다.
    # 불일치 시 AssertionError를 발생시키며 err_msg=name으로 어떤 검증이 실패했는지 표시
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)
    # 여기까지 예외 없이 도달했다면 검증 통과 -> 성공 로그 출력
    print(f"[allclose OK] {name}".rstrip())
