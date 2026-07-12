# CuPy 2일 집중 코스 — 노트북 세트

NumPy/SciPy 사용자를 위한 CuPy(GPU 가속) 실습 교재입니다. 설계 상세는 `course_design.md` 참고.

## 구성 (Day 1)

| 단원 | 노트북 | 내용 |
|------|--------|------|
| 1. 개론 | `00_intro_env` | CPU vs GPU, GPU 프로그래밍 모델, CuPy 소개·루틴 지도, 환경 점검 |
| 1. 개론 | `01_benchmark_basics` | 올바른 벤치마킹, 손익분기점, 전송 비용, 벡터화 |
| 2. NumPy/SciPy | `02_ndarray_core` | ndarray 구조·뷰/복사·브로드캐스팅, NumPy→CuPy 포팅 |
| 2. NumPy/SciPy | `03_numpy_routines` | `cupy.*` 모듈 함수, `linalg`+PCA, `fft`+2D 디노이즈, `random`+몬테카를로, `@cupy.fuse` |
| 2. NumPy/SciPy | `04_scipy_routines` | `scipy.fft`+DCT 압축, `linalg`, `ndimage`+연결요소, `sparse`+2D 푸아송, `signal`, `special`/`stats` |
| 3. 메모리·성능 | `05_memory_profiling` | 메모리풀·temporary(out=)·전송 병목·CUB·프로파일링·Power Iteration |
| 4. 스트림·비동기 | `06_streams_async` | 스트림·이벤트·이중버퍼 오버랩·스트림 스케일링·CUDA Graph·멀티GPU |
| 통합 | `day1_capstone` | 생성→표준화→배치 FFT→linalg 특징추출→벤치마크 종합 실습 |

> Day 1 마무리: `day1_capstone` (v0→v1→v2 통합 최적화 실습)

## 구성 (Day 2)

| 단원 | 노트북 | 내용 |
|------|--------|------|
| 5. CuPy 커널 | `07_cupy_kernels` | **CUDA 커널 개념**(인덱싱·SIMT·메모리·coalescing·atomic·occupancy) + Elementwise/Reduction/fuse |
| 6. Numba CUDA | `08_numba_copy` | `@cuda.jit`·grid/block·**coalescing**·(선택)Nsight Compute |
| 6. Numba CUDA | `09_numba_histogram` | 데이터 레이스→**atomic**→**공유메모리** 단계 최적화 |
| 7. 병렬 알고리즘 | `10_cccl` | `cuda.compute`(cuda-cccl) reduce/transform/iterator |
| 8. 심화 | `11_rawkernel` | **RawKernel(CUDA C)** + 커널 최적화 기술·공유메모리 리덕션 |
| 9. 실전 | `12_interop_frameworks` | DLPack·`__cuda_array_interface__` PyTorch 무복사 연동 |
| 9. 실전 | `13_dl_preprocess_capstone` | 커스텀 커널+interop DL 전처리 종합 캡스톤 |

> 커널 개념은 `07`에서 충실히 세우고, `08~11`이 각 개념을 어떻게 구현하는지 연결됩니다.
> Day 2는 GTC 강의자료(06·07 Numba 커널, 05 cccl)를 우선 반영했습니다. 각 노트북은 연습문제(💡 접기 해답)·예측측정 실험을 포함합니다.

### Day 2 추가 패키지
- `numba`(CUDA) — `08·09·13`
- `cuda-cccl`(실험적, `cuda.compute`) — `10`
- (선택) `torch`(CUDA) — `12·13` 무복사 연동
- (선택) Nsight Compute(`ncu`) — `08·09` 프로파일

보조 파일: `course_utils.py`(공통 유틸리티), `images/slides/`(NVIDIA GTC 강의 도해).

## 실행

1. 이 폴더에서 Jupyter 실행: `jupyter lab`
2. `00`번부터 순서대로 실행
3. 각 노트북의 연습문제는 `💡 해답 보기`(접기)로 정답 확인

## 필요 패키지

- `cupy-cuda12x` 또는 `cupy-cuda11x` (최신 안정판 권장)
- `numpy`, `scipy`, `matplotlib`
- (선택) `torch` — 프레임워크 상호운용(단원 7)
- (선택) `nvtx`, Nsight Systems — 프로파일링 실습

> GPU 실행에는 NVIDIA CUDA GPU가 필요합니다.
