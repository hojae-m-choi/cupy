# CuPy 2일 집중 튜토리얼 — 교재 설계 & 구성 (제작 완료)

> 근거: `cupy_syllabus.pdf`(강의계획서), `gtc/`(2026 GTC 1-day 실자료 + 슬라이드), CuPy 공식 문서.
> 방침: ① 이론은 **인라인 마크다운(GTC식)** ② 연습은 **인라인 접기 해답** ③ 측정은 `cupyx.profiler.benchmark` 통일
> ④ **커널은 GTC 우선(Numba CUDA 중심)** — RawKernel은 심화로 분리, 도해는 **자체 제작(NVIDIA 무의존)**.

---

## 1. 설계 원칙

1. **실습 70% / 이론 30%** — 이론은 섹션 도입 마크다운으로 녹이고, 무게중심은 실행 셀·연습.
2. **포팅 중심 서사**: "NumPy/SciPy → `cp.` 치환 → 함정(전송·동기화) → 최적화". 05·06은 GTC식 **Power Iteration** 예제로 연결.
3. **자기완결형 노트북**: 학습목표 + 목차 + 이론 + 예제 + 연습(접기 해답) + 체크포인트.
4. **정확성·성능 동시 검증**: `course_utils.allclose`(CPU 기준)·`bench`/`compare`(성능).
5. **개념 연결(Day 2)**: `07`에서 CUDA 커널 개념을 세우고 `08~11`이 각 개념을 어떻게 구현하는지 연결.
6. **환경 견고성**: `cuda-cccl`·`torch`·`ncu` 등 외부 의존은 **미설치 시 안내 후 건너뛰기**.

---

## 2. 2일 시간표

**공식 14H = 9개 교육 단원** (강의계획서 6개 + 확장 3개). Day 1 / Day 2 각 **7.0H**(공식 배분).

| Day | 교육 단원 | 시간 | 노트북 |
|-----|-----------|------|--------|
| **Day 1** | **1. GPU 컴퓨팅과 CuPy 개론** (계획서 ①) | 1.5H | `00_intro_env`, `01_benchmark_basics` |
| | **2. NumPy/SciPy CuPy 프로그래밍** (계획서 ②) | 3.0H | `02_ndarray_core`, `03_numpy_routines`, `04_scipy_routines` |
| | **3. 메모리 관리 및 성능측정** (계획서 ③) | 1.5H | `05_memory_profiling` |
| | **4. 스트림과 비동기 처리** (계획서 ⑤) | 1.0H | `06_streams_async` |
| | (Day 1 마무리) 통합 캡스톤 | — | `day1_capstone` |
| **Day 2** | **5. CUDA 커널 개념 + CuPy 고수준 커널** (계획서 ④) | 1.0H | `07_cupy_kernels` |
| | **6. Numba CUDA 커널 작성** (GTC 06·07) | 2.5H | `08_numba_copy`, `09_numba_histogram` |
| | **7. 병렬 알고리즘 cuda-cccl** (GTC 05) | 1.0H | `10_cccl` |
| | **8. RawKernel & 커널 최적화 기술** (심화) | 1.0H | `11_rawkernel` |
| | **9. 실전 예제** (계획서 ⑥) | 1.5H | `12_interop_frameworks`, `13_dl_preprocess_capstone` |

**권장 진행 순서**
- Day 1: `00 → 01 → 02 → 03 → 04 → 05 → 06 → day1_capstone`
- Day 2: `07 → 08 → 09 → 10 → 11 → 12 → 13`

> **분량 메모**: 공식 시간표는 14H(7/7)지만, 각 노트북에 연습·예측측정 실험·미니앱을 풍부히 담아(실제 핵심+선택 합산 시 하루 ~8–9h 분량) **강사 재량으로 선택(접기/심화) 항목을 조절**해 페이스를 맞춥니다.

**시간·구조 배분 근거**
- **메모리(③) + 비동기(⑤)를 Day 1 말미에 연속 배치**: "성능을 위한 데이터 이동/오버랩"이라는 한 맥락 — GTC ch03 Memory → ch04 Asynchrony 흐름.
- **Day 2 커널은 GTC 우선(Numba 중심)**: GTC 06·07이 Numba CUDA + Nsight Compute로 커널을 가르치고, 강의계획서 차별성 *"CUDA C 없이 Python만"* 과도 부합 → **Numba를 주축**, RawKernel(CUDA C)은 **심화(`11`)** 로 분리.
- **커널 개념을 `07`에 선집중**: 실행모델·인덱싱·SIMT·메모리계층·coalescing·atomic·occupancy를 먼저 세우고 08~11로 구현 연결.

---

## 3. 공통 규약

### 3.1 셀 구성
헤더(목표) → 목차 → 공통 셋업(`from course_utils import ...; print_env()`) → 이론/예제 반복 → 연습(TODO) + 접기 해답 → 체크포인트.

### 3.2 인라인 해답(접기)
연습은 `TODO` 코드 셀로 제시하고 바로 다음 마크다운에 `<details><summary>💡 해답 보기</summary>` 코드블록으로 정답 제공.

### 3.3 `course_utils.py` (점진 확장, 노트북별 필요 함수만 추가)
- `bytes_human`, `print_env` (00)
- `bench`(=`cupyx.profiler.benchmark` 래퍼), `gpu_ms`, `cpu_ms` (00)
- `print_bench`, `compare`(CPU/GPU wall-clock 비교+speedup) (01)
- `allclose`(cupy→numpy 자동 변환 후 `np.testing.assert_allclose`) (03)

> 측정은 전부 `bench`/`compare`로 통일. NVTX·프로파일은 노트북에서 `cupyx.profiler.time_range`/`profile` 직접 사용.

### 3.4 도해
모든 개념 도해는 **자체 제작**(matplotlib, 중립 팔레트)으로 `images/figures/new_*.png`에 저장(NVIDIA 로고·자료 무의존).

---

## 4. 노트북별 상세

### Day 1

#### `00_intro_env` — GPU 컴퓨팅 & CuPy 개론 (단원 1, 1.5H)
- 이론: CPU vs GPU(지연/처리량·대역폭·SIMT), 프로그래밍 모델(host/device·grid/block/thread), CuPy 소개·차이점·**루틴 지도**, 현재 디바이스/스트림, 일회성 오버헤드.
- 실습: 환경 점검, 첫 GPU 연산, 비동기 타이밍 함정, 전송 비용, NumPy→CuPy 포팅 연습.

#### `01_benchmark_basics` — NumPy vs CuPy 벤치마크 (단원 1)
- 이론: 올바른 벤치마킹(워밍업/동기화/반복), wall-clock vs kernel, 손익분기·연산 강도, `%gpu_timeit`.
- 실습: 크기별 비교, 전송 e2e, 벡터화, dtype·matmul·손익분기 연습.

#### `02_ndarray_core` — ndarray 핵심 & 포팅 (단원 2, 레퍼런스)
- ndarray 구조(strides)·뷰/복사·축·브로드캐스팅, 장치 비종속 코드(`get_array_module`)·NEP18 디스패치·암묵 전송·디바이스 관리. 연습: 열 정규화·뷰 수정·`transform` 포팅.

#### `03_numpy_routines` — NumPy 루틴 (단원 2)
- 모듈 함수(`cupy.*`)·`linalg`(+**PCA**)·`fft`(+**2D 디노이즈**)·`random`(+**몬테카를로**)·**`@cupy.fuse`**. 연습: Top-k·이동평균·lstsq·fuse.

#### `04_scipy_routines` — SciPy 루틴 (단원 2)
- `scipy.fft`(DCT 압축)·`linalg`(lu/expm)·`ndimage`(연결요소 라벨링)·**`sparse`**(포맷·CSR 내부·2D 라플라시안 SpMV)·`sparse.linalg`(2D 푸아송 cg/spsolve/eigsh)·`signal`·`special`/`stats`. 연습: 그래디언트·1D 라플라시안+CG·butter 필터·안정 softmax.

#### `05_memory_profiling` — 메모리 관리 & 성능측정 (단원 3, 1.5H)
- 메모리풀(device/pinned)·캐싱·한도(`CUPY_GPU_MEMORY_LIMIT`)·할당자, `out=` temporary 감소, 전송 병목, `benchmark`/`time_range`/`profile`, **CUB/`CUPY_ACCELERATORS`**, 실전 Power Iteration. 연습: 파이프라인 2배 개선.

#### `06_streams_async` — 스트림과 비동기 처리 (단원 4, 1.0H, Day 1 마무리)
- 스트림·이벤트, 다중 스트림+`wait_event`, 비동기 전송/pinned, **이중 버퍼 청크 오버랩**, 스트림 스케일링, NVTX+Power Iteration, (심화) CUDA Graph·멀티GPU. 연습: 이중 버퍼 직접 구현.

#### `day1_capstone` — Day 1 통합 캡스톤(심화)
- 신호 배치 전처리·특징추출 파이프라인을 **v0(naive)→v1(fuse·메모리)→v2(스트림 오버랩)** 로 점진 최적화 + 정확성 검증 + 이벤트 프로파일.

### Day 2

#### `07_cupy_kernels` — CUDA 커널 개념 + CuPy 고수준 커널 (단원 5, 1.0H)
- **A. CUDA 커널 개념(충실)**: 실행모델·스레드 인덱싱·SIMT/warp·메모리계층·coalescing·atomic/동기화·occupancy + **개념→노트북(07~11) 매핑표**.
- **B. CuPy 커널**: `ElementwiseKernel`(clamp/LeakyReLU)·`ReductionKernel`(L1/제곱합)·`@cupy.fuse` 관계.

#### `08_numba_copy` — Numba CUDA ①: Copy & coalescing (단원 6, GTC 06)
- `@cuda.jit`·`cuda.grid`·인덱싱. **blocked vs coalesced(grid-stride)** 메모리 접근 비교, occupancy 파라미터 스윕. (선택) `%%writefile`+`ncu`+`nsightful` 프로파일.

#### `09_numba_histogram` — Numba CUDA ②: 히스토그램 (단원 6, GTC 07)
- **데이터 레이스 → `cuda.atomic.add` → 공유메모리(`cuda.shared.array`)+`syncthreads`** 단계 최적화. 데이터는 자체 생성 난수 바이트(GTC 원본은 책 텍스트). (선택) cooperative load·ncu.

#### `10_cccl` — 병렬 알고리즘 cuda-cccl (단원 7, GTC 05)
- `cuda.compute`의 `reduce_into`(내장 `OpKind`·**커스텀 이항연산**)·`unary_transform`·**Iterators**(Counting/Transform). 언제 cccl vs Numba vs CuPy. **환경 주의**: `cuda-cccl`(실험적) 설치·미설치 안내.

#### `11_rawkernel` — RawKernel & 커널 최적화 기술 (단원 8, 심화)
- `RawKernel`(CUDA C) saxpy·2D 스텐실·블록 튜닝, **커널 최적화 기술표**(coalescing·타일링·분기 최소화·`__restrict__`·occupancy·융합), **공유메모리 블록 리덕션**. 연습: clamp RawKernel. (07 개념을 CUDA C로 직접 구현)

#### `12_interop_frameworks` — 프레임워크 통합 (단원 9)
- DLPack(`from_dlpack`)·`__cuda_array_interface__`, CuPy↔PyTorch **무복사(zero-copy)** 변환·포인터 공유, 소유권·동기화 주의. 연습: CuPy 전처리→torch→CuPy 무복사 연결. torch 미설치 시 graceful skip.

#### `13_dl_preprocess_capstone` — DL 전처리 캡스톤 (단원 9, 종합)
- 표준화→**커스텀 커널 비선형(Elementwise/fuse)**→특징→v0/v1(fuse)/v2(스트림) 최적화→**DLPack 무복사 PyTorch 입력**. 정확성 검증·성능 비교·도전 과제.

---

## 5. 도구·환경

| 항목 | 권장 |
|------|------|
| CuPy | `cupy-cuda12x`/`cupy-cuda11x` 최신 안정판(14.x) |
| 기반 | numpy, scipy, matplotlib |
| 커널 | `numba`(CUDA) — `08·09·13` |
| 병렬 알고리즘 | `cuda-cccl`(실험적, `cuda.compute`) — `10` |
| 상호운용 | `torch`(CUDA, 선택) — `12·13` 무복사 |
| 프로파일 | `cupyx.profiler`(benchmark/time_range/profile), (선택) Nsight Systems/Compute |

> 외부 의존(`cuda-cccl`·`torch`·`ncu`)은 **미설치 시 안내 후 건너뛰기**로 처리.

---

## 6. 산출물

- **노트북 15개**: Day 1 `00~06` + `day1_capstone`, Day 2 `07~13`.
- **`course_utils.py`**: 공통 유틸(점진 확장).
- **`images/figures/new_*.png`**: 자체 제작 개념 도해(16종, NVIDIA 무의존).
- **`slides/slides_*.pptx`**: Day 1 강사용 슬라이드 8덱(각 20~28장, 발표자 노트 포함).
- **`README.md`**: 2일 구성·실행법·패키지.
- `previous/`: 원본 초안 노트북(보존).

---

## 7. 남은 작업 / 검증

1. **Day 2 강사용 슬라이드**(`07~13`) — 미제작.
2. **GPU 실행 검증**: 본 세션엔 GPU가 없어 **코드 문법·import 정합성만** 점검 완료. 강의장(CUDA GPU + `numba`/`cuda-cccl`/`torch`/`ncu`)에서 `00`부터 순차 실행해 `assert`·벤치 통과를 확인 권장(가능하면 `nbconvert --execute`).
3. (선택) 강사 진행 가이드(타임테이블·체크리스트).
