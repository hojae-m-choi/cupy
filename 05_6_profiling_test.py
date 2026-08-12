import cupy as cp
import time
from cupyx.profiler import time_range
from cupy.cuda import profiler

print("--- NVTX 프로파일링 테스트 시작 ---")

# 1.  데이터 준비
x = cp.random.random(20_000_000, dtype=cp.float32)

# 2. nsys 프로파일러 수집 시작 명령 
profiler.start()

# [테스트 1] 컨텍스트 매니저 방식
print("Step 1: Context Manager 테스트 중...")
with time_range('MY_CONTEXT_RANGE', color_id=1):
    for _ in range(20):
        y = (cp.sin(x) + 1).sum()
    cp.cuda.Device().synchronize() # NVTX 영역이 닫히기 전 GPU 연산 보장

time.sleep(0.5) # 타임라인 구분용 공백 시간

# [테스트 2] 데코레이터 방식
@time_range('MY_DECORATOR_RANGE', color_id=3)
def run_stage(data):
    for _ in range(20):
        data = (cp.cos(data) ** 2).sum()
    return data.sum()

print("Step 2: Decorator 테스트 중...")
res = run_stage(x)
cp.cuda.Device().synchronize()

# 3. 프로파일러 수집 종료 명령
profiler.stop()

print("--- 테스트 완료! 결과를 nsys-rep 파일로 저장합니다. ---")
