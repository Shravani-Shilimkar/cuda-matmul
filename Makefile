# ─────────────────────────────────────────────
# Makefile — CUDA Matrix Multiplication
# ─────────────────────────────────────────────

NVCC       = nvcc
TARGET     = matmul
SRC        = matmul.cu

# Adjust -arch for your GPU:
#   sm_75  → Turing  (RTX 2000 series)
#   sm_80  → Ampere  (A100, RTX 3000 series)
#   sm_86  → Ampere  (RTX 3080/3090)
#   sm_89  → Ada     (RTX 4000 series)
ARCH       = sm_80

NVCC_FLAGS = -O3 -arch=$(ARCH) -lineinfo \
             --generate-line-info \
             -Xptxas -v            # print register/shared mem usage

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run with default N=1024
run: $(TARGET)
	./$(TARGET) 1024

# Run with larger matrix for more meaningful bandwidth numbers
bench: $(TARGET)
	./$(TARGET) 4096

# Nsight Compute — kernel metrics
profile: $(TARGET)
	ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
	./$(TARGET) 2048

# Nsight Systems — timeline
nsys: $(TARGET)
	nsys profile --stats=true ./$(TARGET) 2048

clean:
	rm -f $(TARGET)

.PHONY: all run bench profile nsys clean
