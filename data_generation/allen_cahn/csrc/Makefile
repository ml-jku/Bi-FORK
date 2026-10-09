# Builds the fast Allen-Cahn solver backends.
#   make          -> CPU (always) + CUDA (if nvcc is available)
#   make cpu      -> build/liballencahn_cpu.so
#   make cuda     -> build/liballencahn_cuda.so

BUILD    := build
NVCC     := $(shell command -v nvcc 2>/dev/null)
CXXFLAGS := -O3 -march=native -fopenmp -fPIC -shared -Wall
CUFLAGS  := -O3 -Xcompiler -fPIC -shared

ifdef NVCC
all: cpu cuda
else
all: cpu
	@echo "nvcc not found - skipping CUDA backend"
endif

cpu: $(BUILD)/liballencahn_cpu.so
cuda: $(BUILD)/liballencahn_cuda.so

$(BUILD)/liballencahn_cpu.so: allencahn_cpu.cpp | $(BUILD)
	g++ $(CXXFLAGS) $< -o $@ -lfftw3 -lm

$(BUILD)/liballencahn_cuda.so: allencahn_cuda.cu | $(BUILD)
	nvcc $(CUFLAGS) $< -o $@ -lcufft

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all cpu cuda clean
