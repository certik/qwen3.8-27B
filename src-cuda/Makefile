# Build: make            (needs nvcc and a C compiler; targets sm_90 = H100/H200)
NVCC ?= nvcc
CC ?= gcc
ARCH ?= sm_90
CFLAGS = -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -D_POSIX_C_SOURCE=200809L
CUDA_HOME ?= $(dir $(shell which $(NVCC)))..
CPPFLAGS = -I$(CUDA_HOME)/include

OBJ = main.o tokenizer.o json.o safetensors.o kernels.o

qwen: $(OBJ)
	$(NVCC) -arch=$(ARCH) -o $@ $(OBJ)

%.o: %.c model.h json.h safetensors.h tokenizer.h unicode_tables.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

kernels.o: kernels.cu model.h
	$(NVCC) -O3 -arch=$(ARCH) -std=c++17 -c $< -o $@

clean:
	rm -f qwen $(OBJ)

.PHONY: clean
