# Pure-C CPU reference for dense Qwen3.8-27B (and Qwen3.6-35B-A3B MoE), MLX 8-bit.
#
# Pure-C CPU reference (default target):
#   make                       # builds ./qwen35-cpu
#   make run                   # runs the MoE CPU reference
#   make tests                 # model-free CPU precision checks
#
# Fortran implementation (src-fortran):
#   make fortran-gpu           # builds ./qwen-fortran-cuda with LFortran (--gpu=cuda)
#   make fortran-tokenizer-test MODEL_DIR=...   # tokenizer vs HuggingFace tokenizers
#
# CUDA implementation (src-cuda):
#   make cuda                  # builds src-cuda/qwen with nvcc
#
# Common:
#   make clean

CC      ?= clang
CFLAGS  := -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -fno-fast-math -ffp-contract=off -D_POSIX_C_SOURCE=200809L -Isrc-cpu
# The reference numerics assume no auto-vectorization (clang flags).
ifneq ($(findstring clang,$(shell $(CC) --version 2>/dev/null)),)
CFLAGS  += -fno-vectorize -fno-slp-vectorize
else
CFLAGS  += -fno-tree-vectorize
endif
LDFLAGS := -lm

BIN     := qwen35-cpu

SRCS    := src-cpu/main.c \
           src-cpu/kernels.c \
           src-cpu/safetensors.c \
           src-cpu/tokenizer.c \
           src-cpu/qwen_chat.c \
           src-cpu/utils/json.c \
           src-cpu/utils/iofile.c \
           src-cpu/utils/bytebuf.c \
           src-cpu/utils/utf8.c

OBJS    := $(SRCS:.c=.o)
UTIL_OBJS := src-cpu/utils/json.o src-cpu/utils/iofile.o \
             src-cpu/utils/bytebuf.o src-cpu/utils/utf8.o
UTIL_HDRS := src-cpu/utils/json.h src-cpu/utils/jsmn.h \
             src-cpu/utils/iofile.h src-cpu/utils/bytebuf.h \
             src-cpu/utils/utf8.h src-cpu/utils/bf16.h

TEST_BINS := src-cpu/tests/test_validation

BUILD_TOK := src-cpu/build_tokenizer

.PHONY: all clean tests run tokenizer qwen35-cpu-check fortran-gpu fortran-tokenizer-test cuda

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

src-cpu/%.o: src-cpu/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# tokenizer.bin is a build prerequisite for `run` (not for the binary).
tokenizer: src-cpu/tokenizer.bin

$(BUILD_TOK): src-cpu/build_tokenizer.c $(UTIL_OBJS) $(UTIL_HDRS)
	$(CC) $(CFLAGS) src-cpu/build_tokenizer.c $(UTIL_OBJS) -o $@ $(LDFLAGS)

src-cpu/tokenizer.bin: $(BUILD_TOK)
	./$(BUILD_TOK) --out src-cpu/tokenizer.bin

tests: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "==> $$t" && ./$$t || exit 1; done

src-cpu/tests/test_%: src-cpu/tests/test_%.c src-cpu/kernels.o src-cpu/safetensors.o src-cpu/tokenizer.o src-cpu/qwen_chat.o $(UTIL_OBJS)
	$(CC) $(CFLAGS) $(filter %.c %.o,$^) -o $@ $(LDFLAGS)

src-cpu/tests/test_kernels src-cpu/tests/test_validation src-cpu/tests/test_precision_replay: src-cpu/tests/load_ref.h src-cpu/tests/compare.h

qwen35-cpu-check: $(SRCS) $(UTIL_HDRS) src-cpu/kernels.h src-cpu/safetensors.h
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(SRCS) -o $@ $(LDFLAGS)

# Directory of the MLX 8-bit checkpoint (safetensors + config.json + tokenizer.json).
MODEL_DIR ?= /path/to/model

run: $(BIN) src-cpu/tokenizer.bin
	./$(BIN) --prompt "Hello, who are you?" --max-tokens 8 --model $(MODEL_DIR)

# ---------- Fortran implementation ------------------------------------------

FSRCS := src-fortran/model_io_support.f90 src-fortran/json_tokens.f90 \
         src-fortran/safetensors_io.f90 src-fortran/qwen_config.f90 \
         src-fortran/qwen_tokenizer.f90 src-fortran/kernels.f90 src-fortran/qwen.f90

# LFortran (recent main) offloads the do concurrent loops to the GPU.
# The linker defaults to clang; use LFORTRAN_LINKER=gcc where there is none.
fortran-gpu: qwen-fortran-cuda
qwen-fortran-cuda: $(FSRCS)
	mkdir -p build/fortran
	cd build/fortran && lfortran --fast --realloc-lhs-arrays --gpu=cuda --gpu-overlap \
	    $(addprefix $(CURDIR)/,$(FSRCS)) -o $(CURDIR)/$@
	rm -rf build/fortran

# The tokenizer test needs only the model's tokenizer files; with `pip install tokenizers` it also
# compares random texts against HuggingFace tokenizers.
fortran-tokenizer-test:
	mkdir -p build/fortran-tokenizer/lfortran
	cd build/fortran-tokenizer/lfortran && lfortran \
	    $(addprefix $(CURDIR)/,src-fortran/qwen_tokenizer.f90 src-fortran/tests/test_tokenizer.f90) -o test_tokenizer
	python3 src-fortran/tests/test_tokenizer.py --model $(MODEL_DIR) --binary build/fortran-tokenizer/lfortran/test_tokenizer
	rm -rf build/fortran-tokenizer/lfortran

cuda:
	$(MAKE) -C src-cuda

clean:
	$(MAKE) -C src-cuda clean
	rm -f src-cpu/*.o src-cpu/utils/*.o src-cpu/tests/*.o $(TEST_BINS) \
	      src-cpu/tests/test_kernels src-cpu/tests/test_precision_replay \
	      $(BIN) qwen35-cpu-check $(BUILD_TOK) \
	      qwen-fortran-cuda
	rm -rf build/fortran build/fortran-tokenizer
