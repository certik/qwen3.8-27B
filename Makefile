# Pure-C CPU reference for dense Qwen3.8-27B (and Qwen3.6-35B-A3B MoE), MLX 8-bit.
#
# Pure-C CPU reference (default target):
#   make                       # builds ./qwen35-cpu
#   make run                   # runs the MoE CPU reference
#   make tests                 # model-free CPU precision checks
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

.PHONY: all clean tests run tokenizer qwen35-cpu-check

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

clean:
	rm -f src-cpu/*.o src-cpu/utils/*.o src-cpu/tests/*.o $(TEST_BINS) \
	      src-cpu/tests/test_kernels src-cpu/tests/test_precision_replay \
	      $(BIN) qwen35-cpu-check $(BUILD_TOK)
