OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
GCC ?= g++
NV_VER = /usr/local/cuda-6.5
NVC = $(NV_VER)/bin/nvcc  -ccbin $(GCC)

OP_FLAG      = -O2
NVC_FLAGS   :=
NVC_FLAGS   += $(OP_FLAG)
NVC_FLAGS   += -m${OS_SIZE} -use_fast_math #-Xptxas -v 

CUDA_INC = -I$(NV_VER)/include/ -I$(NV_VER)/samples/common/inc

ifeq ($(OS_SIZE), 64)
  CUDA_LIB = -L$(NV_VER)/lib64
else
  CUDA_LIB = -L$(NV_VER)/lib
endif

LINKLIB = -lcudart 

# CUDA code generation flags
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM30) $(GENCODE_SM35)

all: build

build: fiber 

fiber: fiber.cu 
	$(NVC) $(NVC_FLAGS) $(CUDA_INC) $(CUDA_LIB) $(GENCODE_FLAGS) -o $@ $^ $(LINKLIB)

run: build
	./fiber

.PHONY: clean
clean:
	rm -rf fiber 

