# Change this to point to wherever CUDA is installed on your local system.
CUDA_PATH := /opt/net/apps/cuda
NVCC=${CUDA_PATH}/bin/nvcc
CXX=g++

CC_FLAGS := -I../ -I${CUDA_PATH}/include

NVCC_FLAGS := ${CC_FLAGS} -arch=sm_21 
ARCH := $(shell getconf LONG_BIT)

LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64

LIB_FLAGS := $(LIB_FLAGS_$(ARCH)) -lcudart

.PHONY: math clean
all: math

math: math_host_g++ math_host_nvcc math_device 
	for prog in $^; do \
		./$$prog; \
	done

math_device: math.cc
	${NVCC} ${LIB_FLAGS} -x cu ${CC_FLAGS} ${NVCC_FLAGS} $< -o $@

math_host_g++: math.cc
	${CXX} ${LIB_FLAGS} ${CC_FLAGS} $< -o $@

math_host_nvcc: math.cc
	${NVCC} ${LIB_FLAGS} ${CC_FLAGS} ${NVCC_FLAGS} $< -o $@
