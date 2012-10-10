CUDA_PATH := /usr/local/cuda

CC_FLAGS  := -I../ -I../../ -I$(CUDA_PATH)/include 

# uncomment for debug
#DEBUG_FLAGS := -g -DDEBUG
#DEBUG_FLAGS_NVCC := -G

# comment for debug
CC_FLAGS += -O3

CC_FLAGS += $(DEBUG_FLAGS)
NVCC_FLAGS := $(CC_FLAGS) $(DEBUG_FLAGS_NVCC)

NVCC_FLAGS := $(CC_FLAGS) $(DEBUG_FLAGS_NVCC)

ARCH := $(shell getconf LONG_BIT)

LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64

LIB_FLAGS := $(LIB_FLAGS_$(ARCH)) -lcudart

all: bs_device bs_host_nvcc bs_host_g++

bs_device: blackscholes.cpp
	nvcc blackscholes.cpp $(NVCC_FLAGS) $(LIB_FLAGS) -x cu -o bs_device

bs_host_nvcc: blackscholes.cpp
	nvcc blackscholes.cpp $(CC_FLAGS) $(LIB_FLAGS) -x c++ -o bs_host_nvcc

bs_host_g++: blackscholes.cpp
	g++ blackscholes.cpp $(CC_FLAGS) $(LIB_FLAGS) -o bs_host_g++

clean:
	rm -rf bs_device bs_host_nvcc bs_host_g++
