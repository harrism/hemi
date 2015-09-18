# operating system
HOST_OS := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

# architecture 
ARCH := $(shell getconf LONG_BIT)

NVCC := nvcc

ifeq ($(HOST_OS),darwin)
	CXX := clang++
else
	CXX := g++
endif

STD := --std=c++11

CXX_FLAGS  := $(STD) -I../ -I../../

HOST_ONLY_FLAGS := -DHEMI_CUDA_DISABLE

# uncomment for debug
#DEBUG_FLAGS := -g -DDEBUG
#DEBUG_FLAGS_NVCC := -G

# comment for debug
CXX_FLAGS += -O3

CXX_FLAGS += $(DEBUG_FLAGS)

NVCC_FLAGS := $(CXX_FLAGS) $(DEBUG_FLAGS_NVCC) --expt-extended-lambda

all: parallel_for_device parallel_for_host parallel_for_host_nvcc

parallel_for_device: parallel_for.cpp
	$(NVCC) $? $(NVCC_FLAGS) -x cu -o $@

parallel_for_host_nvcc: parallel_for.cpp
	$(NVCC) $? $(CXX_FLAGS) -x c++ -o $@

parallel_for_host: parallel_for.cpp
	$(CXX) $? $(CXX_FLAGS) $(HOST_ONLY_FLAGS) -o $@

clean:
	rm -rf parallel_for_device parallel_for_host_nvcc parallel_for_host
