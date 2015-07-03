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

STD := -std=c++11

CXX_FLAGS  := $(STD) -I../ -I../../

HOST_ONLY_FLAGS := -DHEMI_CUDA_DISABLE

# uncomment for debug
#DEBUG_FLAGS := -g -DDEBUG
#DEBUG_FLAGS_NVCC := -G

# comment for debug
CXX_FLAGS += -O3

CXX_FLAGS += $(DEBUG_FLAGS)
NVCC_FLAGS := $(CXX_FLAGS) $(DEBUG_FLAGS_NVCC) --expt-extended-lambda

all: bs_device bs_host_nvcc bs_host

bs_device: blackscholes.cpp
	$(NVCC) blackscholes.cpp $(NVCC_FLAGS) -x cu -o bs_device

bs_host_nvcc: blackscholes.cpp
	$(NVCC) blackscholes.cpp $(CXX_FLAGS) $(HOST_ONLY_FLAGS) -x c++ -o bs_host_nvcc

bs_host: blackscholes.cpp
	$(CXX) blackscholes.cpp $(CXX_FLAGS) $(HOST_ONLY_FLAGS) -o bs_host

clean:
	rm -rf bs_device bs_host_nvcc bs_host
