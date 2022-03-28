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

CXX_FLAGS  := $(STD) -I../../

HOST_ONLY_FLAGS := -DHEMI_CUDA_DISABLE

# uncomment for debug
#DEBUG_FLAGS := -g -DDEBUG
#DEBUG_FLAGS_NVCC := -G

# comment for debug
CXX_FLAGS += -O3

CXX_FLAGS += $(DEBUG_FLAGS)

NVCC_FLAGS := $(CXX_FLAGS) $(DEBUG_FLAGS_NVCC) --expt-extended-lambda

EXES := gemm_host gemm_device

all: $(EXES)

%_host: %.cpp
	$(CXX) $? $(CXX_FLAGS) $(HOST_ONLY_FLAGS) -o $@

%_host_nvcc: %.cpp
	$(NVCC) $? $(CXX_FLAGS) -x c++ -o $@

%_device: %.cpp
	$(NVCC) $? $(NVCC_FLAGS) -x cu -o $@

clean:
	rm -f $(EXES) *.o
	rm -rf *.dSYM
