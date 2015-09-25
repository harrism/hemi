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

EXES := hello_device hello_host_nvcc hello_host \
		hello_lambda_device hello_lambda_host_nvcc hello_lambda_host \
        hello_global \
        saxpy_host saxpy_device

all: $(EXES)

hello_device: hello.cpp
	$(NVCC) $? $(NVCC_FLAGS) -x cu -o $@

hello_host_nvcc: hello.cpp
	$(NVCC) $? $(CXX_FLAGS) -x c++ -o $@

hello_host: hello.cpp
	$(CXX) $? $(CXX_FLAGS) $(HOST_ONLY_FLAGS) -o $@

hello_lambda_device: hello_lambda.cpp
	$(NVCC) $? $(NVCC_FLAGS) -x cu -o $@

hello_lambda_host_nvcc: hello_lambda.cpp
	$(NVCC) $? $(CXX_FLAGS) -x c++ -o $@

hello_lambda_host: hello_lambda.cpp
	$(CXX) $? $(CXX_FLAGS) $(HOST_ONLY_FLAGS) -o $@

hello_global: hello_global.cu
	$(NVCC) $? $(NVCC_FLAGS) -x cu -o $@

saxpy_host: saxpy.cpp
	$(CXX) $? $(CXX_FLAGS) $(HOST_ONLY_FLAGS) -o $@

saxpy_device: saxpy.cpp
	$(NVCC) $? $(NVCC_FLAGS) -x cu -o $@

clean:
	rm -f $(EXES) *.o
	rm -rf *.dSYM
