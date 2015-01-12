# operating system
HOST_OS := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

# architecture 
ARCH := $(shell getconf LONG_BIT)

NVCC := nvcc

ifeq ($(HOST_OS),darwin)
	CXX := clang++
	CUDA_PATH ?= /Developer/NVIDIA/CUDA-7.0
	LIB_FLAGS := -L$(CUDA_PATH)/lib
else
	CXX) := g++
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
	LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64
	LIB_FLAGS := $(LIB_FLAGS_$(ARCH))
endif

LIB_FLAGS += -lcudart

STD := -std=c++11

CXX_FLAGS  := $(STD) -I../ -I../../ -I$(CUDA_PATH)/include 

# uncomment for debug
#DEBUG_FLAGS := -g -DDEBUG
#DEBUG_FLAGS_NVCC := -G

# comment for debug
CXX_FLAGS += -O3

CXX_FLAGS += $(DEBUG_FLAGS)
NVCC_FLAGS := $(CXX_FLAGS) $(DEBUG_FLAGS_NVCC)

NVCC_FLAGS := $(CXX_FLAGS) $(DEBUG_FLAGS_NVCC)

%.o : %.cpp
	$(CXX)  -c $< -o $@ $(CXX_FLAGS)

%.cuo : %.cu
	$(NVCC) -c $< -o $@ $(NVCC_FLAGS)

EXE := blackscholes

all: $(EXE)

OBJS := blackscholes.o blackscholes.cuo

$(EXE) : $(OBJS)
	$(CXX) $(OBJS) $(CXX_FLAGS) $(LIB_FLAGS) -o $@

clean:
	rm -rf *.o *.cuo $(EXE)
