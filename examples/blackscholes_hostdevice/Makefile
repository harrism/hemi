CXX := g++
NVCC := nvcc

CUDA_PATH := /usr/local/cuda

CXX_FLAGS  := -I../ -I../../ -I$(CUDA_PATH)/include 

# uncomment for debug
#DEBUG_FLAGS := -g -DDEBUG
#DEBUG_FLAGS_NVCC := -G

# comment for debug
CXX_FLAGS += -O3

CXX_FLAGS += $(DEBUG_FLAGS)
NVCC_FLAGS := $(CXX_FLAGS) $(DEBUG_FLAGS_NVCC)

ARCH := $(shell getconf LONG_BIT)

LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64

LIB_FLAGS := $(LIB_FLAGS_$(ARCH)) -lcudart

%.o : %.cpp
	$(CXX)  -c $< -o $@ $(CXX_FLAGS)

%.cuo : %.cu
	$(NVCC) -c $< -o $@ $(NVCC_FLAGS)

EXE := blackscholes

all: $(EXE)

OBJS := blackscholes.o blackscholes.cuo

$(EXE) : $(OBJS)
	$(CXX) $(OBJS) $(LIB_FLAGS) -o $@

clean:
	rm -rf *.o *.cuo $(EXE)
