# operating system
HOST_OS := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

# architecture 
ARCH := $(shell getconf LONG_BIT)

NVCC := nvcc

ifeq ($(HOST_OS),darwin)
	CXX := clang++
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS := -L$(CUDA_PATH)/lib
else
	CXX := g++
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
	LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64
	LIB_FLAGS := $(LIB_FLAGS_$(ARCH))
endif

# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
GTEST_DIR = googletest
HEMI_DIR = ../
CUDA_DIR := /usr/local/cuda

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# Flags passed to the C++ compiler.
CXXFLAGS += -g -Wall -Wextra -pthread --std=c++11

# Flags passed to nvcc compiler.
NVCCFLAGS += -g -arch=sm_30 --std=c++11 --expt-extended-lambda

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
TESTS_HOST = test_execution_policy_host \
	test_portable_launch_host \
	test_parallel_for_host \
	test_array_host

TESTS_DEVICE = test_execution_policy_device \
	test_launch_device \
	test_portable_launch_device \
	test_parallel_for_device \
	test_array_device

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

HEMI_HEADERS = $(HEMI_DIR)/hemi/*.h

# House-keeping build targets.

all : test_hemi_device test_hemi_host

clean :
	rm -f test_hemi_device test_hemi_host gtest.a gtest_main.a *.o
	rm -rf *.dSYM

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Build tests

test_execution_policy_host.o: test_execution_policy.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -I$(HEMI_DIR) -I$(CUDA_DIR)/include $^ -o $@

test_execution_policy_device.o: test_execution_policy.cpp
	nvcc -c $(CPPFLAGS) $(NVCCFLAGS) -I$(HEMI_DIR) $^ -o $@
	
test_launch_device.o: test_launch.cu
	nvcc -c $(CPPFLAGS) $(NVCCFLAGS) -I$(HEMI_DIR) $^ -o $@

test_portable_launch_host.o: test_portable_launch.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -I$(HEMI_DIR) -I$(CUDA_DIR)/include  $^ -o $@	

test_portable_launch_device.o: test_portable_launch.cpp
	nvcc -x cu $(CPPFLAGS) $(NVCCFLAGS) -I$(HEMI_DIR) -I$(CUDA_DIR)/include $^ -c -o $@

test_parallel_for_host.o: test_parallel_for.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -I$(HEMI_DIR) -I$(CUDA_DIR)/include $^ -o $@

test_parallel_for_device.o: test_parallel_for.cpp
	nvcc -x cu $(CPPFLAGS) $(NVCCFLAGS) -I$(HEMI_DIR) -I$(CUDA_DIR)/include $^ -c -o $@

test_array_host.o: test_array.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -I$(HEMI_DIR) -I$(CUDA_DIR)/include $^ -o $@

test_array_device.o: test_array.cpp 
	nvcc -x cu $(CPPFLAGS) $(NVCCFLAGS) -I$(HEMI_DIR) -I$(CUDA_DIR)/include $^ -c -o $@
	
test_hemi_device: $(TESTS_DEVICE:=.o) gtest_main.a
	nvcc $^ gtest_main.a -lpthread $(LIB_FLAGS) -o $@

test_hemi_host: $(TESTS_HOST:=.o) gtest_main.a
	nvcc $^ gtest_main.a -lpthread $(LIB_FLAGS) -o $@

