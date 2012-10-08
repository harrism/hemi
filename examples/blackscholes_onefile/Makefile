all: bs_device bs_host_nvcc bs_host_g++

bs_device:
	nvcc blackscholes.cpp -I../ -I../../ -x cu -o bs_device

bs_host_nvcc:
	nvcc blackscholes.cpp -I../ -I../../ -x c++ -o bs_host_nvcc

bs_host_g++:
	g++ blackscholes.cpp -I../ -I../../ -o bs_host_g++

clean:
	rm -rf bs_device bs_host_nvcc bs_host_g++
