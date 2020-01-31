# compilers
CXX=g++
NVCC=nvcc

# paths of cuda libraries
CUDAPATH=/software/cuda/10.0.130/usr/local/cuda-10.0/
# for cudpp:
#CUDASDKPATH=/usr/local/cuda/SDK/NVIDIA_GPU_Computing_SDK/C

# includes for header files
CXXINCLUDE=-I$(CUDAPATH)/include
CUDAINCLUDE=-I$(CUDAPATH)/include

# compiler flags: include all warnings, optimization level 3
CXXFLAGS=-O3 -DGOLD_FUNCS=0
NVCCFLAGS=--ptxas-options=-v -O3 -arch=sm_35 -DGOLD_FUNCS=0
# linker flags: include all warnings, include library files
LDFLAGS=-Wall -L$(CUDAPATH)/lib64 -lcudart

# object files
CXXOBJECTS=cross_box.o cudaErr.o dat_file_input.o
NVCCOBJECTS=find_neighbors.o calculate_stress_energy.o data_primitives_2.o strain_dynamics.o

# final executable
EXE=../../exe/cross_box
RSZ_EXE=../../exe/cross_box_resize

#rules:

all: $(EXE) $(RSZ_EXE)
strain: $(EXE)
resz: $(RSZ_EXE)

######################
# compile executable #
######################

$(EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main.o
	$(CXX) -o $(EXE) main.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)

$(RSZ_EXE): $(CXXOBJECTS) $(NVCCOBJECTS) main_resize.o
	$(CXX) -o $(RSZ_EXE) main_resize.o $(CXXOBJECTS) $(NVCCOBJECTS) $(LDFLAGS)


###################
# compile objects #
###################

#c++ files
main.o: main.cpp cross_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main.cpp

main_resize.o: main_resize.cpp cross_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c main_resize.cpp

cross_box.o: cross_box.cpp cross_box.h
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c cross_box.cpp

cudaErr.o: cudaErr.h cudaErr.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c cudaErr.cpp

dat_file_input.o: file_input.h dat_file_input.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDE) -c dat_file_input.cpp

#cuda files
find_neighbors.o: cross_box.h data_primitives.h find_neighbors.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -c find_neighbors.cu

calculate_stress_energy.o: cross_box.h calculate_stress_energy.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -c calculate_stress_energy.cu

strain_dynamics.o: cross_box.h strain_dynamics.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -c strain_dynamics.cu

data_primitives_2.o: data_primitives.h
	$(NVCC) $(NVCCFLAGS) $(CUDAINCLUDE) -c data_primitives_2.cu

#clean up object files or other assembly files
clean:
	rm -f *.o *.ptx

