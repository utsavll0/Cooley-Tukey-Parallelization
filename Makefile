CC = mpic++
CXXFLAGS = -Wall -O3 -std=c++17
LDFLAGS = -lm

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17

MPICXX = mpic++
MPICXXFLAGS = -Wall -O3 -std=c++17
MPILDFLAGS = -lm

TARGET = fft_serial
CUDA_TARGET = fft_cuda
MPI_TARGET = fft_bailey

SRC = fft_serial.cpp
CUDA_SRC = fft_cuda.cu
MPI_SRC = fft_bailey.cpp

OBJ = $(SRC:.cpp=.o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
MPI_OBJ = $(MPI_SRC:.cpp=.o)

all: $(TARGET) $(CUDA_TARGET) $(MPI_TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(CUDA_TARGET): $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(MPI_TARGET): $(MPI_SRC)
	$(MPICXX) $(MPICXXFLAGS) -o $@ $^ $(MPILDFLAGS)

%.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) $(CUDA_OBJ) $(CUDA_TARGET) $(MPI_OBJ) $(MPI_TARGET)

.PHONY: all clean run_serial run_cuda run_mpi run_mpi_large run_all