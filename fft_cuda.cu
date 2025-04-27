#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cuComplex.h>

const double PI = 3.14159265358979323846;
__constant__ double PI_d = 3.14159265358979323846;

void print_device_memory_usage(const std::string &label = "")
{
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    std::cout << "[Memory] " << label
              << " | Free: " << free_bytes / (1024.0 * 1024.0) << " MB"
              << " | Total: " << total_bytes / (1024.0 * 1024.0) << " MB"
              << " | Used: " << (total_bytes - free_bytes) / (1024.0 * 1024.0) << " MB"
              << std::endl;
}

/**
 * This function performs bit-reversal on the input index.
 * It is used to reorder the input data for the FFT algorithm.
 * The function takes an integer x and the logarithm of the size
 * of the input data (log_n) and returns the bit-reversed index.
 * The bit-reversal is done by iterating through the bits of x
 * and constructing the reversed index.
 */
int bit_reverse(int x, int log_n)
{
    int result = 0;
    for (int i = 0; i < log_n; ++i)
    {
        if (x & (1 << i))
            result |= 1 << (log_n - 1 - i);
    }
    return result;
}

/**
 * This is the kernel function that performs the butterfly operation
 * for the FFT algorithm. It computes the FFT in parallel on the GPU.
 * Each thread handles a specific butterfly operation based on its
 * index.
 */
__global__ void butterfly_kernel(cuDoubleComplex *d_data, int n, int stage, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n / 2)
    {
        int group = idx / (len / 2);
        int butterfly = idx % (len / 2);

        int i = group * len + butterfly;
        int j = i + len / 2;

        if (j >= n)
            return;

        double angle = -2.0 * PI_d * butterfly / len;
        cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));

        cuDoubleComplex u = d_data[i];
        cuDoubleComplex t;
        t.x = w.x * d_data[j].x - w.y * d_data[j].y;
        t.y = w.x * d_data[j].y + w.y * d_data[j].x;

        d_data[i].x = u.x + t.x;
        d_data[i].y = u.y + t.y;
        d_data[j].x = u.x - t.x;
        d_data[j].y = u.y - t.y;
    }
}

// Host function to manage the CUDA FFT execution
void fft_cuda(std::vector<std::complex<double>> &a)
{
    int n = a.size();
    int log_n = std::log2(n);

    std::vector<std::complex<double>> bit_reversed(n);
    for (int i = 0; i < n; ++i)
    {
        int j = bit_reverse(i, log_n);
        bit_reversed[j] = a[i];
    }

    std::vector<cuDoubleComplex> h_data(n);
    for (int i = 0; i < n; ++i)
    {
        h_data[i] = make_cuDoubleComplex(bit_reversed[i].real(), bit_reversed[i].imag());
    }

    print_device_memory_usage("Before cudaMalloc");

    cuDoubleComplex *d_data;
    cudaMalloc(&d_data, n * sizeof(cuDoubleComplex));

    print_device_memory_usage("After cudaMalloc");

    cudaMemcpy(d_data, h_data.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    print_device_memory_usage("After cudaMemcpy to device");

    int blockSize = 256;
    int gridSize = (n / 2 + blockSize - 1) / blockSize;

    for (int len = 2; len <= n; len <<= 1)
    {
        gridSize = (n / 2 + blockSize - 1) / blockSize;
        butterfly_kernel<<<gridSize, blockSize>>>(d_data, n, log_n - std::log2(len), len);
        cudaDeviceSynchronize();
    }

    print_device_memory_usage("After computation");

    cudaMemcpy(h_data.data(), d_data, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
    {
        a[i] = std::complex<double>(h_data[i].x, h_data[i].y);
    }

    cudaFree(d_data);
    print_device_memory_usage("After cudaFree");
}

// Serial FFT implementation using Cooley-Tukey algorithm
void fft_serial(std::vector<std::complex<double>> &a)
{
    int n = a.size();
    int log_n = std::log2(n);

    for (int i = 0; i < n; ++i)
    {
        int j = bit_reverse(i, log_n);
        if (i < j)
            std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1)
    {
        double angle = -2 * PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len)
        {
            std::complex<double> w(1);
            for (int j = 0; j < len / 2; ++j)
            {
                std::complex<double> u = a[i + j];
                std::complex<double> t = w * a[i + j + len / 2];
                a[i + j] = u + t;
                a[i + j + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
}

// Calculate total error
double calculate_total_error(const std::vector<std::complex<double>> &a,
                             const std::vector<std::complex<double>> &b)
{
    if (a.size() != b.size())
        return -1.0;

    double total_error = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        total_error += std::abs(a[i].real() - b[i].real()) + std::abs(a[i].imag() - b[i].imag());
    }
    return total_error;
}

// Benchmark function with memory profiling
void benchmark(int max_power = 20)
{
    std::cout << "Benchmarking FFT implementations:" << std::endl;
    std::cout << "Size\tSerial (ms)\tCUDA (ms)\tSpeedup\tTotal Error" << std::endl;

    double benchmark_inputs[] = {1024 * 1024, 2048 * 2048, 4096 * 4096, 8192 * 8192,
                                 16384 * 16384};

    for (int iter = 0; iter < sizeof(benchmark_inputs) / sizeof(benchmark_inputs[0]); ++iter)
    {
        double n = benchmark_inputs[iter];
        std::vector<std::complex<double>> data_serial(n);
        std::vector<std::complex<double>> data_cuda(n);

        for (int i = 0; i < n; ++i)
        {
            double val = i % 16;
            data_serial[i] = std::complex<double>(val, 0);
            data_cuda[i] = std::complex<double>(val, 0);
        }

        auto start_serial = std::chrono::high_resolution_clock::now();
        fft_serial(data_serial);
        auto end_serial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> serial_time = end_serial - start_serial;

        auto start_cuda = std::chrono::high_resolution_clock::now();
        fft_cuda(data_cuda);
        auto end_cuda = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cuda_time = end_cuda - start_cuda;

        double speedup = serial_time.count() / cuda_time.count();
        double total_error = calculate_total_error(data_serial, data_cuda);

        std::cout << n << "\t"
                  << serial_time.count() << "\t\t"
                  << cuda_time.count() << "\t\t"
                  << speedup << "\t\t"
                  << total_error << std::endl;
    }
}

int main()
{
    std::vector<std::complex<double>> data = {
        {1, 0}, {0, 0}, {-1, 0}, {0, 0}, {1, 0}, {0, 0}, {-1, 0}, {0, 0}, {1, 0}, {0, 0}, {-1, 0}, {0, 0}, {1, 0}, {0, 0}, {-1, 0}, {0, 0}};
    std::vector<std::complex<double>> data_cuda = data;

    fft_serial(data);
    fft_cuda(data_cuda);

    double error = calculate_total_error(data, data_cuda);
    std::cout << "Initial test case total error: " << error << std::endl;

    benchmark();

    return 0;
}
