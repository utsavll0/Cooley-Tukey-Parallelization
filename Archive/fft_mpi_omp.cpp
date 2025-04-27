#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <mpi.h>

const double PI = std::acos(-1);

// Removed custom MPI_COMPLEX declaration

// Bit reversal permutation function
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

// Local FFT without OpenMP parallelism
void fft_local(std::vector<std::complex<double>> &a)
{
    int n = a.size();
    int log_n = std::log2(n);

    // Bit-reversal reordering
    for (int i = 0; i < n; ++i)
    {
        int j = bit_reverse(i, log_n);
        if (i < j)
            std::swap(a[i], a[j]);
    }

    // Iterative FFT without OpenMP parallelism
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

// Distributed memory FFT using MPI (no OpenMP)
void fft_parallel(std::vector<std::complex<double>> &local_data, int n, int rank, int size)
{
    int local_n = local_data.size();
    int log_n = std::log2(n);
    int local_log_n = std::log2(local_n);

    // Perform local FFT on each process
    fft_local(local_data);

    // Perform the distributed stages of the FFT
    for (int len = local_n * 2; len <= n; len <<= 1)
    {
        int stage = std::log2(len) - 1;
        int partner = rank ^ (1 << (stage - local_log_n));

        // Create buffers for send/receive
        std::vector<std::complex<double>> send_buffer(local_n);
        std::vector<std::complex<double>> recv_buffer(local_n);

        double angle = -2 * PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));

        // Fill send buffer
        for (int i = 0; i < local_n; ++i)
        {
            send_buffer[i] = local_data[i];
        }

        // Exchange data with partner process
        MPI_Sendrecv(send_buffer.data(), local_n, MPI_DOUBLE_COMPLEX, partner, 0,
                     recv_buffer.data(), local_n, MPI_DOUBLE_COMPLEX, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform butterfly operations
        for (int i = 0; i < local_n; ++i)
        {
            int global_i = rank * local_n + i;
            std::complex<double> w = std::pow(wlen, global_i % (len / 2));

            if ((global_i / (len / 2)) % 2 == 0)
            {
                // Upper part of butterfly
                local_data[i] = send_buffer[i] + w * recv_buffer[i];
            }
            else
            {
                // Lower part of butterfly
                local_data[i] = send_buffer[i] - w * recv_buffer[i];
            }
        }
    }
}

// Calculate error between two complex vectors
double calculate_total_error(const std::vector<std::complex<double>> &v1,
                             const std::vector<std::complex<double>> &v2)
{
    if (v1.size() != v2.size())
        return -1.0; // Error: vectors of different sizes

    double max_error = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        double real_diff = std::abs(v1[i].real() - v2[i].real());
        double imag_diff = std::abs(v1[i].imag() - v2[i].imag());
        max_error = std::max(max_error, std::max(real_diff, imag_diff));
    }
    return max_error;
}

// Gather all distributed data to rank 0
void gather_results(std::vector<std::complex<double>> &local_data,
                    std::vector<std::complex<double>> &full_data,
                    int rank, int size)
{
    int local_n = local_data.size();
    int n = local_n * size;

    if (rank == 0)
    {
        full_data.resize(n);
    }

    MPI_Gather(local_data.data(), local_n, MPI_DOUBLE_COMPLEX,
               full_data.data(), local_n, MPI_DOUBLE_COMPLEX,
               0, MPI_COMM_WORLD);
}

// Serial FFT implementation for comparison
void fft_serial(std::vector<std::complex<double>> &a)
{
    int n = a.size();
    int log_n = std::log2(n);

    // Bit-reversal reordering
    for (int i = 0; i < n; ++i)
    {
        int j = bit_reverse(i, log_n);
        if (i < j)
            std::swap(a[i], a[j]);
    }

    // Iterative FFT
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

// Benchmark function to compare serial and parallel FFT implementations
void benchmark(int max_power = 20, int rank = 0, int size = 1)
{
    if (rank == 0)
    {
        std::cout << "Benchmarking FFT implementations:" << std::endl;
        std::cout << std::setw(10) << "Size" << "\t"
                  << std::setw(15) << "Serial (ms)" << "\t"
                  << std::setw(15) << "Parallel (ms)" << "\t"
                  << std::setw(10) << "Speedup" << "\t"
                  << std::setw(15) << "Max Error" << std::endl;
    }

    // Set the minimum power to ensure the data can be divided among all processes
    int min_power = std::ceil(std::log2(size));
    min_power = std::max(min_power, 3); // At least 2^3 = 8 elements

    for (int power = min_power; power <= max_power; ++power)
    {
        int n = 1 << power;

        // Skip if n cannot be evenly distributed across processes
        if (n % size != 0)
            continue;

        int local_n = n / size;

        // Create full test data on rank 0
        std::vector<std::complex<double>> full_data;
        std::vector<std::complex<double>> data_serial;

        if (rank == 0)
        {
            // Initialize data
            full_data.resize(n);
            data_serial.resize(n);

            // Fill with a test pattern
            for (int i = 0; i < n; ++i)
            {
                double val = i % 16; // Simple repeating pattern
                full_data[i] = std::complex<double>(val, 0);
                data_serial[i] = std::complex<double>(val, 0);
            }
        }

        // Distribute data to all processes
        std::vector<std::complex<double>> local_data(local_n);

        MPI_Scatter(rank == 0 ? full_data.data() : nullptr, local_n, MPI_DOUBLE_COMPLEX,
                    local_data.data(), local_n, MPI_DOUBLE_COMPLEX,
                    0, MPI_COMM_WORLD);

        // Measure serial FFT time (only on rank 0)
        double serial_time = 0.0;
        if (rank == 0)
        {
            auto start_serial = std::chrono::high_resolution_clock::now();
            fft_serial(data_serial);
            auto end_serial = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end_serial - start_serial;
            serial_time = duration.count();
        }

        // Synchronize before parallel timing
        MPI_Barrier(MPI_COMM_WORLD);

        // Measure parallel FFT time
        auto start_parallel = std::chrono::high_resolution_clock::now();
        fft_parallel(local_data, n, rank, size);
        auto end_parallel = std::chrono::high_resolution_clock::now();

        double local_time = std::chrono::duration<double, std::milli>(
                                end_parallel - start_parallel)
                                .count();

        double parallel_time = 0.0;
        MPI_Reduce(&local_time, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Gather results to rank 0
        std::vector<std::complex<double>> full_result;
        gather_results(local_data, full_result, rank, size);

        // Calculate error between serial and parallel results (only on rank 0)
        double error = 0.0;
        if (rank == 0)
        {
            error = calculate_total_error(data_serial, full_result);
        }

        // Output benchmark results
        if (rank == 0)
        {
            double speedup = serial_time / parallel_time;
            std::cout << std::setw(10) << n << "\t"
                      << std::setw(15) << serial_time << "\t"
                      << std::setw(15) << parallel_time << "\t"
                      << std::setw(10) << speedup << "\t"
                      << std::setw(15) << error << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    benchmark(20, rank, size);

    MPI_Finalize();
    return 0;
}
