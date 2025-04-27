#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <mpi.h>
#include <random>
#include <chrono>

const double PI = std::acos(-1);

void print_matrix(const std::vector<std::complex<double>> &matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

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

// Iterative Cooley-Tukey FFT (Radix-2 DIT)
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

void transpose(std::vector<std::complex<double>> &local_data, int rows, int cols)
{
    std::vector<std::complex<double>> transposed(rows * cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            transposed[j * rows + i] = local_data[i * cols + j];
        }
    }
    local_data = transposed;
}

// Calculate how many rows each processor should handle
std::vector<int> calculate_row_distribution(int total_rows, int size)
{
    std::vector<int> rows_per_proc(size);
    int base_rows = total_rows / size;
    int extra = total_rows % size;

    for (int i = 0; i < size; i++)
    {
        rows_per_proc[i] = base_rows + (i < extra ? 1 : 0);
    }

    return rows_per_proc;
}

// Calculate displacements for MPI_Scatterv/Gatherv
std::vector<int> calculate_displacements(const std::vector<int> &counts, int cols)
{
    std::vector<int> displacements(counts.size());
    displacements[0] = 0;

    for (int i = 1; i < counts.size(); i++)
    {
        displacements[i] = displacements[i - 1] + counts[i - 1] * cols;
    }

    return displacements;
}

void fft_parallel(std::vector<std::complex<double>> &data, int rank, int size)
{
    int n = data.size();
    int n_r = std::sqrt(n); // Number of rows/columns in the square matrix

    // Calculate how many rows each processor should handle
    std::vector<int> rows_per_proc = calculate_row_distribution(n_r, size);
    int local_rows = rows_per_proc[rank];

    // Calculate displacements and send counts for MPI communication
    std::vector<int> send_counts(size);
    for (int i = 0; i < size; i++)
    {
        send_counts[i] = rows_per_proc[i] * n_r;
    }
    std::vector<int> displacements = calculate_displacements(rows_per_proc, n_r);

    // First transpose to handle cache misses
    if (rank == 0)
    {
        transpose(data, n_r, n_r);
    }

    // Allocate memory for local data
    std::vector<std::complex<double>> local_data(local_rows * n_r);

    // Scatter the data
    MPI_Scatterv(data.data(), send_counts.data(), displacements.data(), MPI_DOUBLE_COMPLEX,
                 local_data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
                 0, MPI_COMM_WORLD);

    // 1D FFT on each row
    for (int i = 0; i < local_rows; i++)
    {
        std::vector<std::complex<double>> row(local_data.begin() + i * n_r,
                                              local_data.begin() + (i + 1) * n_r);
        fft_serial(row);
        // Copy the transformed row back
        for (int j = 0; j < n_r; j++)
        {
            local_data[i * n_r + j] = row[j];
        }
    }

    // Apply twiddle factors
    for (int i = 0; i < local_rows; i++)
    {
        // Calculate the global row index
        int global_row = displacements[rank] / n_r + i;

        for (int j = 0; j < n_r; j++)
        {
            double factor = -2 * PI * global_row * j / n_r;
            local_data[i * n_r + j] *= std::complex<double>(std::cos(factor), std::sin(factor));
        }
    }

    // Gather the results
    MPI_Gatherv(local_data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
                data.data(), send_counts.data(), displacements.data(), MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

    // Transpose for column-wise FFT
    if (rank == 0)
    {
        transpose(data, n_r, n_r);
    }

    // Scatter the transposed data
    MPI_Scatterv(data.data(), send_counts.data(), displacements.data(), MPI_DOUBLE_COMPLEX,
                 local_data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
                 0, MPI_COMM_WORLD);

    // 1D FFT on each row (which are columns of the original matrix)
    for (int i = 0; i < local_rows; i++)
    {
        std::vector<std::complex<double>> row(local_data.begin() + i * n_r,
                                              local_data.begin() + (i + 1) * n_r);
        fft_serial(row);
        // Copy the transformed row back
        for (int j = 0; j < n_r; j++)
        {
            local_data[i * n_r + j] = row[j];
        }
    }

    // Gather the results
    MPI_Gatherv(local_data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
                data.data(), send_counts.data(), displacements.data(), MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

    // Final transpose to restore original orientation
    if (rank == 0)
    {
        transpose(data, n_r, n_r);
    }
}

std::vector<std::complex<double>> generateRandomData(long long size)
{
    std::vector<std::complex<double>> data;
    data.reserve(size);

    std::random_device rd;                           // Seed
    std::mt19937 gen(rd());                          // Mersenne Twister engine
    std::uniform_real_distribution<> dis(-1.0, 1.0); // Range for random values

    for (long long i = 0; i < size; ++i)
    {
        double real = dis(gen);
        double imag = dis(gen);
        data.emplace_back(real, imag);
    }

    return data;
}

int main()
{
    MPI_Init(nullptr, nullptr);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long array_size = 16 * 16; // Can be any perfect square now

    auto data = generateRandomData(array_size);
    auto data_serial = data; // Make a copy for comparison

    int n = data.size();
    // Check if n is a perfect square
    int sqr = std::sqrt(n);
    if (sqr * sqr != n)
    {
        if (rank == 0)
        {
            std::cerr << "Error: Data size must be a perfect square\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    if (rank == 0)
    {
        start = std::chrono::high_resolution_clock::now();
    }

    fft_parallel(data, rank, size);

    if (rank == 0)
    {
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Parallel FFT time: " << duration.count() << " ms\n";
    }

    // Perform the serial FFT for comparison
    if (rank == 0)
    {
        start = std::chrono::high_resolution_clock::now();
        fft_serial(data_serial);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Serial FFT time: " << duration.count() << " ms\n";

        // Verify the results (optional)
        double error = 0.0;
        for (size_t i = 0; i < data.size(); ++i)
        {
            error += std::abs(data[i] - data_serial[i]);
        }
        std::cout << "Average error: " << error / data.size() << std::endl;
    }

    MPI_Finalize();
    return 0;
}