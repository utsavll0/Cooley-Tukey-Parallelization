#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <mpi.h>
#include <random>
#include <chrono>

const double PI = 3.14159265358979323846;

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

// Normal serial FFT using Cooley-Tukey algorithm
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

void transpose(std::vector<std::complex<double>> &local_data)
{
    int n = local_data.size();
    int dim = std::sqrt(n);
    std::vector<std::complex<double>> transposed(n);
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            transposed[j * dim + i] = local_data[i * dim + j];
        }
    }
    local_data = transposed;
}

/**
 * Follows the Bailey's algorithm for parallel FFT
 * Transpose happens on the root node then the data is scattered
 * Each process computes the FFT on its local data (rows of scattered data)
 * Then the data is gathered on the root node
 * The root node transposes the data again and scatters it again
 * Each process computes the FFT on its local data (rows of scattered data)
 * Then the data is gathered on the root node
 */
void fft_parallel(std::vector<std::complex<double>> &data, int rank, int size)
{
    int n = data.size();
    int n_r = std::sqrt(n);
    int local_rows = n_r / size;

    if (rank == 0)
        transpose(data);

    std::vector<std::complex<double>> local_data(local_rows * n_r);
    MPI_Scatter(data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
                local_data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

    std::vector<std::complex<double>> local_fft(local_rows * n_r);
    for (int i = 0; i < local_rows; i++)
    {
        std::vector<std::complex<double>> row(local_data.begin() + i * n_r,
                                              local_data.begin() + (i + 1) * n_r);
        fft_serial(row);
        for (int j = 0; j < n_r; j++)
        {
            local_fft[i * n_r + j] = row[j];
        }
    }
    for (int i = 0; i < local_rows; i++)
    {
        for (int j = 0; j < n_r; j++)
        {
            int p = rank + i;
            int q = j;
            double factor = -2 * PI * p * q / n;
            local_fft[i * n_r + j] = local_fft[i * n_r + j] * std::exp(std::complex<double>(0, factor));
        }
    }

    MPI_Gather(local_fft.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
               data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
               0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        transpose(data);
    }
    MPI_Scatter(data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
                local_data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
    for (int i = 0; i < local_rows; i++)
    {
        std::vector<std::complex<double>> row(local_data.begin() + i * n_r,
                                              local_data.begin() + (i + 1) * n_r);
        fft_serial(row);
        for (int j = 0; j < n_r; j++)
        {
            local_data[i * n_r + j] = row[j];
        }
    }
    MPI_Gather(local_data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
               data.data(), local_rows * n_r, MPI_DOUBLE_COMPLEX,
               0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        transpose(data);
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

int main(int argc, char *argv[])
{
    MPI_Init(nullptr, nullptr);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2)
    {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <matrix size>\n";
        MPI_Finalize();
        return 1;
    }

    int s = std::stoi(argv[1]);
    long long array_size = s * s;

    auto data = generateRandomData(array_size);
    auto data_serial = generateRandomData(array_size);

    int n = data.size();
    // check if n is a perfect square
    int sqr = std::sqrt(n);
    if (sqr * sqr != n)
    {
        std::cerr << "Error: Data size must be a perfect square\n";
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

    // Perform the FFT
    if (rank == 0)
    {
        start = std::chrono::high_resolution_clock::now();
        fft_serial(data_serial);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Serial FFT time: " << duration.count() << " ms\n";
    }
    MPI_Finalize();

    return 0;
}
