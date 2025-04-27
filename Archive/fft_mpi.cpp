#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <mpi.h>

const double PI = std::acos(-1);

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

void reorder(int n, int log_n, std::vector<std::complex<double>> &a)
{
    for (int i = 0; i < n; ++i)
    {
        int j = bit_reverse(i, log_n);
        if (i < j)
            std::swap(a[i], a[j]);
    }
}

void fft_serial(std::vector<std::complex<double>> &a)
{
    int n = a.size();
    int log_n = std::log2(n);

    // Bit-reversal reordering
    reorder(n, log_n, a);

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

void fft_parallel(std::vector<std::complex<double>> &a, int rank, int size)
{
    int n = a.size(); // global size
    int log_n = std::log2(n);
    int local_n = n / size; // size per process

    if (rank == 0)
        reorder(n, log_n, a); // Only done on rank 0

    std::vector<std::complex<double>> local_data(local_n);
    MPI_Scatter(a.data(), local_n, MPI_CXX_DOUBLE_COMPLEX,
                local_data.data(), local_n, MPI_CXX_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

    for (int len = 2; len <= n; len <<= 1)
    {
        double angle = -2 * PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        int half_len = len / 2;

        if (len <= local_n)
        {
            // Local FFT step
            for (int i = 0; i < local_n; i += len)
            {
                std::complex<double> w(1);
                for (int j = 0; j < half_len; ++j)
                {
                    std::complex<double> u = local_data[i + j];
                    std::complex<double> t = w * local_data[i + j + half_len];
                    local_data[i + j] = u + t;
                    local_data[i + j + half_len] = u - t;
                    w *= wlen;
                }
            }
        }
        else
        {
            // Inter-process butterflies
            int group_size = len / local_n;
            int group_id = rank / group_size;
            int partner = rank ^ (group_size / 2);

            // Allocate buffer to exchange data
            std::vector<std::complex<double>> recv_buf(local_n);

            MPI_Sendrecv(local_data.data(), local_n, MPI_CXX_DOUBLE_COMPLEX, partner, 0,
                         recv_buf.data(), local_n, MPI_CXX_DOUBLE_COMPLEX, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Compute base global index of local data
            int global_offset = rank * local_n;
            std::vector<std::complex<double>> new_local(local_n);

            for (int i = 0; i < local_n; ++i)
            {
                int global_index = global_offset + i;
                int butterfly_pair = global_index % len;

                std::complex<double> w = std::polar(1.0, angle * butterfly_pair);

                bool lower_half = (global_index % len) < half_len;

                if ((rank & (group_size / 2)) == 0)
                {
                    // This is the lower half processor
                    new_local[i] = local_data[i] + w * recv_buf[i];
                }
                else
                {
                    // Upper half processor
                    new_local[i] = w * recv_buf[i] - local_data[i];
                }
            }

            local_data = std::move(new_local);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Gather(local_data.data(), local_n, MPI_CXX_DOUBLE_COMPLEX,
               a.data(), local_n, MPI_CXX_DOUBLE_COMPLEX,
               0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((size & (size - 1)) != 0)
    {
        std::cerr << "Error: Number of processes must be a power of 2\n";
        MPI_Finalize();
        return 1;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::complex<double>> data = {
        {1, 0}, {0, 0}, {-1, 0}, {0, 0}, {1, 0}, {0, 0}, {-1, 0}, {0, 0}};
    std::vector<std::complex<double>> data_2 = {
        {1, 0}, {0, 0}, {-1, 0}, {0, 0}, {1, 0}, {0, 0}, {-1, 0}, {0, 0}};
    fft_serial(data);
    if (rank == 0)
    {
        for (int i = 0; i < data.size(); ++i)
        {
            std::cout << "data[" << i << "] = " << data[i] << std::endl;
        }
    }

    fft_parallel(data_2, rank, size);
    if (rank == 0)
    {
        for (int i = 0; i < data_2.size(); ++i)
        {
            std::cout << "data_2[" << i << "] = " << data_2[i] << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
