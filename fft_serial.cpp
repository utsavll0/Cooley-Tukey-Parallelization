#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>

const double PI = std::acos(-1);

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

// Write function to output data to a file
void write_to_file(const std::vector<std::complex<double>> &v, const std::string &filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (const auto &c : v)
        {
            file << c.real() << " " << c.imag() << "\n";
        }
        file.close();
    }
    else
    {
        std::cerr << "Error opening file " << filename << "\n";
    }
}

int main()
{
    std::vector<std::complex<double>> data = {
        {1, 0}, {0, 0}, {-1, 0}, {0, 0}, {1, 0}, {0, 0}, {-1, 0}, {0, 0}};

    // Write the original data to a file
    write_to_file(data, "input_data.txt");

    // Perform the FFT
    fft_serial(data); // Forward FFT

    // Write the FFT output to a file
    write_to_file(data, "fft_output.txt");

    return 0;
}
