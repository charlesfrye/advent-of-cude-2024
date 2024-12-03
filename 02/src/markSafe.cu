// #include <cuda/std/cstdint> // don't need this, int8_t is simple
typedef signed char int8_t;

#define COLS 8

/**
 * @brief Counts "safe" rows of reports, as defined in part 1 of day 2 of Advent of Code 2024.
 *
 * @param arr       Input array of COLS-element arrays of 8 bit integers (device pointer).
 * @param out       Output array of boolean "safety" values for each row of input (device pointer).
 * @param n         Number of rows in the input array and elements in the output array.
 *
 * Assumptions:
 * - Both d_reports and d_safe have the same number of "rows", n
 */

extern "C" __global__ void markSafe(const int8_t *arr, int8_t *out, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.y;
    int tid = row * COLS + threadIdx.y;
    int local_tid = threadIdx.x * COLS + threadIdx.y;
    __shared__ int8_t scratch[32 * COLS];
    int8_t is_safe = 1;
    int8_t delta;
    int8_t direction;
    if (row < n)
    {
        scratch[local_tid] = arr[tid];
    }
    __syncthreads();

    delta = (col > 0) ? scratch[local_tid] - scratch[local_tid - 1] : -100;
    delta = (scratch[local_tid] > 0) ? delta : -100;
    __syncthreads();

    scratch[local_tid] = delta;
    __syncthreads();

    if (row < n)
    {
        if (col == 0)
        {
            direction = 2 * (scratch[local_tid + 1] > 0) - 1;
            for (int ii = 1; ii < COLS; ii++)
            {
                delta = scratch[local_tid + ii];
                if (delta == -100)
                {
                    out[row] = is_safe;
                    return;
                }
                if (delta * direction < 1 || delta * direction > 3)
                {
                    is_safe = 0;
                }
            }
            out[row] = is_safe;
        }
    }
}
