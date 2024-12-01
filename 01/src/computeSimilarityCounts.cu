/**
 * @brief Computes a similarity score as defined in AoC 2024 Problem 01, part II.
 *
 * @param arr          Input array of non-negative integers (device pointer).
 * @param counts       Input array for counts of each integer (device pointer).
 * @param score        Output integer for score (device pointer).
 * @param n            Number of elements in the input array.
 * @param k            Maximum value in arr and size of counts array.
 *
 * This kernel maps threads onto elements in arr and increments the score by the element in counts
 * whose index is equal to that element's value.
 *
 * Assumptions:
 * - Input values in `arr` are in the range [0, k-1].
 */
extern "C" __global__ void computeSimilarityCounts(const int *arr, const int *counts, int *score, int n, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        int value = arr[tid];
        if (value < k)
        {
            atomicAdd(score, value * counts[value]);
        }
    }
}
