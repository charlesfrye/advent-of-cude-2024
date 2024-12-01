/**
 * @brief Counts occurrences of non-negative integers in an array.
 *
 * @param arr          Input array of non-negative integers (device pointer).
 * @param counts       Output array for counts of each integer (device pointer).
 * @param n            Number of elements in the input array.
 * @param k            Maximum value in arr and size of counts array. Unused.
 *
 * This kernel maps threads onto elements in arr and increments the element in counts
 * whose index is equal to that element's value.
 *
 * Assumptions:
 * - Input values in `arr` are in the range [0, k-1].
 */
extern "C" __global__ void countOccurrences(const int *arr, int *counts, int n, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        atomicAdd(&counts[arr[tid]], 1);
    }
}

/**
 * @brief Counts occurrences of non-negative integers in a sorted array using shared memory.
 *
 * @param sorted_arr   Input array of non-negative integers (device pointer).
 * @param counts       Output array for counts of each integer (device pointer).
 * @param n            Number of elements in the input array.
 * @param k            Number of unique values (size of counts array).
 *
 * This kernel maps threads onto entries in the sorted_arr. Threads at the beginnings of blocks
 * or at places where the value changes count up matching entries.
 * The number of contended writes is at most two per block.
 *
 * Assumptions:
 * - Input values in `sorted_arr` are in the range [0, k-1].
 * - `sorted_arr` is sorted before the kernel is launched.
 */
extern "C" __global__ void countOccurrencesSorted(const int *sorted_arr, int *counts, int n, int k)
{
    extern __shared__ int shared_data[];
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int global_tid = block_start + tid;

    // pre-load into shared memory
    if (global_tid < n)
    {
        shared_data[tid] = sorted_arr[global_tid];
    }
    __syncthreads();

    // if we're the first thread in a block or if we're at a change point
    if (tid == 0 || (tid < blockDim.x && shared_data[tid] != shared_data[tid - 1]))
    {
        int val = shared_data[tid]; // count occurrences for our value
        if (global_tid < n && val < k)
        {
            int count = 1;

            // iterate through shared memory while our value still matches
            while (tid + count < blockDim.x && val == shared_data[tid + count])
            {
                count++;
            }

            // add to global count (atomically to handle counting across blocks)
            atomicAdd(&counts[val], count);
        }
    }
}

// bad -- shared memory is O(N)

// extern "C" __global__ void countOccurrencesShared(const int *arr, int *counts, int n, int k)
// {
//     // Shared memory for block-local counts
//     extern __shared__ int shared_counts[];
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // Initialize shared memory
//     for (int i = threadIdx.x; i < k; i += blockDim.x)
//     {
//         shared_counts[i] = 0;
//     }
//     __syncthreads();

//     // Update local counts
//     if (tid < n)
//     {
//         atomicAdd(&shared_counts[arr[tid]], 1);
//     }
//     __syncthreads();

//     // Accumulate results into global memory
//     for (int i = threadIdx.x; i < k; i += blockDim.x)
//     {
//         atomicAdd(&counts[i], shared_counts[i]);
//     }
// }
