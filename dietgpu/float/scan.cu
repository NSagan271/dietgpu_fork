#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// KERNELS USED FOR THE IMPLEMENTATION OF EXCLUSIVE_SCAN + FIND_REPEATS

__global__ void construct_adj_flags(int* device_input, int length, int* flags_output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length - 1) return;
    if (device_input[tid] == device_input[tid+1])
        flags_output[tid] = 1;
}
__global__ void map_flags(int* flags_input, int* indices_input, int length, int* device_output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // only the first N-1 elements are candidates for repeats
    if (tid >= length - 1) return;
    // we only want to write results for repeats (since ES produces multiple copies of indices)
    if (flags_input[tid] == 0) return;
    device_output[indices_input[tid]] = tid;
}

__global__ void upsweep_step(int* input, int length, int two_d, int* result, bool final) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long two_dplus1 = 2 * two_d;
    long i = tid * two_dplus1;
    // we index EVERY two_dplus1, so we must respect this in kernel (else we can have race conditions)
    if (i >= length) return;
    result[i+two_dplus1-1] = result[i+two_dplus1-1] + result[i+two_d-1];
    if (final && i + two_dplus1 >= length)
        result[length-1] = 0;
}

__global__ void downsweep_step(int* input, int length, int two_d, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long two_dplus1 = 2 * two_d;
    long i = tid * two_dplus1;
    if (i >= length) return;
    // we index EVERY two_dplus1, so we must respect this in kernel (else we can have race conditions)
    int t = result[i+two_d-1];
    result[i+two_d-1] = result[i+two_dplus1-1];
    result[i+two_dplus1-1] = result[i+two_dplus1-1] + t; 

}



// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{
    long rounded_length = nextPow2(N);
    cudaMemcpy(result, input, rounded_length*sizeof(int), cudaMemcpyDeviceToDevice);
    // upsweep phase
    for (long two_d=1; two_d<=rounded_length/2; two_d*=2) {
        bool final = (two_d*2 > rounded_length/2);
        // use only as many threads are as needed
        int n_threads_needed = (rounded_length / (two_d * 2)); // will always cleanly divide
        int num_blocks = (n_threads_needed+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
        upsweep_step<<<num_blocks, THREADS_PER_BLOCK>>>(input, rounded_length, two_d, result, final);
    }
    cudaDeviceSynchronize();
    // downsweep phase
    for (long two_d=rounded_length/2; two_d>=1; two_d/=2) {
        int n_threads_needed = (rounded_length / (two_d * 2)); // will always cleanly divide
        int num_blocks = (n_threads_needed+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
        downsweep_step<<<num_blocks, THREADS_PER_BLOCK>>>(input, rounded_length, two_d, result);
    }
    cudaDeviceSynchronize();
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__ void find_index(int* input, int N, int* result) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N - 1) return;
    // printf("tid: %d\n", tid);
    // printf("input[tid]: %d\n", input[tid]);
    if (input[tid] != input[tid+1]) {
        result[input[tid]] = tid;
    }
}



template <typename T>
__global__ void fill_output_sparse(int* input, int N, T* sparseResult, T* decResult) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // printf("tid: %d ", tid);
    // printf("input: %d ", input[tid]);
    sparseResult[input[tid]] = decResult[tid];
}

template <typename T>
__global__ void fill_origin_dense(int* sparseIdx, int N, T* sparseInput, T* denseInput) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // printf("tid: %d\n", tid);
    // printf("sparseIdx: %d ", sparseIdx[tid]);
    // printf("sparseInput[sparseIdx[tid]]: %lld\n", sparseInput[sparseIdx[tid]]);
    denseInput[tid] = sparseInput[sparseIdx[tid]];
    // printf("tid: %d\n", tid);
    // printf("input[tid]: %d\n", input[tid]);
    // denseInput[tid] = sparseInput[input[tid]];
}

template <typename T>
__global__ void fill_last_bit(T* res, T* inp, int res_i, int inp_i) {
    inp[inp_i] = res[res_i];
}


__global__ void fill_last_bit_with_int(int f, int* inp, int inp_i) {
    // printf("f: %d\n", f);
    // printf("inp_i: %d\n", inp_i);
    inp[inp_i] = f;

    // printf("inp[inp_i]: %d\n", inp[inp_i]);
}
// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    // first get the flags representing adjacencies
    int* adj_flags;
    int rounded_length = nextPow2(length);
    int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // cudaMalloc((void **)&adj_flags, length*sizeof(int));
    cudaMalloc((void **)&adj_flags, rounded_length*sizeof(int));
    construct_adj_flags<<<num_blocks, THREADS_PER_BLOCK>>>(device_input, length, adj_flags);
    int* indices_into_output;
    // cudaMalloc((void **)& indices_into_output, length*sizeof(int));
    cudaMalloc((void **)& indices_into_output, rounded_length*sizeof(int));
    // run ES to get indices to place answers into output
    exclusive_scan(adj_flags, length, indices_into_output);
    // get desired 'solution indices'
    map_flags<<<num_blocks, THREADS_PER_BLOCK>>>(adj_flags, indices_into_output, length, device_output);

    // // since this function returns the # of repeats, we must query last item of `indices_into_output`
    int num_repeats[1];
    cudaMemcpy(num_repeats, indices_into_output + (length - 1), sizeof(int), cudaMemcpyDeviceToHost);
    // // clean up CUDA arrays allocated during this function call
    cudaFree(adj_flags);
    cudaFree(indices_into_output);
    return num_repeats[0];
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
