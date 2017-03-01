#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
    { gpuAssert((ans), __FILE__, __LINE__); }


__global__ void histogram_kernel_shared(unsigned int *input, unsigned int *bins, unsigned int inputLength, unsigned int binLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int stride = blockDim.x * gridDim.x;

    extern __shared__ unsigned int binsShared[]; //extern??

	for (int x = threadIdx.x; x < binLength; x += blockDim.x)
	{
		binsShared[x] = 0;
	}

	__syncthreads();

    for (int x = i; x < inputLength; x += stride)
    {
        atomicAdd(&(binsShared[input[x]]), 1);
    }

    __syncthreads();

    for (int x = threadIdx.x; x < binLength; x += blockDim.x)
    {

        atomicAdd(&(bins[x]), binsShared[x]);
    }

    __syncthreads();




}

__global__ void truncate(unsigned int *input, unsigned int *bins, unsigned int inputLength, unsigned int binLength)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < binLength)
	{
		if (bins[i] > 127)
			bins[i] = 127;
	}
}

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins, unsigned int inputLength, unsigned int binLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int stride = blockDim.x * gridDim.x;

	__syncthreads();

 
	if (i < binLength)
    {
        atomicAdd(&(bins[input[i]]), 1);
	
		if (i == 0)
			bins[0] = 0;

    }

	bins[0] = 0; //remove


    __syncthreads();


    if (i < binLength)
    {
		if (bins[i] > 127)
		{
			bins[i] = 127;
		}
    }


}


inline void gpuAssert(cudaError_t code, const char *file, int line,
	bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
			file, line);
		if (abort)
			exit(code);
	}
}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
		&inputLength, "Integer");
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating GPU memory.");
	// TODO: Allocate GPU memory here
	cudaMalloc((void**) &deviceInput, inputLength * sizeof(unsigned int));
	cudaMalloc((void**) &deviceBins, NUM_BINS * sizeof(unsigned int));

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	// TODO: Copy memory to the GPU here

    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Launch kernel
	// ----------------------------------------------------------
	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");

	// TODO: Perform kernel computation here

	cudaMemset((void *)deviceBins, 0, NUM_BINS * sizeof(unsigned int));
	memset((void *)hostBins, 0, NUM_BINS * sizeof(unsigned int));
	

	CUDA_CHECK(cudaDeviceSynchronize());

    dim3 gridDim(32);
    dim3 blockDim(512);

    histogram_kernel_shared<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
	truncate << <gridDim, blockDim, NUM_BINS * sizeof(unsigned int) >> >(deviceInput, deviceBins, inputLength, NUM_BINS);


	// You should call the following lines after you call the kernel.
	// CUDA_CHECK(cudaGetLastError());
	// CUDA_CHECK(cudaDeviceSynchronize());

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	// TODO: Copy the GPU memory back to the CPU here

    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	// TODO: Free the GPU memory here

    cudaFree(deviceInput);
    cudaFree(deviceBins);

	wbTime_stop(GPU, "Freeing GPU Memory");

	// Verify correctness
	// -----------------------------------------------------
	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
