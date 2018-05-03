/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void hist_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    extern __shared__ unsigned int hist[];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int initial = i;
    if(i < num_bins)
	hist[i] = 0;
    //if(threadIdx.x == 256)
//	hist[threadIdx.x] = 1;
    __syncthreads();

    //int i = threadIdx.x + blockDim.x * blockIdx.x;

    int stride = blockDim.x * gridDim.x;

    while(i < num_elements) {
	atomicAdd(&(hist[input[i]]), 1);
	i += stride;
    }
    __syncthreads();
    
    if(initial < num_bins)
	atomicAdd(&bins[initial], hist[initial]);
	//bins[threadIdx.x] = hist[threadIdx.x];
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    // INSERT CODE HERE
    const unsigned int BLOCK_SIZE = 256;

    dim3 DimGrid( (num_elements-1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    printf("Number of blocks: %d\n", (num_elements-1)/BLOCK_SIZE + 1);

    hist_kernel<<<DimGrid, DimBlock, num_bins*sizeof(unsigned int)>>>(input, bins, num_elements, num_bins);

}
