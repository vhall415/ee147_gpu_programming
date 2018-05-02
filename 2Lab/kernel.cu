/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];

    // determine row/col location
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue = 0;	// product sum


	// LOOP DOESN'T WORK WHEN ONE DIMENSION IS MUCH LARGER THAN THE OTHERS
	// NEED TO FIGURE OUT HOW TO TILE APPROPRIATE NUMBER ACROSS AND DOWN BOTH A AND B MATRICES


    // loop over tiles of matrices M and N to compute product sum
    for(int c = 0; c < (TILE_SIZE+k-1)/TILE_SIZE+1; c++) {
        // load M and N tiles into shared memory
        if(Row < m && c*TILE_SIZE+threadIdx.x < k)
            ds_A[threadIdx.y][threadIdx.x] = A[Row*k + c*TILE_SIZE+threadIdx.x]; // A[Row][c*TILE_SIZE + threadIdx.x];
        else
	    ds_A[threadIdx.y][threadIdx.x] = 0.0;
	if(c*TILE_SIZE + threadIdx.y < k && Col < n)
	    ds_B[threadIdx.y][threadIdx.x] = B[(c*TILE_SIZE+threadIdx.y) * n + Col]; // B[c*TILE_SIZE + threadIdx.y][Col];
        else
	    ds_B[threadIdx.y][threadIdx.x] = 0.0;
	
	__syncthreads();	// wait for all loads to finish

        if(Row < m && Col < n) {
	    for(int i = 0; i < TILE_SIZE; i++)
                Cvalue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if(Row < m && Col < n)
	C[Row*n+Col] = Cvalue;
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------
    
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE

    dim3 DimGrid( (n+BLOCK_SIZE-1)/BLOCK_SIZE + 1, (m+BLOCK_SIZE-1)/BLOCK_SIZE + 1, 1);
    dim3 DimBlock( BLOCK_SIZE, BLOCK_SIZE, 1);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

    mysgemm<<<DimGrid, DimBlock>>>(m, n, k, A, B, C);
}


