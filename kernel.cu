/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
    

__global__ void VecAdd(int n, const float *A, const float *B, float* C) {
	//DEVICE(GPU)CODE
    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (1 * n) vector
     *   where B is a (1 * n) vector
     *   where C is a (1 * n) vector
     *
     ********************************************************************/
//added for extra compute time
long long start = clock64();
long long cycles_elapsed;
do{cycles_elapsed = clock64() - start;}
while(cycles_elapsed <20000);
//end of added compute time
    // INSERT KERNEL CODE HERE
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    	if (i < n)
		C[i] = A[i] + B[i];
}


void basicVecAdd( float *A,  float *B, float *C, int n)
{
	//HOST(CPU)CODE
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = 256; 

    //INSERT CODE HERE
dim3 dimGRID((n-1)/BLOCK_SIZE + 1,1,1);
dim3 dimBLOCK(BLOCK_SIZE,1,1);
VecAdd<<<dimGRID,dimBLOCK>>>(n,A,B,C);
}
