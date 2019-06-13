/******************************************************************************
 *Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
//use pinned memory and buffers cudaHostAlloc()
#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

int main (int argc, char *argv[])
{
    //set standard seed
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;
    cudaStream_t stream0,stream1,stream2;
   cudaStreamCreate(&stream0);
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   const unsigned int BLOCK_SIZE = 256;   
 // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d0, *B_d0, *C_d0;
    float *A_d1, *B_d1, *C_d1;
    float *A_d2, *B_d2, *C_d2;
    size_t A_sz, B_sz, C_sz, total_sz;
    unsigned VecSize;
   
    dim3 dim_grid, dim_block;

      if (argc == 1) {
        VecSize = 1000000;
      } 
      else if (argc == 2) {
      VecSize = atoi(argv[1]);     
      }
      else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
      }

    A_sz = 333333;
    B_sz = 333333;
    C_sz = 333333;
    total_sz=1000000;
    cudaHostAlloc( (void **) &A_h, sizeof(float)*total_sz,cudaHostAllocDefault );
    for (unsigned int i=0; i < total_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    cudaHostAlloc( (void **) &B_h, sizeof(float)*total_sz,cudaHostAllocDefault );
    for (unsigned int i=0; i < total_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    cudaHostAlloc( (void **) &C_h, sizeof(float)*total_sz,cudaHostAllocDefault);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------
//tell the GPU/device how much of its (the device's) memory to allocate
    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
	cudaMalloc((void**) &A_d0,4*A_sz);
	cudaMalloc((void**) &B_d0,4*B_sz);
	cudaMalloc((void**) &C_d0,4*C_sz);
        cudaMalloc((void**) &A_d1,4*A_sz);
        cudaMalloc((void**) &B_d1,4*B_sz);
        cudaMalloc((void**) &C_d1,4*C_sz);
        cudaMalloc((void**) &A_d2,4*(A_sz+1));//need to cover all 1000000 in vectors
        cudaMalloc((void**) &B_d2,4*(B_sz+1));
        cudaMalloc((void**) &C_d2,4*(C_sz+1));

    //cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    //startTime(&timer);

    //INSERT CODE HERE
	cudaMemcpyAsync(A_d0,A_h,4*A_sz,cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(B_d0,B_h,4*B_sz,cudaMemcpyHostToDevice,stream0);
	VecAdd<<<(A_sz-1)/BLOCK_SIZE+1,BLOCK_SIZE,0,stream0>>>(A_sz,A_d0,B_d0,C_d0);
	//cudaDeviceSynchronize();
	
	cudaMemcpyAsync(A_d1,A_h+A_sz,4*A_sz,cudaMemcpyHostToDevice,stream1);
        cudaMemcpyAsync(B_d1,B_h+B_sz,4*B_sz,cudaMemcpyHostToDevice,stream1);
	VecAdd<<<(A_sz-1)/BLOCK_SIZE+1,BLOCK_SIZE,0,stream1>>>(A_sz,A_d1,B_d1,C_d1);
	//cudaDeviceSynchronize();	

	cudaMemcpyAsync(A_d2,A_h+2*A_sz,4*(A_sz+1),cudaMemcpyHostToDevice,stream2);
        cudaMemcpyAsync(B_d2,B_h+2*B_sz,4*(B_sz+1),cudaMemcpyHostToDevice,stream2);
	VecAdd<<<(A_sz+1-1)/BLOCK_SIZE+1,BLOCK_SIZE,0,stream2>>>(A_sz+1,A_d2,B_d2,C_d2);
	//cudaDeviceSynchronize();

//	cudaStreamSynchronize(stream0);
//	cudaStreamSynchronize(stream1);
//	cudaStreamSynchronize(stream2);
    
//cudaDeviceSynchronize();
    //stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel  ---------------------------
    //printf("Launching kernel..."); fflush(stdout);
    //startTime(&timer);
    //cuda_ret = cudaDeviceSynchronize();
//	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    //stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // Copy device variables from host ----------------------------------------

    //printf("Copying data from device to host..."); fflush(stdout);
    //startTime(&timer);

    //INSERT CODE HERE
	cudaMemcpyAsync(C_h,C_d0,4*C_sz,cudaMemcpyDeviceToHost,stream0);
 	cudaMemcpyAsync(C_h+C_sz,C_d1,4*C_sz,cudaMemcpyDeviceToHost,stream1);
	cudaMemcpyAsync(C_h+2*C_sz,C_d2,4*(C_sz+1),cudaMemcpyDeviceToHost,stream2);

        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
    //cudaDeviceSynchronize();
    //stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    //printf("Verifying results..."); fflush(stdout);

    //verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);

    //INSERT CODE HERE
	cudaFree(A_d0);
	cudaFree(B_d0);
	cudaFree(C_d0);
	cudaFree(A_d1);
        cudaFree(B_d1);
        cudaFree(C_d1);
        cudaFree(A_d2);
        cudaFree(B_d2);
        cudaFree(C_d2);

    return 0;

}
