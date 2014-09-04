/* cudaErr.cu  
 *
 * contains simple function for reporting GPU errors
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cudaErr.h"

void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }                         
}
