#ifndef CUDA_SHUFFLE_SCAN_H
#define	CUDA_SHUFFLE_SCAN_H


//######################################################################################//
//									My library include									//
//######################################################################################//

#include	<oreore/MathLib.h>
#include	<oreore/cudaHelper.h>


//######################################################################################//
//										CUDA Kernels									//
//######################################################################################//


// TODO: 100万個以上の要素数にも耐えられるようにしたい. 2013.12.22
// 要素の総数が65536個を超えるとおかしくなる -> 


// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call, then
// uniformly adding across the input data via the uniform_add<<<>>> kernel.

__global__ void shfl_scan_test( int *data, int width, int *partial_sums=NULL )
{
	extern __shared__ int sums[];
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = id % warpSize;
	// determine a warp_id within a block
	int warp_id = threadIdx.x / warpSize;

	// Below is the basic structure of using a shfl instruction
	// for a scan.
	// Record "value" as a variable - we accumulate it along the way
	int value = data[id];

	// Now accumulate in log steps up the chain
	// compute sums, with another thread's value who is
	// distance delta away (i).  Note
	// those threads where the thread 'i' away would have
	// been out of bounds of the warp are unaffected.  This
	// creates the scan sum.
#pragma unroll
	
	// "<=" から "<" へ変更。結果は変化なし
	for (int i=1; i<=width; i*=2)
	{
		int n = __shfl_up(value, i, width);
		if( lane_id >= i ) value += n;
	}

	// value now holds the scan value for the individual thread
	// next sum the largest values for each warp

	// write the sum of the warp to smem
	if (threadIdx.x % warpSize == warpSize-1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	//
	// scan sum the warp sums
	// the same shfl scan operation, but performed on warp sums
	//
	if( warp_id == 0 )
	{
		int warp_sum = sums[lane_id];

		// "<=" から "<" へ変更。結果は変化なし
		for (int i=1; i<=width; i*=2)
		{
			int n = __shfl_up(warp_sum, i, width);
			if (lane_id >= i) warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// perform a uniform add across warps in the block
	// read neighbouring warp's sum and add it to threads value
	int blockSum = 0;

	if (warp_id > 0)
	{
		blockSum = sums[warp_id-1];
	}

	value += blockSum;

	// Now write out our result
	data[id] = value;

	// last thread has sum, write write out the block's sum
	if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
	{
		partial_sums[blockIdx.x] = value;
	}
}



// Uniform add: add partial sums array
__global__ void uniform_add(int *data, int *partial_sums, int len)
{
	__shared__ int buf;
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (id > len) return;

	if (threadIdx.x == 0)
	{
		buf = partial_sums[blockIdx.x];
	}

	__syncthreads();
	data[id] += buf;
}






// ShuffleScanの結果が正しいかどうかチェックする関数
bool CPUverify(int *h_data, int *h_result, int n_elements)
{
	// cpu verify
	for (int i=0; i<n_elements-1; i++)
	{
		h_data[i+1] = h_data[i] + h_data[i+1];
	}

	int diff = 0;

	for (int i=0 ; i<n_elements; i++)
	{
		diff += h_data[i]-h_result[i];
	}

	printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
	bool bTestResult = false;

	if (diff == 0) bTestResult = true;

	//StopWatchInterface *hTimer = NULL;
	//sdkCreateTimer(&hTimer);
	//sdkResetTimer(&hTimer);
	//sdkStartTimer(&hTimer);

	//for (int j=0; j<100; j++)
	//	for (int i=0; i<n_elements-1; i++)
	//	{
	//		h_data[i+1] = h_data[i] + h_data[i+1];
	//	}

		//sdkStopTimer(&hTimer);
		//double cput= sdkGetTimerValue(&hTimer);
		//printf("CPU sum (naive) took %f ms\n", cput/100);
		return bTestResult;
}





//// プログラムの改良が必要
//
//bool ShuffleScan( int *h_data, int *h_result, int n_elements )
//{
//	int /**h_data,*/ *h_partial_sums/*, *h_result*/;
//	int *d_data, *d_partial_sums;
//	//const int n_elements = 65536;
//	int sz = sizeof(int)*n_elements;
//	//int cuda_device = 0;
//
//	printf("Starting shfl_scan\n");
//
//	//cudaMallocHost((void **)&h_data, sizeof(int)*n_elements);//checkCudaErrors(cudaMallocHost((void **)&h_data, sizeof(int)*n_elements));
//	//cudaMallocHost((void **)&h_result, sizeof(int)*n_elements);//checkCudaErrors(cudaMallocHost((void **)&h_result, sizeof(int)*n_elements));
//
//	//initialize data:
//	printf("Computing Simple Sum test\n");
//	printf("---------------------------------------------------\n");
//
//	printf("Initialize test data [1, 1, 1...]\n");
//
//
//	int blockSize		= 1024;	// n_elementsを65536以上に対応したいなら、ブロックサイズを大きくするしかない。
//	int gridSize		= DivUp( n_elements, blockSize );// n_elements/blockSize;//	// 要素数がblockSize未満の場合にも対応するよう変更
//	int nWarps			= DivUp( blockSize, 32 );// blockSize/32
//	int shmem_sz		= nWarps * sizeof(int);	// シェアードメモリのサイズ
//	int n_partialSums	= DivUp( n_elements, blockSize );// n_elements/blockSize;	//
//	int partial_sz		= n_partialSums*sizeof(int);
//
//	printf("Scan summation for %d elements, %d partial sums\n",
//		n_elements, n_elements/blockSize);
//
//	int p_blockSize = min( n_partialSums, blockSize );
//	int p_gridSize = DivUp(n_partialSums, p_blockSize);
//	printf("Partial summing %d elements with %d blocks of size %d\n",
//		n_partialSums, p_gridSize, p_blockSize);
//
//	// initialize a timer
//	cudaEvent_t start, stop;
//	CudaSafeCall( cudaEventCreate(&start) );
//	CudaSafeCall( cudaEventCreate(&stop) );
//	float et = 0;
//	float inc = 0;
//
//	CudaSafeCall( cudaMalloc((void **)&d_data, sz) );
//	CudaSafeCall( cudaMalloc((void **)&d_partial_sums, partial_sz) );
//	CudaSafeCall( cudaMemset(d_partial_sums, 0, partial_sz) );
//
//	CudaSafeCall( cudaMallocHost((void **)&h_partial_sums, partial_sz) );
//	CudaSafeCall( cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice) );
//
//	CudaSafeCall( cudaEventRecord(start, 0) );
//
//
//	//############################# PrefixSumのカーネル ################################//
//	shfl_scan_test<<<gridSize,blockSize, shmem_sz>>>( d_data, 32, d_partial_sums );// ここまではうまくいっている
//
//	shfl_scan_test<<<p_gridSize,p_blockSize, shmem_sz>>>( d_partial_sums, 32 );// 65536超えるとおかしくなる. shmem_sz関係ない, p_blockSize:??? p_gridSize:???
//	uniform_add<<<gridSize-1, blockSize>>>(d_data+blockSize, d_partial_sums, n_elements);
//	//##################################################################################//
//
//
//	CudaSafeCall( cudaEventRecord(stop, 0) );
//	CudaSafeCall( cudaEventSynchronize(stop) );
//	CudaSafeCall( cudaEventElapsedTime(&inc, start, stop) );
//	et+=inc;
//
//	CudaSafeCall( cudaMemcpy( h_result, d_data, sz, cudaMemcpyDeviceToHost ) );
//	CudaSafeCall( cudaMemcpy( h_partial_sums, d_partial_sums, partial_sz, cudaMemcpyDeviceToHost ) );
//
//	printf("Test Sum: %d\n", h_partial_sums[n_partialSums-1]);
//	printf("Time (ms): %f\n", et);
//	printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
//		n_elements, et, n_elements/(et/1000.0f)/1000000.0f);
//
//
//
//	// TODO: デバッグ目的で必要
//	bool bTestResult = CPUverify(h_data, h_result, n_elements);
//
//	//cudaFreeHost(h_data);//checkCudaErrors(cudaFreeHost(h_data));
//	//cudaFreeHost(h_result);//checkCudaErrors(cudaFreeHost(h_result));
//	CudaSafeCall( cudaFreeHost(h_partial_sums) );
//	CudaSafeCall( cudaFree(d_data) );
//	CudaSafeCall( cudaFree(d_partial_sums) );
//
//	return bTestResult;
//}


#endif // !CUDA_SHUFFLE_SCAN_H