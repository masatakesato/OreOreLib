
/* Matrix multiplication: C = A * B.
 * Device code.
 */
 
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include	"Common_Cuda.cuh"


//#define GPU_PROFILING


// Thread block size
#define BLOCK_SIZE 32	// 注意! Aの行数/Bの列数は、BLOCK_SIZEで割り切れる必要がある
// CUDA5.0なら1024スレッドまで可能

float*	d_Dist;	// 距離値を格納する配列
int*	d_Idx;	// インデックス配列
unsigned int mem_size_Dist;
unsigned int mem_size_Idx;




//######################################################################################################//
//									cuda kernel prototype declaration									//
//######################################################################################################//

__global__ void nonsquare(float*M, float*N, float*P, int uWM,int uWN, int uWP );
__global__ void matrixMul( float* C, float* A, float* B, int a_rows, int a_cols, int b_rows, int b_cols, int c_rows, int c_cols );
__global__ void kNN_InsertionSort( float *dist, int *ind, int vecdim_offset, int numfeatures, int numelms, int k );

__global__ void kNN_ParallelReduction( float *dist, int *idx, int vecdim_offset, int numfeatures, int numbins, int k );


__global__ void cudaConstructHistogram( int *data, int *histo, int size );



//######################################################################################################//
//										entry points for c/c++											//
//######################################################################################################//

// 行列計算用のバッファを予め確保する
void AllocateMatrixBuffer( )
{


}


// 確保したバッファを開放する
void ReleaseMatrixBuffer()
{


}



void MatrixMultiplication(	float* h_A, unsigned int mem_size_A, int a_rows, int a_cols,
							float* h_B, unsigned int mem_size_B, int b_rows, int b_cols,
							float* h_C, unsigned int mem_size_C, int c_rows, int c_cols )
{

	// 8. allocate device memory
	float* d_A;
	float* d_B;
	MyCheckCudaErrors( cudaMalloc((void**) &d_A, mem_size_A) );
	MyCheckCudaErrors( cudaMalloc((void**) &d_B, mem_size_B) );

	// 9. copy host memory to device
	MyCheckCudaErrors( cudaMemcpy( d_A, h_A, mem_size_A, cudaMemcpyHostToDevice ) );
	MyCheckCudaErrors( cudaMemcpy( d_B, h_B, mem_size_B, cudaMemcpyHostToDevice ) );

	// 10. allocate device memory for the result
	float* d_C;
	MyCheckCudaErrors( cudaMalloc( (void**) &d_C, mem_size_C ) );

	// 5. perform the calculation
	// setup execution parameters
	dim3 threads( BLOCK_SIZE, BLOCK_SIZE );
	//dim3 grid( b_cols / threads.x, a_rows / threads.y );
	dim3 grid( DivUp( b_cols, BLOCK_SIZE ), DivUp( a_rows, BLOCK_SIZE ) );
	
	

#ifdef GPU_PROFILING
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );

#endif


	// execute the kernel
	//matrixMul<<< grid, threads >>>( d_C, d_A, d_B, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols );
	//__prod<<< grid, threads >>>( d_A, d_B, a_cols, b_cols, d_C );
	nonsquare<<< grid, threads >>>( d_A, d_B, d_C, a_cols, b_cols, c_cols );



#ifdef GPU_PROFILING

     cudaEventRecord( stop, 0 );
     cudaEventSynchronize( stop );
     
     cudaEventElapsedTime( &elapsed_time_ms, start, stop );
#endif

	// 11. copy result from device to host
	MyCheckCudaErrors( cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );


#ifdef GPU_PROFILING
	printf( "time: %8.6f ms\n", elapsed_time_ms );
#endif


	MyCheckCudaErrors( cudaFree(d_A) );
	MyCheckCudaErrors( cudaFree(d_B) );
	MyCheckCudaErrors( cudaFree(d_C) );


	cudaDeviceSynchronize();
}




// kNN計算用のデバイスメモリを確保する
void AllocateKnnBuffer( int max_num_vecs, int max_num_vws )
{
	mem_size_Dist	= max_num_vecs * max_num_vws * sizeof(float);
	mem_size_Idx	= max_num_vecs * max_num_vws * sizeof(int);
	
	MyCheckCudaErrors( cudaMalloc( (void**) &d_Dist, mem_size_Dist ) );
	MyCheckCudaErrors( cudaMalloc( (void**) &d_Idx, mem_size_Idx ) );
}


// kNN計算用のデバイスメモリを解放する
void ReleaseKnnBuffer()
{
	MyCheckCudaErrors( cudaFree( d_Dist ) );
	MyCheckCudaErrors( cudaFree( d_Idx ) );

	mem_size_Dist	= 0;
	mem_size_Idx	= 0;
}




// h_Dist.......行数: 特徴ベクトルの数, 列数: VisualWordsの数, 
// h_Idx........インデックス配列. VisualWordsのインデックスを格納
void kNN( float* h_Dist, int* h_Idx, int max_num_vecs, int max_num_vws,	int num_vecs, int num_vws, int k )
{
	//====================== デバイスメモリへデータを転送する ==================//
	MyCheckCudaErrors( cudaMemcpy( d_Dist, h_Dist, mem_size_Dist, cudaMemcpyHostToDevice ) );
	MyCheckCudaErrors( cudaMemcpy( d_Idx, h_Idx, mem_size_Idx, cudaMemcpyHostToDevice ) );


	//============================ ソートを実行する ============================//
	// スレッド数、グリッド数の設定
	dim3 blocks( DivUp( num_vecs, BLOCK_SIZE ), 1 );
	dim3 threads( BLOCK_SIZE, 1 );// スレッド1つに行1つを割り当てる
	


#ifdef GPU_PROFILING
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );

#endif

	// カーネル実行
	kNN_InsertionSort<<< blocks, threads >>>( d_Dist, d_Idx, max_num_vws, num_vecs, num_vws, k );


#ifdef GPU_PROFILING

     cudaEventRecord( stop, 0 );
     cudaEventSynchronize( stop );
     
     cudaEventElapsedTime( &elapsed_time_ms, start, stop );
#endif

	//====================== ホストメモリへ計算結果を転送する =====================//
	MyCheckCudaErrors( cudaMemcpy( h_Dist, d_Dist, mem_size_Dist, cudaMemcpyDeviceToHost ) );
	MyCheckCudaErrors( cudaMemcpy( h_Idx, d_Idx, mem_size_Idx, cudaMemcpyDeviceToHost ) );



#ifdef GPU_PROFILING
	printf( "time: %8.6f ms\n", elapsed_time_ms );
#endif

	
	cudaDeviceSynchronize();
}






void kNN_Parallel( float* h_Dist, int* h_Idx, int max_num_vecs, int max_num_vws, int num_vecs, int num_vws, int k )
{
	//======================== デバイスメモリを確保する ========================//
	float*	d_Dist;	// 距離値を格納する配列
	int*	d_Idx;	// インデックス配列
	
	unsigned int mem_size_Dist	= max_num_vecs * max_num_vws * sizeof(float);
	unsigned int mem_size_Idx	= max_num_vecs * max_num_vws * sizeof(int);
	
	MyCheckCudaErrors( cudaMalloc( (void**) &d_Dist, mem_size_Dist ) );
	MyCheckCudaErrors( cudaMalloc( (void**) &d_Idx, mem_size_Idx ) );

	
	//====================== デバイスメモリへデータを転送する ==================//
	MyCheckCudaErrors( cudaMemcpy( d_Dist, h_Dist, mem_size_Dist, cudaMemcpyHostToDevice ) );
	MyCheckCudaErrors( cudaMemcpy( d_Idx, h_Idx, mem_size_Idx, cudaMemcpyHostToDevice ) );


	//============================ ソートを実行する ============================//
	// スレッド数、グリッド数の設定
	dim3 blocks( num_vecs, 1 );	// ブロック1つに1行割り当てる
	dim3 threads( num_vws, 1 );// スレッド1つに1要素を割り当てる
	

	
#ifdef GPU_PROFILING
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
#endif

	// カーネル実行
	kNN_ParallelReduction<<< blocks, threads >>>( d_Dist, d_Idx, max_num_vws, num_vecs, num_vws, k );


#ifdef GPU_PROFILING

     cudaEventRecord( stop, 0 );
     cudaEventSynchronize( stop );
     
     cudaEventElapsedTime( &elapsed_time_ms, start, stop );
#endif
	 
	//====================== ホストメモリへ計算結果を転送する =====================//
	MyCheckCudaErrors( cudaMemcpy( h_Dist, d_Dist, mem_size_Dist, cudaMemcpyDeviceToHost ) );
	MyCheckCudaErrors( cudaMemcpy( h_Idx, d_Idx, mem_size_Idx, cudaMemcpyDeviceToHost ) );
	

	//======================= デバイスメモリを解放する ============================//
	MyCheckCudaErrors( cudaFree( d_Dist ) );
	MyCheckCudaErrors( cudaFree( d_Idx ) );


//#ifdef GPU_PROFILING
//	printf( "time: %8.6f ms\n", elapsed_time_ms );
//#endif
	
	
	cudaDeviceSynchronize();
}











//######################################################################################################//
//										cuda kernel implementation										//
//######################################################################################################//


// 非正方行列の乗算
__global__ void nonsquare(float*M, float*N, float*P, int uWM,int uWN, int uWP )
{
	__shared__ float MS[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float NS[BLOCK_SIZE][BLOCK_SIZE];


	int tx=threadIdx.x, ty=threadIdx.y, bx=blockIdx.x, by=blockIdx.y;
	int rowM=ty+by*BLOCK_SIZE;
	int colN=tx+bx*BLOCK_SIZE;
	float Pvalue=0;


	for(int m=0; m< uWM/BLOCK_SIZE;++m)
	{
		MS[ty][tx] = M[rowM*uWM+(m*BLOCK_SIZE+tx)];
		NS[ty][tx] = N[colN + uWN*(m*BLOCK_SIZE+ty)];

		__syncthreads();

		for(int k=0;k<BLOCK_SIZE;k++)
		{
			float d	= MS[ty][k] - NS[k][tx];
			Pvalue += d*d;//MS[ty][k]*NS[k][tx];//
		}

		__syncthreads();

		P[rowM * uWP + colN]=Pvalue;
	}

	for( int m=0; m< uWM/BLOCK_SIZE; ++m )
	{
		int colM = m*BLOCK_SIZE+tx;
		int rowN = m*BLOCK_SIZE+ty;

		if( rowM > uWN || rowN > uWM || colM > uWM || colN > uWN )
		{
			MS[ty][tx]=0.;
			NS[ty][tx]=0.;
		}
		else
		{
			MS[ty][tx]=M[rowM*uWM+colM];
			NS[ty][tx]=N[colN + uWN*rowN];
		}

	}
}


//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//////////////////////////////////////////////////////
__global__ void matrixMul( float* C, float* A, float* B, int a_rows, int a_cols, int b_rows, int b_cols, int c_rows, int c_cols )
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;// 対応する行列の要素番号
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if( row>=c_rows || col>=c_cols )	return;
	
    // Block index. サブマトリクスのインデックス
	int bcol = blockIdx.x;// サブマトリクスインデックス(y方向)
	int brow = blockIdx.y;// サブマトリクスインデックス(x方向)

 
    // Thread index(0～BLOCK_SIZE-1). 部分領域内の要素のインデックス
	int tcol = threadIdx.x;
	int trow = threadIdx.y;
 
    // Index of the first sub-matrix of A processed 
    // by the block
	// Aの列方向の要素数 * Aの行方向の要素数 * 横方向のサブマトリクスインデックス 
	int aBegin = a_cols * BLOCK_SIZE * brow;// 行列Aのサブマトリクスの開始インデックス（行ベクトル）常に行列の左端にある

    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + a_cols - 1;		// 行列Aのサブマトリクス終了インデックス（行ベクトル）

    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;
 
    // Index of the first sub-matrix of B processed 
    // brow the block
    int bBegin = BLOCK_SIZE * bcol;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * b_cols;

	float Csub	= 0.0f;
	
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int i=0, int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep, ++i) 
    {

        // Declaration of the shared memory array As 
        // used to store the sub-matrix of A
        __shared__ float As[ BLOCK_SIZE ][ BLOCK_SIZE ];
 
        // Declaration of the shared memory array Bs 
        // used to store the sub-matrix of B
        __shared__ float Bs[ BLOCK_SIZE ][ BLOCK_SIZE ];
 

        // Load the matrices from global memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[trow][tcol] = A[ a + a_cols * trow + tcol ];
        Bs[trow][tcol] = B[ b + b_cols * trow + tcol ];

        // Synchronize to make sure the matrices 
        // are loaded
        __syncthreads();
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for( int k = 0; k < BLOCK_SIZE; ++k )
		{
			float d	= As[trow][k] - Bs[k][tcol];
            Csub += d*d;//As[trow][k] * Bs[k][tcol];//
		}
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
 
		
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = b_cols * BLOCK_SIZE * brow + BLOCK_SIZE * bcol;
    C[ c + b_cols * trow + tcol ] = Csub;
	
}





// 挿入ソートを用いて、配列の上位k個の最小値を探す。
// スレッド1つにつき特徴ベクトル1個分のkNNを実行する
//
__global__ void kNN_InsertionSort( float *dist, int *ind, int vecdim_offset, int numfeatures, int numelms, int k )
{
	// Variables
	int l, i, j;
    float *p_dist;
	int   *p_ind;
    float curr_dist, max_dist;
    int   curr_row,  max_row;
	const int vecIndex = blockIdx.x * blockDim.x + threadIdx.x;// ) * numrows;
	

//	__shared__ float	s_dist[];
//	__shared__ int		s_idx[];



	if( vecIndex >= numfeatures )	return;
 
//printf( " threadIdx = %d\n", vecIndex );
   
	// Pointer shift, initialization, and max value
	p_dist   = dist + vecIndex * vecdim_offset;
	p_ind    = ind + vecIndex * vecdim_offset;
	max_dist = p_dist[0];
	p_ind[0] = 0;


	//===================== 配列の最初のk個を昇順にソートする =====================//
	for( l=1; l<k; ++l )
	{
		curr_row  = l;
		curr_dist = p_dist[l];

		if( curr_dist < max_dist )
		{
			i = l-1;
			for( int a=0; a<l-1; ++a )
			{
				if( p_dist[a] > curr_dist )
				{
					i = a;
					break;
				}
			}// end of a loop

			for( j=l; j>i; --j )
			{
				p_dist[j]	= p_dist[j-1];
				p_ind[j]	= p_ind[j-1];
			}
			p_dist[i]	= curr_dist;
			p_ind[i]	= l;
		}
		else
			p_ind[l] = l;

		max_dist = p_dist[curr_row];
	}


	//========================= 上位k個の最小距離値を見つける ========================//
	max_row = (k-1);
	for( l=k; l<numelms; ++l )
	{
		curr_dist = p_dist[l];
		if( curr_dist<max_dist )
		{
			i = k-1;
			for( int a=0; a<k-1; ++a )
			{
				if( p_dist[a] > curr_dist )
				{
					i = a;
					break;
				}
			}// end of a loop
			for( j=k-1; j>i; --j )
			{
				p_dist[j]	= p_dist[j-1];
				p_ind[j]	= p_ind[j-1];
			}
			p_dist[i]	= curr_dist;
			p_ind[i]	= l;
			max_dist	= p_dist[max_row];
		}
	}// end of l loop


}// end of method










// 並列リダクションを使ったkNN.
// ブロック毎に特徴ベクトル1個を割り当てる
// スレッド毎に、スカラー値同士のより距離が近いVisualWordsを絞り込む
// dim3 blocks( );
// dim3 threads( numVWs, 1 );
#define NUM_MAX_VW 1024


__global__ void kNN_ParallelReduction( float *dist, int *idx, int vecdim_offset, int numfeatures, int numbins, int k )
{
	int tidx	= threadIdx.x;	// VisualWordsの通し番号インデックス
	int gidx	= blockIdx.x * vecdim_offset + tidx;// スレッドID取得 ( )

	
	//==================== 特徴ベクトル1個と各VWとの距離をコピーする ========================//
	__shared__ float	s_dist[ NUM_MAX_VW ];
	__shared__ int		s_idx[ NUM_MAX_VW ];


	if( threadIdx.x < 24  )// 無効なシェアードメモリ(1000～1023)を初期値で埋めとく
	{
		s_dist[ numbins + tidx ]	= FLT_MAX;
		s_idx[ numbins + tidx ]		= -1;
	}


	s_dist[ tidx ]	= dist[ gidx ];	// スレッド毎に距離値をコピーする
	s_idx[ tidx ]	= tidx;// VisualWordsのインデックス(通し番号)を保存するidx[ gidx ];

	__syncthreads();// 全スレッドが値をコピーし終わるのを待つ


	//==================== 並列リダクションでk個付近まで候補を絞り込む ======================//
	int s;
	for( s = NUM_MAX_VW/2; s>=k; s>>=1 )
	{
		if( tidx < s )
		{
			int tidx2 = tidx + s;
			if( s_dist[ tidx ] > s_dist[ tidx2 ] )
			{
				s_dist[ tidx ]	= s_dist[ tidx2 ];
				s_idx[ tidx ]	= s_idx[ tidx2 ];
			}
		}
	
		__syncthreads();
	}
	
//printf("%d\n", s );
	
	//========================== 絞り込んだ残りの要素をソートする ===========================//
	if( tidx == 0 )
	{
		int i, j, l;
		int curr_idx;
		float curr_dist;
		float max_dist = s_dist[0];
s<<=1;
		for( l=1; l<s; ++l )
		{
			curr_idx  = s_idx[l];
			curr_dist = s_dist[l];

			if( curr_dist < max_dist )// l番目の距離値curr_distがmax_distよりも小さかった場合
			{
				i = l-1;// l-1番目の要素
				for( int a=0; a<l-1; ++a )// 0～l-1番目の要素の中で、curr_distを挿入する位置[i]を探す
				{
					if( s_dist[a] > curr_dist )
					{
						i = a;
						break;
					}
				}// end of a loop

				for( j=l; j>i; --j )// i番目を空けるため、要素を1個ずつ後ろにずらす
				{
					s_dist[j]	= s_dist[j-1];
					s_idx[j]	= s_idx[j-1];
				}
				s_dist[i]	= curr_dist;// [i]番目に、curr_distを格納する
				s_idx[i]	= curr_idx; //s_idx[l];	// [i]番目にcurr_idxを格納する
			}
			//else
			//	s_idx[l] = curr_idx;//s_idx[l];

			max_dist = s_dist[curr_idx];
		}


		for( i=0; i<k; ++i )
		{
			dist[ gidx + i ]	= s_dist[ tidx + i ];
			idx[ gidx + i ]		= s_idx[ tidx + i ];
		}


	}// end of if tidx.x==0

}







































#define BLOCK_SIZE_BIN	1024

// ヒストグラムを作成する
// 入力データ: DistanceMatrix, ヒストグラム配列
void ConstructHistogram( int numData, int numBins )
{
	int *h_data, *h_histo;

	h_data = new int[numData];
	h_histo	= new int[numBins];


	for(int i=0; i<numData; ++i)
	{
		h_data[i] = numBins-1;// バンクコンフリクト発生率最大のケースを想定した値設定
		//h_data[i] = rand()%(numBins-1);//
	}

	for(int i=0; i<numBins; ++i)
	{
		h_histo[i] = 0;	// 全部インデックスゼロに累積される
	}


	

	int *d_data, *d_histo;
	int size_data	= numData * sizeof(int);
	int size_histo	= numBins * sizeof(int);

	cudaMalloc( (void**) &d_data, size_data );
	cudaMalloc( (void**) &d_histo, size_histo );


	cudaMemcpy( d_data, h_data, size_data, cudaMemcpyHostToDevice );
	cudaMemcpy( d_histo, h_histo, size_histo, cudaMemcpyHostToDevice );


	// buffer: シェアードメモリに収まる単位に小分けする
	// histogram: これもシェアードメモリに収まる範囲に小分けする
	dim3 blocks( DivUp(numBins, BLOCK_SIZE_BIN), 1);// 
	dim3 threads( BLOCK_SIZE_BIN, 1 );

	

#ifdef GPU_PROFILING
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );

#endif

	cudaConstructHistogram <<< blocks, threads >>>( d_data, d_histo, numData );


#ifdef GPU_PROFILING

     cudaEventRecord( stop, 0 );
     cudaEventSynchronize( stop );
     
     cudaEventElapsedTime( &elapsed_time_ms, start, stop );
#endif


	// デバイスメモリからホストメモリへ結果転送
	cudaMemcpy( h_histo, d_histo, size_histo, cudaMemcpyDeviceToHost );


for( int i=0; i<numBins; ++i )
{
	printf( "[%d]: %d\n", i, h_histo[i] );

}
#ifdef GPU_PROFILING
	printf( "time: %8.6f ms\n", elapsed_time_ms );
#endif


	// 後片付け
	cudaFree( d_data );
	cudaFree( d_histo );

	delete [] h_data;
	delete [] h_histo;

}






// bin毎のスレッド
__global__ void cudaConstructHistogram( int *data, int *histo, int data_size )
{
	int tidx_bin	= threadIdx.x;		// シェアードメモリのヒストグラムインデックス
	int gidx_bin	= BLOCK_SIZE_BIN * blockIdx.x + tidx_bin;	// グローバルメモリのヒストグラムインデックス


	//=================== シェアードメモリに必要なデータをアップロード ===============//
	__shared__ int		s_bin[ 1024 ];	// ヒストグラムは分割せずに次元数ぶん確保する

	s_bin[gidx_bin]	= 0;
	__syncthreads();
	
	
	//======================== 該当するbinにデータを加算する =========================//
	for( int s=tidx_bin; s<data_size; s+=BLOCK_SIZE_BIN )
	{
		//printf("%d, ", data[ data_idx ] );
		atomicAdd( &s_bin[ data[ s ] ], 1 );
	}

	__syncthreads();// 全部スレッドが終了するまで待つ

	
	//======================== グローバルメモリに累積する ============================//
	atomicAdd( &histo[ gidx_bin ], s_bin[ gidx_bin ] );
}





// threadIdx.x, ヒストグラムのインデックス
// size
//__global__ void CalcHistogram( unsigned char *buffer, unsigned int *histo, long size )
//{
//	//=================== ヒストグラムの途中結果を作る ===================//
//	__shared__ float temp[256];
//	
//	__syncthreads();
//
//
//	int i		= threadIdx.x + blockIdx.x * blockDim.x;// bufferのインデックス, 
//	int offset	= blockDim.x * gridDim.x;				// buffer配列上での開始位置.ブロック毎にユニーク
//
//	while( i<size )
//	{
//		atomicAdd( &temp[ buffer[i] ], 1 );
//		i += offset;
//	}
//	__syncthreads();
//
//
//	atomicAdd( &(histo[ threadIdx.x ]), temp[threadIdx.x] );// グローバルメモリに書き戻す
//}
//






#endif