#include	<cuda_runtime.h>

#include	<stdio.h>


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif


#ifndef THREAD_NUM_X
#define	THREAD_NUM_X	16
#endif

#ifndef THREAD_NUM_Y
#define	THREAD_NUM_Y	16
#endif


#ifndef DivUp
#define DivUp(a, b)( ((a%b)==0)?(a/b):(a/b+1) )
#endif

#ifndef	M_PI
#define	M_PI 3.14159265358979323846
#endif

#ifndef	_2_M_PI
#define	_2_M_PI	3.14159265358979323846 * 2.0
#endif



// オリエンテーションの探査方向の数
#ifndef	ORI_SEARCH_BINS
#define ORI_SEARCH_BINS 192	// 96	// 42
#endif

// オリエンテーションの角度レンジ
#ifndef	ORI_WINDOW_RNGE
#define	ORI_WINDOW_RNGE	M_PI/3.0
#endif

// 探査方向インクリメント毎の角度変化量
#ifndef	ORI_INC
#define	ORI_INC	2.0*M_PI / ORI_SEARCH_BINS
#endif




// Orientation算出用サンプル点(円形領域内の109個)の相対x座標.
static const int coord_x[109] = {
	-5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
	-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,
	 0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
	 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5
};

// Orientation算出用サンプル点(円形領域内の109個)の相対y座標.
static const int coord_y[109] = {
	-3, -2, -1,  0,  1,  2,  3, -4, -3, -2, -1,  0,  1,  2,  3,  4, -5, -4, -3, -2, -1, 0, 1,
	 2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0, 1, 2,
	 3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1, 2, 3, 4,
	 5, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5, -5, -4, -3, -2, -1,  0,  1,  2,  3, 4, 5,
	-4, -3, -2, -1,  0,  1,  2,  3,  4, -3, -2, -1,  0,  1,  2,  3
};


// Orientation算出用サンプル点(円形領域内の109個)のガウス関数値(σ=2.5)
static const float gauss_lin[109] =
{
	0.000958195f, 0.00167749f, 0.00250251f, 0.00318132f, 0.00250251f, 0.00167749f, 0.000958195f, 0.000958195f, 0.00196855f, 0.00344628f, 0.00514125f, 0.00653581f, 0.00514125f, 0.00344628f, 0.00196855f, 0.000958195f, 0.000695792f, 0.00167749f,
	0.00344628f, 0.00603331f, 0.00900064f, 0.0114421f, 0.00900064f, 0.00603331f, 0.00344628f, 0.00167749f, 0.000695792f, 0.001038f, 0.00250251f, 0.00514125f, 0.00900064f, 0.0134274f, 0.0170695f, 0.0134274f, 0.00900064f, 0.00514125f, 0.00250251f,
	0.001038f, 0.00131956f, 0.00318132f, 0.00653581f, 0.0114421f, 0.0170695f, 0.0216996f, 0.0170695f, 0.0114421f, 0.00653581f, 0.00318132f, 0.00131956f, 0.00142946f, 0.00344628f, 0.00708015f, 0.012395f, 0.0184912f, 0.0235069f, 0.0184912f, 0.012395f,
	0.00708015f, 0.00344628f, 0.00142946f, 0.00131956f, 0.00318132f, 0.00653581f, 0.0114421f, 0.0170695f, 0.0216996f, 0.0170695f, 0.0114421f, 0.00653581f, 0.00318132f, 0.00131956f, 0.001038f, 0.00250251f, 0.00514125f, 0.00900064f, 0.0134274f,
	0.0170695f, 0.0134274f, 0.00900064f, 0.00514125f, 0.00250251f, 0.001038f, 0.000695792f, 0.00167749f, 0.00344628f, 0.00603331f, 0.00900064f, 0.0114421f, 0.00900064f, 0.00603331f, 0.00344628f, 0.00167749f, 0.000695792f, 0.000958195f, 0.00196855f,
	0.00344628f, 0.00514125f, 0.00653581f, 0.00514125f, 0.00344628f, 0.00196855f, 0.000958195f, 0.000958195f, 0.00167749f, 0.00250251f, 0.00318132f, 0.00250251f, 0.00167749f, 0.000958195f
};



// Surf特徴ベクトル計算用のガウス関数値(σ=3.3)
static const float gauss33[12][12] =
{
	0.014614763f,0.013958917f,0.012162744f,0.00966788f,0.00701053f,0.004637568f,0.002798657f,0.001540738f,0.000773799f,0.000354525f,0.000148179f,0.0f,
	0.013958917f,0.013332502f,0.011616933f,0.009234028f,0.006695928f,0.004429455f,0.002673066f,0.001471597f,0.000739074f,0.000338616f,0.000141529f,0.0f,
	0.012162744f,0.011616933f,0.010122116f,0.008045833f,0.005834325f,0.003859491f,0.002329107f,0.001282238f,0.000643973f,0.000295044f,0.000123318f,0.0f,
	0.00966788f,0.009234028f,0.008045833f,0.006395444f,0.004637568f,0.003067819f,0.001851353f,0.001019221f,0.000511879f,0.000234524f,9.80224E-05f,0.0f,
	0.00701053f,0.006695928f,0.005834325f,0.004637568f,0.003362869f,0.002224587f,0.001342483f,0.000739074f,0.000371182f,0.000170062f,7.10796E-05f,0.0f,
	0.004637568f,0.004429455f,0.003859491f,0.003067819f,0.002224587f,0.001471597f,0.000888072f,0.000488908f,0.000245542f,0.000112498f,4.70202E-05f,0.0f,
	0.002798657f,0.002673066f,0.002329107f,0.001851353f,0.001342483f,0.000888072f,0.000535929f,0.000295044f,0.000148179f,6.78899E-05f,2.83755E-05f,0.0f,
	0.001540738f,0.001471597f,0.001282238f,0.001019221f,0.000739074f,0.000488908f,0.000295044f,0.00016243f,8.15765E-05f,3.73753E-05f,1.56215E-05f,0.0f,
	0.000773799f,0.000739074f,0.000643973f,0.000511879f,0.000371182f,0.000245542f,0.000148179f,8.15765E-05f,4.09698E-05f,1.87708E-05f,7.84553E-06f,0.0f,
	0.000354525f,0.000338616f,0.000295044f,0.000234524f,0.000170062f,0.000112498f,6.78899E-05f,3.73753E-05f,1.87708E-05f,8.60008E-06f,3.59452E-06f,0.0f,
	0.000148179f,0.000141529f,0.000123318f,9.80224E-05f,7.10796E-05f,4.70202E-05f,2.83755E-05f,1.56215E-05f,7.84553E-06f,3.59452E-06f,1.50238E-06f,0.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f
};




#include	"SurfDescriptor.h"
#include	"Common_Cuda.cuh"



__constant__ int	g_TexWidth;
__constant__ int	g_TexHeight;

__constant__ int	dc_coord_x[109];
__constant__ int	dc_coord_y[109];
__constant__ float	dc_gauss_lin[109];
__constant__ float	dc_gauss33[12][12];
//__device__  float	d_featurevecs[4096*384];	// デバイスメモリ上のSURF特徴ベクトル. 本来は特徴ベクトル数に合わせて動的に確保したい
__device__  float	d_featurevecs[ 67108864 ];



surface< void, cudaSurfaceType2D >	surfRef_in;
surface< void, cudaSurfaceType2D >	surfIntegralImage;
texture< float4, cudaTextureType2D, cudaReadModeElementType >	g_texIntegralImage;

surface< void, cudaSurfaceType2D >	surfRef_out;
texture< float4, cudaTextureType2D, cudaReadModeElementType >	g_TexIn;	// texture 専用



int				g_numMaxPoints;
pKeypoint		*d_ipoints;	// デバイスメモリ上のSURFキーポイント






// IntegralImage kernel
__global__ void RGB2Luminance( unsigned int nWidth, unsigned int nHeight );
__global__ void IntegralImage_h( unsigned int nWidth );
__global__ void IntegralImage_v( unsigned int nWidth );
__global__ void HaarWavelet( unsigned int nWidth, unsigned int nHeight );
__global__ void CalcSurfOrientation( pKeypoint* g_ipoints, int numPoints );

__global__ void CalcFeatureDescriptors( pKeypoint* g_ipoints, int numPoints, int numLevels, int points_per_level, bool bUpRight, bool bInheritOrientation );

__global__ void CalcFeatureDescriptorsExt( pKeypoint* g_ipoints, int numPoints );
__global__ void CalcFeatureDescriptorsExt_InheritOrientation( pKeypoint* g_ipoints, int numPoints, int numLevels, int points_per_level, bool bUpRight );

__global__ void NormalizeFeatureDescriptors( pKeypoint* g_ipoints );


__global__ void SampleTexture( unsigned int nWidth, unsigned int nHeight );


__device__ float getAngle( float X, float Y );
__device__ float BoxIntegral( float *data, int width, int height, size_t widthStep, int row, int col, int rows, int cols );
__device__ void haarXY( int x, int y, int roundedScale, float *xResponse, float *yResponse, float gauss );
__device__  void haarXY_float( float x, float y, float Scale, float *xResponse, float *yResponse, float gauss );




//##########################################################################################################//
//												Entry Points												//
//##########################################################################################################//



int CudaSURF_InitLookupTable()
{
	// テーブルを登録する. デバイスメモリを文字列で指定するのはCUDA5.0で廃止になっている
	MyCheckCudaErrors( cudaMemcpyToSymbol( dc_coord_x, coord_x, sizeof(coord_x) ) );
	MyCheckCudaErrors( cudaMemcpyToSymbol( dc_coord_y, coord_y, sizeof(coord_y) ) );
	MyCheckCudaErrors( cudaMemcpyToSymbol( dc_gauss_lin, gauss_lin, sizeof(gauss_lin) ) );
	MyCheckCudaErrors( cudaMemcpyToSymbol( dc_gauss33, gauss33, sizeof(gauss33) ) );


	g_texIntegralImage.addressMode[0]	= cudaAddressModeMirror;//cudaAddressModeClamp;//
	g_texIntegralImage.addressMode[1]	= cudaAddressModeMirror;//cudaAddressModeClamp;//
	g_texIntegralImage.addressMode[2]	= cudaAddressModeMirror;//cudaAddressModeClamp;//
	g_texIntegralImage.filterMode		= cudaFilterModeLinear;//cudaFilterModePoint;//
	g_texIntegralImage.normalized		= false;

	return 0;
}



int CudaSURF_AllocateDescBuffer( int numMaxPoints )
{
	//================= キーポイントをデバイスメモリ上に確保する =================//
	g_numMaxPoints	= numMaxPoints;
	size_t size	= sizeof(pKeypoint) * g_numMaxPoints;

	if( MyCheckCudaErrors( cudaMalloc( (void**)&d_ipoints, size ) ) )
		return 1;


	return 0;
}


// グローバルなデバイスメモリに対してcudamallocができない.固定長配列の宣言で対処.本メソッドは一旦使用禁止.2013.09.13
//int CudaSURF_AllocateFeatureBuffer( int size )
//{
//	//=============== 特徴ベクトル行列をデバイスメモリ上に確保する ===============//
//	//size_t size	= numMaxPoints * MaxVecDim * sizeof(float);
//	//if( MyCheckCudaErrors( cudaMalloc( (void**)&d_vecs, size ) ) )
//	//	return 1;
//
//	//if( MyCheckCudaErrors( cudaMemcpyToSymbol( d_featurevecs, d_vecs, size) ) )
//	//	return 1;
//
//	return 0;
//}



// 入力画像のCudaArrayにcudaSurfaceをバインドする
int CudaSURF_BindTexture( cudaArray* cuArray_in, cudaArray* cuArray_intImg, int width, int height )
{
	//========================= 画像のピクセル数をコンスタント変数に登録する ===========================//
	MyCheckCudaErrors( cudaMemcpyToSymbol( g_TexWidth, &width, sizeof(width) ) );
	MyCheckCudaErrors( cudaMemcpyToSymbol( g_TexHeight, &height, sizeof(height) ) );


	//=========================== cuArray_inをcudaSurfaceにバインドする ================================//
	if( MyCheckCudaErrors( cudaBindSurfaceToArray( surfRef_in, cuArray_in ) ) )
		return 1;

	//========================= IntegralImageをcudaSurfaceにバインドする ===========================//
	if( MyCheckCudaErrors( cudaBindSurfaceToArray( surfIntegralImage, cuArray_intImg ) ) )//, channelDesc );
		return 1;

	//========================== IntegralImageをcudaTextureにバインドする =============================//
	if( MyCheckCudaErrors( cudaBindTextureToArray( g_texIntegralImage, cuArray_intImg ) ) )
		return 1;


	return 0;
}



// 出力画像のaudaArrayにcudaSurfaceをバインドする
int CudaSURF_BindOutputTexture( cudaArray* cuArray_out )
{
	//========================== 出力画像をcudaSurfaceにバインドする ==============================//
	if( MyCheckCudaErrors( cudaBindSurfaceToArray( surfRef_out, cuArray_out ) ) )//, channelDesc );
		return 1;

	return 0;
}



int CudaSURF_FreeDescBuffer()
{
	//============================ デバイスメモリを解放する ====================//
	if( MyCheckCudaErrors( cudaFree( d_ipoints ) ) )
		return 1;

	return 0;
}


int CudaSURF_FreeFeatureBuffer()
{
//	if( MyCheckCudaErrors( cudaFree( d_vecs ) ) )
//		return 1;

	return 0;
}





// 入力画像をグレースケールに変換する
int CudaSURF_RGB2Luminance( int nWidth, int nHeight, int nDepth )
{
	if( nDepth != 4 )	return 1;

	// スレッド数、ブロック数を設定する
	dim3 blocks( DivUp( nWidth, THREAD_NUM_X ), DivUp( nHeight, THREAD_NUM_Y ), 1 );
	dim3 threads( THREAD_NUM_X, THREAD_NUM_Y, 1 );

	// RGB2Luminanceカーネルを呼び出す
	RGB2Luminance<<< blocks, threads >>>( nWidth, nHeight );

	// 全CUDAスレッドの終了タイミングで同期する
	cudaDeviceSynchronize();
	
	return 0;
}




// IntegralImage
int CudaSURF_IntegralImage( int nWidth, int nHeight, int nDepth )
{
	if( nDepth != 4 )	return 1;// 本当はnDepth==1がただしい。TODO: 実験終わったら4->1にする

	// スレッド数、ブロック数を設定する(y方向に並列化する)
	dim3 blocks( 1, DivUp( nHeight, THREAD_NUM_Y ), 1 );
	dim3 threads( 1, THREAD_NUM_Y, 1 );


	//===================== 横方向に画像をスキャンして輝度値を累積する =====================//
	IntegralImage_h<<< blocks, threads >>>( nWidth );

	// スレッド同期
	cudaDeviceSynchronize();

	//===================== 縦方向に画像をスキャンして輝度値を累積する =====================//
	blocks.x	= DivUp( nWidth, THREAD_NUM_X );
	blocks.y	= 1;
	blocks.z	= 1;

	threads.x	= THREAD_NUM_X;
	threads.y	= 1;
	threads.z	= 1;


	IntegralImage_v<<< blocks, threads >>>( nHeight );

	// スレッド同期 
	cudaDeviceSynchronize();


	return 0;
}


int CudaSURF_HaarWavelet( int nWidth, int nHeight, int nDepth )
{
	if( nDepth != 4 )	return 1;
	
	// スレッド数、ブロック数を設定する
	dim3 blocks( DivUp( nWidth, THREAD_NUM_X ), DivUp( nHeight, THREAD_NUM_Y ), 1 );
	dim3 threads( THREAD_NUM_X, THREAD_NUM_Y, 1 );
	
	// HaarWaveletカーネルを呼び出す
	HaarWavelet<<< blocks, threads >>>( nWidth, nHeight );
	
	// 全CUDAスレッドの終了タイミングで同期する
	cudaDeviceSynchronize();
	
	return 0;
}




// キーポイントのオリエンテーションを計算する
void CudaSURF_ComputeOrientations( pKeypoint *d_ipoints, int numPoints )
{
	dim3 blocks( numPoints, 1 );	// キーポイント毎に1ブロックを割り当てる
	dim3 threads( ORI_SEARCH_BINS, 1, 1 );		// ORI_SEARCH_BINSスレッド並列に実行

	CalcSurfOrientation<<< blocks, threads >>>( d_ipoints, numPoints );

	// 全CUDAスレッドの終了タイミングで同期する
	cudaDeviceSynchronize();
}


//void CudaSURF_ComputeOrientations( pKeypoint *d_ipoints, int numPoints )
//{
//	dim3 blocks( numPoints, 1 );	// キーポイント毎に1ブロックを割り当てる
//	dim3 threads( 42, 1, 1 );		// 42スレッド並列に実行
//
//	CalcSurfOrientation<<< blocks, threads >>>( d_ipoints, numPoints );
//
//	// 全CUDAスレッドの終了タイミングで同期する
//	cudaDeviceSynchronize();
//}



// キーポイントの特徴ベクトルを計算する
void CudaSURF_ComputeFeatureVectors( pKeypoint *d_ipoints, int numPoints, int numLevels, bool bExtended, bool bUpRight, bool bInheritOrientation )
{
	if( (!d_ipoints) || (numPoints<=0) )	return;

	dim3 blocks( numPoints, 1 );	// キーポイント毎に1ブロックを割り当てる
	dim3 threads( 5, 5, 16 );		// なんで？？？

	// キーポイント近傍には、特徴量を計算する領域（縦4つx横4つの計16個）がある→threads.zのスレッドIDで、どの領域かを判別する
	// 領域内には5x5個のサンプル点が存在する → 各サンプル点の2次元ID:
	if( bExtended==true )
	{
		if( bInheritOrientation)
			CalcFeatureDescriptorsExt_InheritOrientation<<< blocks, threads >>>( d_ipoints, numPoints, numLevels, numPoints/numLevels, bUpRight );
		else
			CalcFeatureDescriptorsExt<<< blocks, threads >>>( d_ipoints, bUpRight );
	}
	else
	{
		CalcFeatureDescriptors<<< blocks, threads >>>( d_ipoints, numPoints, numLevels, numPoints/numLevels, bUpRight, bInheritOrientation );
	}
	// 全CUDAスレッドの終了タイミングで同期する
	cudaDeviceSynchronize();
}



// キーポイントの特徴ベクトルを正規化する
void CudaSURF_NormalizeDescriptors( pKeypoint *d_ipoints, int numPoints, bool bExtended )
{
	if( (!d_ipoints) || (numPoints<=0) )	return;

	dim3 blocks( numPoints, 1 );
	dim3 threads( bExtended ? 128 : 64, 1, 1 );


	NormalizeFeatureDescriptors<<< blocks, threads >>>( d_ipoints );
	

	// 全CUDAスレッドの終了タイミングで同期する
	cudaDeviceSynchronize();
}




// キーポイントの特徴量を記述する
void CudaSURF_ComputeDescriptors( pKeypoint *d_ipoints, int numPoints, int numLevels, bool bExtended, bool bUpRight, bool bNormalize, bool bInheritOrientation )
{
	// キーポイントのオリエンテーションを算出
	CudaSURF_ComputeOrientations( d_ipoints, numPoints );

	// SURF特徴ベクトルを計算
	CudaSURF_ComputeFeatureVectors( d_ipoints, numPoints, numLevels, bExtended, bUpRight, bInheritOrientation );

	// 特徴ベクトルを正規化
	if( bNormalize == true)
		CudaSURF_NormalizeDescriptors( d_ipoints, numPoints, bExtended );

}



// cudaMalloc: 容量が不足した場合だけ再確保する方式

// キーポイントのオリエンテーションを計算する
int CudaSURF_GetDescriptors( SurfDescriptors *hSurfDesc, bool bExtended, bool bUpRight, bool bNormalize, bool bInheritOrientation )
{
	//================= キーポイントをデバイスメモリに転送する =================//
	int size_desc		= sizeof(pKeypoint) * hSurfDesc->numKeypoints();
	if( MyCheckCudaErrors( cudaMemcpy( d_ipoints, hSurfDesc->GetKeypoint(), size_desc, cudaMemcpyHostToDevice ) ) )
		return 1;

	//================= 特徴ベクトルデータをデバイスメモリに転送する ============//
	int size_featurevec	= hSurfDesc->GetFeatures()->Size();
	float *h_mat	= (float *)hSurfDesc->GetFeatures()->pData();
	
	if( MyCheckCudaErrors( cudaMemcpyToSymbol( d_featurevecs, h_mat, size_featurevec) ) )
		return 1;

	//================== キーポイントのSURF特徴量を計算する ====================//
	CudaSURF_ComputeDescriptors( d_ipoints, hSurfDesc->numKeypoints(), hSurfDesc->numLevels(), bExtended, bUpRight, bNormalize, bInheritOrientation );
	

	//================= 計算したSURF特徴量をホストに転送する ===================//
	if( MyCheckCudaErrors( cudaMemcpy( hSurfDesc->GetKeypoint(), d_ipoints, size_desc, cudaMemcpyDeviceToHost ) ) )
		return 1;	
	
	if( MyCheckCudaErrors( cudaMemcpyFromSymbol( h_mat, d_featurevecs, size_featurevec) ) )
		return 1;


	//// pKeypointのポインタをデバイスからホストにつなぎかえる
	//pDescs = hSurfDesc->GetKeypoint();
	//for( int i=0; i<numPoints; ++i )
	//{
	//	pDescs[i].refFeature = (float *)(hSurfDesc->GetFeatures().pData() + pDescs[i].vecIdx);
	//
	//}


	return 0;
}






int CudaSURF_SampleTexture( int nWidth, int nHeight, int nDepth )
{
	if( nDepth != 4 )	return 1;
	
	// スレッド数、ブロック数を設定する
	dim3 blocks( DivUp( nWidth, THREAD_NUM_X ), DivUp( nHeight, THREAD_NUM_Y ), 1 );
	dim3 threads( THREAD_NUM_X, THREAD_NUM_Y, 1 );
	
	// InvertColorカーネルを呼び出す
	SampleTexture<<< blocks, threads >>>( nWidth, nHeight );
	
	// 全CUDAスレッドの終了タイミングで同期する
	cudaDeviceSynchronize();
	
	return 0;
}






//##########################################################################################################//
//												Global Functions											//
//##########################################################################################################//



// 画像をRGBから輝度値に変換する
__global__ void RGB2Luminance( unsigned int nWidth, unsigned int nHeight )
{
	// x,y座値を計算する
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y	= blockDim.y * blockIdx.y + threadIdx.y;

	if( x >= nWidth )	return;
	if( y >= nHeight )	return;
	
	float4 src, dest;
	surf2Dread( &src, surfRef_in, x*sizeof(float4), y );
	
	// 輝度値を計算する
	float luminance = 0.298912 * src.x + 0.586611 * src.y + 0.114478 * src.z;
	dest.x = dest.y = dest.z = luminance*255.0;
	dest.w = src.w;

	surf2Dwrite( dest, surfRef_in, x*sizeof(float4), y );// 注意. バイトアドレス単位で要素にアクセスしている。

//printf( "%d, %d\n", g_TexWidth, g_TexHeight );

}




// Haar Wavelet filter kernel
// IntegralImageを使ってHaarWaveletResponseを計算する
__global__ void HaarWavelet( unsigned int nWidth, unsigned int nHeight )
{
	// x,y座値を計算する
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y	= blockDim.y * blockIdx.y + threadIdx.y;

	if( x >= nWidth )	return;
	if( y >= nHeight )	return;
	
	//float4 src;
	float4 dest;
	//src = tex2D( g_TexIn, x, y );
	//surf2Dread( &src, surfIntegralImage, x*sizeof(float4), y );

	const float scale = 1.0f;
	float haarXResult, haarYResult;

//printf( "%d, %d\n", g_TexWidth, g_TexHeight );

	haarXY( x, y, scale, &haarXResult, &haarYResult, 1.0 );


	// ピクセル値を反転させて出力する
	dest.x	= haarXResult;//1.0f - src.x;
	dest.y	= haarYResult;//1.0f - src.y;
	dest.z	= 0.0f;//1.0f - src.z;
	dest.w	= 0.0f;//src.w;


	surf2Dwrite( dest, surfRef_out, x*sizeof(float4), y );// 注意. バイトアドレス単位で要素にアクセスしている。
	
	//printf("%d, %d \n", x, y);
	//printf(" %f, %f, %f\n", dest.x, dest.y, dest.z );
	//printf(" %f, %f\n", haarXResult, haarYResult );
}



// IntergalImage kernel(横方向)
// 現状float4のみ対応。TODO: float1型で動くようにする
__global__ void IntegralImage_h( unsigned int nWidth )
{
	const int Y_CONST = blockDim.y * blockIdx.y + threadIdx.y;

	int x	= 0;

	float4 curr_pixel_value;// = float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 prev_integ_value;// = float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 curr_integ_value;// = float4(0.0f, 0.0f, 0.0f, 0.0f);


	// 最初の1個だけ初期化する
	surf2Dread( &curr_pixel_value, surfRef_in, x, Y_CONST );
	surf2Dwrite( curr_pixel_value, surfIntegralImage, x, Y_CONST );
	prev_integ_value = curr_pixel_value;


	// 横方向にピクセル値を累積する
	for( x=1; x<nWidth; ++x )
	{
		// 現在のピクセルの色を読む
		surf2Dread( &curr_pixel_value, surfRef_in, x*sizeof(float4), Y_CONST );

		// 直前ピクセルの累積値と現在のピクセル値を加算する
		curr_integ_value.x = prev_integ_value.x + curr_pixel_value.x;
		curr_integ_value.y = prev_integ_value.y + curr_pixel_value.y;
		curr_integ_value.z = prev_integ_value.z + curr_pixel_value.z;
		curr_integ_value.w = prev_integ_value.w + curr_pixel_value.w;

		// 新しいピクセル値を登録する
		surf2Dwrite( curr_integ_value, surfIntegralImage, x*sizeof(float4), Y_CONST );

		// prev_integ_valueを更新する
		prev_integ_value = curr_integ_value;
	}// end of x loop

}



// IntergalImage kernel(縦方向)
__global__ void IntegralImage_v( unsigned int nHeight )
{
	const int X_CONST = blockDim.x * blockIdx.x + threadIdx.x;

	int y	= 0;

	float4 curr_pixel_value;// = float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 prev_integ_value;// = float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 curr_integ_value;// = float4(0.0f, 0.0f, 0.0f, 0.0f);


	// 最初の1個だけ初期化する
	surf2Dread( &curr_pixel_value, surfIntegralImage, X_CONST*sizeof(float4), y );
	surf2Dwrite( curr_pixel_value, surfIntegralImage, X_CONST*sizeof(float4), y );
	prev_integ_value = curr_pixel_value;


	// 縦方向にピクセル値を累積する
	for( y=1; y<nHeight; ++y )
	{
		// 現在のピクセルの色を読む
		surf2Dread( &curr_pixel_value, surfIntegralImage, X_CONST*sizeof(float4), y );

		// 直前ピクセルの累積値と現在のピクセル値を加算する
		curr_integ_value.x = prev_integ_value.x + curr_pixel_value.x;
		curr_integ_value.y = prev_integ_value.y + curr_pixel_value.y;
		curr_integ_value.z = prev_integ_value.z + curr_pixel_value.z;
		curr_integ_value.w = prev_integ_value.w + curr_pixel_value.w;

		// 新しいピクセル値を登録する
		surf2Dwrite( curr_integ_value, surfIntegralImage, X_CONST*sizeof(float4), y );

		// prev_integ_valueを更新する
		prev_integ_value = curr_integ_value;
	}// end of y loop
}



// キーポイントの回転を検出する
// Grid内のBlock数をキーポイント数に設定
// Block内のThread数をORI_SEARCH_BINS個に設定
__global__ void CalcSurfOrientation( pKeypoint* g_ipoints, int numPoints )
{
	pKeypoint *g_ipt = g_ipoints + blockIdx.x; // Get a pointer to the interest point processed by this block

	//======================= Step1.キーポイント周辺のサンプル点情報を計算する ======================//
	//int iscale	= fRound(g_ipt->scale);
	float scale	= g_ipt->scale;

	int x	= fRound(g_ipt->x);
	int y	= fRound(g_ipt->y);
	
	// キーポイント半径6s内の情報(スレッド間で共有する)
	__shared__ float	s_resX[109],	// 各サンプル点のx方向HaarWavelet応答(x方向)
						s_resY[109],	// 各サンプル点のy方向HaarWavelet応答(y方向)
						s_ang[109];		// 各サンプル点の向き(HaarWavelet応答から計算)
	
	// 6*scale以内のサンプル点のHaar Wavelet応答を計算する. ORI_SEARCH_BINS個にスレッドを分割しているので、キーポイントはORI_SEARCH_BINS個ジャンプで
	for( int index=threadIdx.x; index<109; index+=ORI_SEARCH_BINS )
	{
		// Get X&Y offset of our sampling point (unscaled)
		int xOffset = dc_coord_x[index];
		int yOffset = dc_coord_y[index];
		float gauss = dc_gauss_lin[index];
		
		//printf( "%d, %d\n", x+xOffset*s, y+yOffset*s );
		// Take the sample
		float haarXResult, haarYResult;
		//haarXY( x+xOffset*iscale, y+yOffset*iscale, 2*iscale, &haarXResult, &haarYResult, gauss );
		haarXY_float( g_ipt->x+xOffset*scale, g_ipt->y+yOffset*scale, 2.0*scale, &haarXResult, &haarYResult, gauss );

	//printf("%f, %f\n", haarXResult, haarYResult );	

		// Store the sample and precomputed angle in shared memory
		s_resX[index] = haarXResult;
		s_resY[index] = haarYResult;
		s_ang[index] = getAngle(haarXResult, haarYResult);
	}
	
	// ORI_SEARCH_BINS個のスレッド全てがHaarWavelet計算完了するのを待つ
	__syncthreads();
	
	
	// calculate the dominant direction
	float sumX, sumY;
	float ang1, ang2, ang;
	//float pi = M_PI;
	//float pi_third = pi / 3.0f; // Size of the sliding window

	// Calculate ang1 at which this thread operates, ORI_SEARCH_BINS times at most
	ang1 = threadIdx.x * ORI_INC;//0.15f;// このスレッドの角度レンジを割り当てる

	// Padded to 48 to allow efficient reduction by 24 threads without branching
	__shared__ float s_metrics[ORI_SEARCH_BINS];
	__shared__ float s_orientations[ORI_SEARCH_BINS];

	
	// Set the padding to 0, so it doesnt interfere.
	//if( threadIdx.x < 6 )
	//
	//	s_metrics[42 + threadIdx.x] = 0.0f;
	//}
	//s_metrics[48] = ********** ********** ********** ********* **000000<-padding
	


	// ORI_SEARCH_BINS個のスレッドが角度ウィンドウそれぞれについて計算実行
	ang2 = ang1+ORI_WINDOW_RNGE > _2_M_PI ? ang1-5.0f*ORI_WINDOW_RNGE : ang1+ORI_WINDOW_RNGE;// 角度レンジの終端が360度越えないよう補正する
	sumX = sumY = 0.0f;

	// 角度ウィンドウ内の全ての点を検出する
	// The x,y results computed above are now interpreted as points
	for( unsigned int k = 0; k < 109; k++ )
	{
		ang = s_ang[k]; // HaarWavelet応答から計算したキーポイント角度を取り出す

		// キーポイントの傾きが、レンジの中に納まっているかどうか調べる
		if( ang1 < ang2 && ang1 < ang && ang < ang2 )
		{
			sumX += s_resX[k];
			sumY += s_resY[k];
		}// end of if
		else if( ang2 < ang1 && ( (ang > 0.0f && ang < ang2) || (ang > ang1 && ang < _2_M_PI) ) )
		{
			sumX += s_resX[k];
			sumY += s_resY[k];
		}// end of else
	}// end of k loop

	// if the vector produced from this window is longer than all
	// previous vectors then this forms the new dominant direction
	s_metrics[ threadIdx.x ]		= sumX*sumX + sumY*sumY;	// 当該角度レンジのメトリックを、サンプル点109個から計算する
	s_orientations[ threadIdx.x ]	= getAngle(sumX, sumY);		// 当該角度レンジの角度を、サンプル点109個から計算する
	
	__syncthreads();// 並列するORI_SEARCH_BINSスレッドが、s_metrics, s_orientationsに値を格納し終わるのを待つ

	//##############################################################//
	// The rest of this function finds the longest vector.
	// The vector length is stored in metrics, while the
	// corresponding orientation is stored in orientations
	// with the same index.
	//##############################################################//
	// ParellelReductionで最大値を探す. 42スレッド→2で割っていくと21で割り切れなくなる→48にすれば3までいける
#pragma unroll 4
	for( int threadCount=ORI_SEARCH_BINS/2; threadCount>=3; threadCount/=2 )// 最大値比較. スレッド0～スレッド3のメモリ領域に、最大値候補を固めて再配置する
	{
		if( threadIdx.x < threadCount )
		{
			if( s_metrics[threadIdx.x] < s_metrics[threadIdx.x + threadCount] )
			{
				s_metrics[threadIdx.x]		= s_metrics[threadIdx.x + threadCount];
				s_orientations[threadIdx.x] = s_orientations[threadIdx.x + threadCount];
			}
		}
		__syncthreads();// 各スレッドで大小比較を行った直後に同期をとる
	}

	if( threadIdx.x==0 )// 最大値候補(3つ)から最大値を取得する
	{
		float max = 0.0f, maxOrientation = 0.0f;
#pragma unroll 3
		for( int i=0; i<3; ++i )
		{
			if( s_metrics[i] > max )
			{
				max = s_metrics[i];
				maxOrientation = s_orientations[i];
			}
		}

		// assign orientation of the dominant response vector
		g_ipt->orientation = maxOrientation;
	}
	
}



// キーポイント毎にブロックを一ずつつ割り当てる
// キーポイント近傍のサンプル点一つずつにスレッドを割り当てる
// SubSquare: 4x4の領域
__global__ void CalcFeatureDescriptors( pKeypoint* g_ipoints, int numPoints, int numLevels, int points_per_level, bool bUpRight, bool bInheritOrientation )
{
	const int iPointIndex		= blockIdx.x;	// キーポイントの通し番号
	const int samplePointXIndex	= threadIdx.x;	// キーポイント近傍のサンプル点xインデックス. The x-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int samplePointYIndex	= threadIdx.y;	// キーポイント近傍のサンプル点yインデックス. The y-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int subSquareId		= threadIdx.z;			// 部分領域の通し番号// The index of the sub-square
	const int subSquareX		= (subSquareId % 4);	// 特徴計算する部分領域のxインデックス　// X-Index of the sub-square
	const int subSquareY		= (subSquareId / 4);	// 特徴量計算する部分領域のyインデックス// Y-Index of the sub-square

	pKeypoint *g_ipt = g_ipoints + iPointIndex; // Pointer to the interest point processed by the current block
	int x = fRound( g_ipt->x );// キーポイントの位置
	int y = fRound( g_ipt->y );// キーポイントの位置
float scale = g_ipt->scale;

	float * const g_desc = d_featurevecs + g_ipt->vecIdx;//g_ipt->descriptor; // Pointer to the interest point descriptor
	float cosine, sine;	// オリエンテーションの正弦と余弦

	// オリエンテーションの正弦、余弦を取得する
	if( !bUpRight )
	{
		cosine	= cosf( g_ipt->orientation );
		sine	= sinf( g_ipt->orientation );
	}

	// スケールを四捨五入する
	int roundedScale = fRound( scale );

	
	// キーポイントの位置を基準としたサンプル点相対位置を計算する
	// Calculate the relative (to x,y) coordinate of sampling point
	int sampleXOffset = subSquareX * 5 + samplePointXIndex - 10;	// 部分領域のxインデックス * 部分領域のx方向サンプル点数 + 部分領域内のxインデックス - 2部分領域分のxサンプル点数
	int sampleYOffset = subSquareY * 5 + samplePointYIndex - 10;	// 部分領域のyインデックス * 部分領域のy方向サンプル点数 + 部分領域内のyインデックス - 2部分領域分のyサンプル点数
	

	// サンプル点の相対座標を用いて、ガウス関数(σ=3.3)ルックアップテーブルから値を取得する
	float gauss = dc_gauss33[ abs(sampleYOffset) ][ abs(sampleXOffset) ];
	
	// サンプル点の絶対座標を取得する
	//int sampleX, sampleY;
	float sampleXf, sampleYf;

	if( !bUpRight )
	{
		//sampleX = fRound( x + (-sampleXOffset*scale*sine + sampleYOffset*scale*cosine ));	// サンプル点x座標 + 相対座標をスケーリングして回転したx座標
		//sampleY = fRound( y + ( sampleXOffset*scale*cosine + sampleYOffset*scale*sine ));	// サンプル点y座標 + 相対座標をスケーリングして回転したy座標	
		sampleXf	= x + (-sampleXOffset*scale*sine + sampleYOffset*scale*cosine );
		sampleYf	= y + ( sampleXOffset*scale*cosine + sampleYOffset*scale*sine );
	}
	else
	{
		//sampleX = fRound( x + sampleXOffset*scale );		// サンプル点x座標 + 相対座標をスケーリングして回転したx座標
		//sampleY = fRound( y + sampleYOffset*scale );		// サンプル点y座標 + 相対座標をスケーリングして回転したy座標
		sampleXf = x + sampleXOffset*scale;		// サンプル点x座標 + 相対座標をスケーリングして回転したx座標
		sampleYf = y + sampleYOffset*scale;		// サンプル点y座標 + 相対座標をスケーリングして回転したy座標
	}

	// サンプル点( SampleX, SampleY)におけるHaarWavelet応答(縦方向横方向それぞれ)を計算する
	float	xResponse, yResponse;	// x/y方向のHaarWavelet応答値
	//haarXY( sampleX, sampleY, roundedScale, &xResponse, &yResponse, gauss );
	haarXY_float( sampleXf, sampleYf, scale, &xResponse, &yResponse, gauss );
	//printf("%f, %f\n", xResponse, yResponse );
	
	// 各スレッドで並列に応答値を計算する。計算結果が全部揃うまで同期する。//Calculate ALL x+y responses for the interest point in parallel
	__shared__ float s_rx[16][5][5];
	__shared__ float s_ry[16][5][5];

	
	// 親キーポイント基準の相対角を計算する
	float rel_cosine = cosine;
	float rel_sine	= sine;

	if(bInheritOrientation)
	{
		int parent_idx	= iPointIndex + points_per_level;

		float parent_orientation	= parent_idx < numPoints ? (g_ipoints + parent_idx)->orientation : 0.0f;
		float relative_orientation	= g_ipt->orientation - parent_orientation;

		rel_cosine	= cosf( relative_orientation );
		rel_sine	= sinf( relative_orientation );
	}


	if( !bUpRight )
	{
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex]	= -xResponse * rel_sine + yResponse * rel_cosine;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex]	= xResponse * rel_cosine + yResponse * rel_sine;

		//s_rx[subSquareId][samplePointXIndex][samplePointYIndex]	= -xResponse * sine + yResponse * cosine;
		//s_ry[subSquareId][samplePointXIndex][samplePointYIndex]	= xResponse * cosine + yResponse * sine;


// TODO: 親キーポイント基準の相対角度にしたい時は、ぶんだけdx, dyを回転させる → オリエンテーションの相対角度
	}
	else
	{
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex] = xResponse;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex] = yResponse;
	}

	// TODO: Can this be optimized? It waits for the results of ALL 400 threads, although they are
	// independent in blocks of 25! (Further work)
	__syncthreads(); // 全スレッドが応答値計算終わるまで待つ



	// 全スレッドで加算やっても無駄→ツリー構造みたいに、隣同士加算を何回か実行する
	__shared__ float s_sums[16][4][5]; // For each sub-square, for the four values (dx,dy,|dx|,|dy|), this contains the sum over five values.
	__shared__ float s_outDesc[16][4]; // The output descriptor partitioned into 16 bins (one for each subsquare)


	// Only five threads per sub-square sum up five values each
	if( threadIdx.y == 0 )
	{
		// Temporary sums
		float tdx = 0.0f, tdy = 0.0f, tmdx = 0.0f, tmdy = 0.0f;

		for( int sy = 0; sy < 5; ++sy )// 横方向5サンプル点分のdx,dyを累積する
		{
			tdx += s_rx[subSquareId][threadIdx.x][sy];
			tdy += s_ry[subSquareId][threadIdx.x][sy];
			tmdx += fabsf(s_rx[subSquareId][threadIdx.x][sy]);
			tmdy += fabsf(s_ry[subSquareId][threadIdx.x][sy]);
		}

		// Write out the four sums to the shared memory
		s_sums[subSquareId][0][threadIdx.x] = tdx;
		s_sums[subSquareId][1][threadIdx.x] = tdy;
		s_sums[subSquareId][2][threadIdx.x] = tmdx;
		s_sums[subSquareId][3][threadIdx.x] = tmdy;
	}

	__syncthreads(); // Wait until all threads have summed their values



	// Only four threads per sub-square can now write out the descriptor
	if (threadIdx.x < 4 && threadIdx.y == 0)
	{
		const float* s_src = s_sums[subSquareId][threadIdx.x]; // Pointer to the sum this thread will write out
		float out = s_src[0] + s_src[1] + s_src[2] + s_src[3] + s_src[4]; // Build the last sum for the value this thread writes out
		int subSquareOffset = (subSquareX + subSquareY * 4) * 4; // Calculate the offset in the descriptor for this sub-square
		g_desc[subSquareOffset + threadIdx.x] = out; // Write the value to the descriptor
		s_outDesc[subSquareId][threadIdx.x] = out; // Write the result to shared memory too, this will be used by the last thread to precompute parts of the length
	}

	__syncthreads();


	// One thread per sub-square now computes the length of the description vector for a sub-square and writes it to global memory
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		g_ipt->lengths[subSquareX][subSquareY] = s_outDesc[subSquareId][0] * s_outDesc[subSquareId][0]
			+ s_outDesc[subSquareId][1] * s_outDesc[subSquareId][1]
			+ s_outDesc[subSquareId][2] * s_outDesc[subSquareId][2]
			+ s_outDesc[subSquareId][3] * s_outDesc[subSquareId][3];
	}
	

	//##############################################################################################################//
}




// キーポイントのSURF特徴量を正規化する
// キーポイント毎にブロック1個を割り当てる
// 64or128次元ベクトルの要素それぞれにスレッド1個を割り当てる
__global__ void NormalizeFeatureDescriptors( pKeypoint *g_ipoints )
{
	pKeypoint *g_ipt = g_ipoints + blockIdx.x;
	
	__shared__ float s_sums[4];
	
	if( threadIdx.x<4 )
	{
		float* g_lengths		= g_ipt->lengths[ threadIdx.x ];
		s_sums[ threadIdx.x ]	= g_lengths[0] + g_lengths[1] + g_lengths[2] + g_lengths[3];
	}// end of if
	
	__syncthreads();
	
	float sum = s_sums[0] + s_sums[1] + s_sums[2] + s_sums[3];
	float len = sum<=1.0e-6f ? 0.0f : rsqrtf( sum );
	d_featurevecs[ g_ipt->vecIdx + threadIdx.x ] *= len;
	//g_ipt->descriptor[ threadIdx.x ] *= len;
	
	g_ipt->orientation = ( len==0.0f ? -1.0f : g_ipt->orientation );
	

}




__global__ void SampleTexture( unsigned int nWidth, unsigned int nHeight )
{
	// x,y座値を計算する
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y	= blockDim.y * blockIdx.y + threadIdx.y;

	if( x >= nWidth )	return;
	if( y >= nHeight )	return;
	
	float4 dest;
	dest = tex2D( g_TexIn, x, y );
	
//printf( "%d, %d\n", g_TexWidth, g_TexHeight );

	surf2Dwrite( dest, surfRef_out, x*sizeof(float4), y );// 注意. バイトアドレス単位で要素にアクセスしている。
	
	//printf("%d, %d \n", x, y);
	//printf(" %f, %f, %f\n", dest.x, dest.y, dest.z );
	//printf(" %f, %f\n", haarXResult, haarYResult );
}




__global__ void CalcFeatureDescriptorsExt( pKeypoint* g_ipoints, int upright )
{
	const int iPointIndex		= blockIdx.x;	// キーポイントの通し番号
	const int samplePointXIndex	= threadIdx.x;	// キーポイント近傍のサンプル点xインデックス. The x-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int samplePointYIndex	= threadIdx.y;	// キーポイント近傍のサンプル点yインデックス. The y-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int subSquareId		= threadIdx.z;			// 部分領域の通し番号// The index of the sub-square
	const int subSquareX		= (subSquareId % 4);	// 特徴計算する部分領域のxインデックス　// X-Index of the sub-square
	const int subSquareY		= (subSquareId / 4);	// 特徴量計算する部分領域のyインデックス// Y-Index of the sub-square

	pKeypoint *g_ipt = g_ipoints + iPointIndex; // Pointer to the interest point processed by the current block
	int x = fRound( g_ipt->x );// キーポイントの位置
	int y = fRound( g_ipt->y );// キーポイントの位置
	float scale = g_ipt->scale;

	float * const g_desc = d_featurevecs + g_ipt->vecIdx;//g_ipt->descriptor; // Pointer to the interest point descriptor
	float cosine, sine;	// オリエンテーションの正弦と余弦

	// オリエンテーションの正弦、余弦を取得する
	if( !upright )
	{
		cosine	= cosf( g_ipt->orientation );
		sine	= sinf( g_ipt->orientation );
	}

	// スケールを四捨五入する
	int roundedScale = fRound( scale );

	
	// キーポイントの位置を基準としたサンプル点相対位置を計算する
	// Calculate the relative (to x,y) coordinate of sampling point
	int sampleXOffset = subSquareX * 5 + samplePointXIndex - 10;	// 部分領域のxインデックス * 部分領域のx方向サンプル点数 + 部分領域内のxインデックス - 2部分領域分のxサンプル点数
	int sampleYOffset = subSquareY * 5 + samplePointYIndex - 10;	// 部分領域のyインデックス * 部分領域のy方向サンプル点数 + 部分領域内のyインデックス - 2部分領域分のyサンプル点数
	

	// サンプル点の相対座標を用いて、ガウス関数(σ=3.3)ルックアップテーブルから値を取得する
	float gauss = dc_gauss33[ abs(sampleYOffset) ][ abs(sampleXOffset) ];
	
	// サンプル点の絶対座標を取得する
	//int sampleX, sampleY;
	float sampleXf, sampleYf;

	if( !upright )
	{
		//sampleX = fRound( x + (-sampleXOffset*scale*sine + sampleYOffset*scale*cosine ) );
		//sampleY = fRound( y + ( sampleXOffset*scale*cosine + sampleYOffset*scale*sine ) );
		sampleXf	= x + (-sampleXOffset*scale*sine + sampleYOffset*scale*cosine );// サンプル点x座標 + 相対座標をスケーリングして回転したx座標
		sampleYf	= y + ( sampleXOffset*scale*cosine + sampleYOffset*scale*sine );// サンプル点y座標 + 相対座標をスケーリングして回転したy座標	
	}
	else
	{
		//sampleX = fRound( x + sampleXOffset*scale );// サンプル点x座標 + 相対座標をスケーリングして回転したx座標
		//sampleY = fRound( y + sampleYOffset*scale );// サンプル点y座標 + 相対座標をスケーリングして回転したy座標
		sampleXf = x + sampleXOffset*scale;
		sampleYf = y + sampleYOffset*scale;
	}

	// サンプル点( SampleX, SampleY)におけるHaarWavelet応答(縦方向横方向それぞれ)を計算する
	float	xResponse, yResponse;	// x/y方向のHaarWavelet応答値
	//haarXY( sampleX, sampleY, roundedScale, &xResponse, &yResponse, gauss );
	haarXY_float( sampleXf, sampleYf, scale, &xResponse, &yResponse, gauss );
	
	// 各スレッドで並列に応答値を計算する。計算結果が全部揃うまで同期する。//Calculate ALL x+y responses for the interest point in parallel
	__shared__ float s_rx[16][5][5];
	__shared__ float s_ry[16][5][5];
	
	
	if( !upright )
	{
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex]	= -xResponse * sine + yResponse * cosine;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex]	= xResponse * cosine + yResponse * sine;
// TODO: 親キーポイント基準の相対角度にしたい時は、ぶんだけdx, dyを回転させる → オリエンテーションの相対角度
	}
	else
	{
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex] = xResponse;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex] = yResponse;
	}

	// TODO: Can this be optimized? It waits for the results of ALL 400 threads, although they are
	// independent in blocks of 25! (Further work)
	__syncthreads(); // 全スレッドが応答値計算終わるまで待つ


	// 全スレッドで加算やっても無駄→ツリー構造みたいに、隣同士加算を何回か実行する
	__shared__ float s_sums[16][8][5];//__shared__ float s_sums[16][4][5]; // For each sub-square, for the four values (dx,dy,|dx|,|dy|), this contains the sum over five values.
	__shared__ float s_outDesc[16][8];	//__shared__ float s_outDesc[16][4]; // The output descriptor partitioned into 16 bins (one for each subsquare)


	// Only five threads per sub-square sum up five values each
	if( threadIdx.y == 0 )
	{
		// Temporary sums
		float	tdx_p = 0.0f, tdx_n = 0.0f,
				tdy_p = 0.0f, tdy_n = 0.0f,
				tmdx_p = 0.0f, tmdx_n	= 0.0f,
				tmdy_p = 0.0f, tmdy_n = 0.0f;

		for( int sy = 0; sy < 5; ++sy )// 横方向5サンプル点分のdx,dyを累積する
		{
			float srx = s_rx[subSquareId][threadIdx.x][sy];
			float sry = s_ry[subSquareId][threadIdx.x][sy];

			//float s_rx_p = srx * float(sry>=0.0f);
			//float s_rx_n = srx * float(sry<0.0f);
			//float s_ry_p = sry * float(srx>=0.0f);
			//float s_ry_n = sry * float(srx<0.0f);

			//tdx_p += s_rx_p;// 正の場合だけ加算
			//tdx_n += s_rx_n;// 負の場合だけ加算

			//tdy_p += s_ry_p;// 正の場合だけ加算
			//tdy_n += s_ry_n;// 負の場合だけ加算

			//tmdx_p += fabsf(s_rx_p);
			//tmdx_n += fabsf(s_rx_n);

			//tmdy_p += fabsf(s_ry_p);
			//tmdy_n += fabsf(s_ry_n);



			if( sry >= 0 )
			{
				tdx_p	+= srx;
				tmdx_p	+= fabsf( srx );
			}
			else
			{
				tdx_n	+= srx;
				tmdx_n	+= fabsf( srx );
			}

			if( srx >= 0 )
			{
				tdy_p	+= sry;
				tmdy_p	+= fabsf( sry );
			}
			else
			{
				tdy_n	+= sry;
				tmdy_n	+= fabsf( sry );
			}




		}

		// Write out the four sums to the shared memory
		s_sums[subSquareId][0][threadIdx.x] = tdx_p;
		s_sums[subSquareId][1][threadIdx.x] = tdy_p;
		s_sums[subSquareId][2][threadIdx.x] = tmdx_p;
		s_sums[subSquareId][3][threadIdx.x] = tmdy_p;

		s_sums[subSquareId][4][threadIdx.x] = tdx_n;
		s_sums[subSquareId][5][threadIdx.x] = tdy_n;
		s_sums[subSquareId][6][threadIdx.x] = tmdx_n;
		s_sums[subSquareId][7][threadIdx.x] = tmdy_n;

	}

	__syncthreads(); // Wait until all threads have summed their values



	// Only four threads per sub-square can now write out the descriptor
	if (threadIdx.x < 4 && threadIdx.y == 0)
	{
// [5]～[7]の処理が効いてない. threadIdx.xは5までしかないため
		float* s_src;
		float out;
		int subSquareOffset = (subSquareX + subSquareY * 4) * 8; // Calculate the offset in the descriptor for this sub-square

		// [0]～[3]
		s_src	= s_sums[subSquareId][threadIdx.x]; // Pointer to the sum this thread will write out
		out		= s_src[0] + s_src[1] + s_src[2] + s_src[3] + s_src[4]; // Build the last sum for the value this thread writes out
		g_desc[subSquareOffset + threadIdx.x]	= out; // Write the value to the descriptor
		s_outDesc[subSquareId][threadIdx.x]		= out; // Write the result to shared memory too, this will be used by the last thread to precompute parts of the length
	
		// [4]～[7]
		s_src	= s_sums[subSquareId][threadIdx.x+4]; // Pointer to the sum this thread will write out
		out		= s_src[0] + s_src[1] + s_src[2] + s_src[3] + s_src[4]; // Build the last sum for the value this thread writes out
		g_desc[subSquareOffset + threadIdx.x+4]	= out; // Write the value to the descriptor
		s_outDesc[subSquareId][threadIdx.x+4]	= out; // Write the result to shared memory too, this will be used by the last thread to precompute parts of the length
	}

	__syncthreads();


	// One thread per sub-square now computes the length of the description vector for a sub-square and writes it to global memory
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		g_ipt->lengths[subSquareX][subSquareY] =	s_outDesc[subSquareId][0] * s_outDesc[subSquareId][0] +
													s_outDesc[subSquareId][1] * s_outDesc[subSquareId][1] +
													s_outDesc[subSquareId][2] * s_outDesc[subSquareId][2] +
													s_outDesc[subSquareId][3] * s_outDesc[subSquareId][3] +

													s_outDesc[subSquareId][4] * s_outDesc[subSquareId][4] +
													s_outDesc[subSquareId][5] * s_outDesc[subSquareId][5] +
													s_outDesc[subSquareId][6] * s_outDesc[subSquareId][6] +
													s_outDesc[subSquareId][7] * s_outDesc[subSquareId][7];
	}
	
}







// SURF128, 親キーポイントの角度継承版
__global__ void CalcFeatureDescriptorsExt_InheritOrientation( pKeypoint* g_ipoints, int numPoints, int numLevels, int points_per_level, bool bUpRight )
{
	const int iPointIndex		= blockIdx.x;	// キーポイントの通し番号
	const int samplePointXIndex	= threadIdx.x;	// キーポイント近傍のサンプル点xインデックス. The x-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int samplePointYIndex	= threadIdx.y;	// キーポイント近傍のサンプル点yインデックス. The y-position of the sampling point within a sub-square, relative to the upper-left corner of the sub square
	const int subSquareId		= threadIdx.z;			// 部分領域の通し番号// The index of the sub-square
	const int subSquareX		= (subSquareId % 4);	// 特徴計算する部分領域のxインデックス　// X-Index of the sub-square
	const int subSquareY		= (subSquareId / 4);	// 特徴量計算する部分領域のyインデックス// Y-Index of the sub-square

	pKeypoint *g_ipt = g_ipoints + iPointIndex; // Pointer to the interest point processed by the current block
	int x = fRound( g_ipt->x );// キーポイントの位置
	int y = fRound( g_ipt->y );// キーポイントの位置
	float scale = g_ipt->scale;

	float * const g_desc = d_featurevecs + g_ipt->vecIdx;//g_ipt->descriptor; // Pointer to the interest point descriptor
	float cosine, sine;	// オリエンテーションの正弦と余弦

	// オリエンテーションの正弦、余弦を取得する
	if( !bUpRight )
	{
		cosine	= cosf( g_ipt->orientation );
		sine	= sinf( g_ipt->orientation );
	}

	// スケールを四捨五入する
	int roundedScale = fRound( scale );

	
	// キーポイントの位置を基準としたサンプル点相対位置を計算する
	// Calculate the relative (to x,y) coordinate of sampling point
	int sampleXOffset = subSquareX * 5 + samplePointXIndex - 10;	// 部分領域のxインデックス * 部分領域のx方向サンプル点数 + 部分領域内のxインデックス - 2部分領域分のxサンプル点数
	int sampleYOffset = subSquareY * 5 + samplePointYIndex - 10;	// 部分領域のyインデックス * 部分領域のy方向サンプル点数 + 部分領域内のyインデックス - 2部分領域分のyサンプル点数
	

	// サンプル点の相対座標を用いて、ガウス関数(σ=3.3)ルックアップテーブルから値を取得する
	float gauss = dc_gauss33[ abs(sampleYOffset) ][ abs(sampleXOffset) ];
	
	// サンプル点の絶対座標を取得する
	//int sampleX, sampleY;
	float sampleXf, sampleYf;

	if( !bUpRight )
	{
		//sampleX = fRound( x + (-sampleXOffset*scale*sine + sampleYOffset*scale*cosine ) );
		//sampleY = fRound( y + ( sampleXOffset*scale*cosine + sampleYOffset*scale*sine ) );
		sampleXf	= x + (-sampleXOffset*scale*sine + sampleYOffset*scale*cosine );// サンプル点x座標 + 相対座標をスケーリングして回転したx座標
		sampleYf	= y + ( sampleXOffset*scale*cosine + sampleYOffset*scale*sine );// サンプル点y座標 + 相対座標をスケーリングして回転したy座標	
	}
	else
	{
		//sampleX = fRound( x + sampleXOffset*scale );// サンプル点x座標 + 相対座標をスケーリングして回転したx座標
		//sampleY = fRound( y + sampleYOffset*scale );// サンプル点y座標 + 相対座標をスケーリングして回転したy座標
		sampleXf = x + sampleXOffset*scale;
		sampleYf = y + sampleYOffset*scale;
	}

	// サンプル点( SampleX, SampleY)におけるHaarWavelet応答(縦方向横方向それぞれ)を計算する
	float	xResponse, yResponse;	// x/y方向のHaarWavelet応答値
	//haarXY( sampleX, sampleY, roundedScale, &xResponse, &yResponse, gauss );
	haarXY_float( sampleXf, sampleYf, scale, &xResponse, &yResponse, gauss );
	
	// 各スレッドで並列に応答値を計算する。計算結果が全部揃うまで同期する。//Calculate ALL x+y responses for the interest point in parallel
	__shared__ float s_rx[16][5][5];
	__shared__ float s_ry[16][5][5];
	


//########################### 親キーポイント基準の相対角を計算する #################################//

// TODO: 原因調査.角度継承をオンにすると、回転不変性が失われる. なんでだ？？？？

	//int numpoints	= points_per_level * numLevels;
	int parent_idx	= iPointIndex + points_per_level;


// numPoints: キーポイントの総数
// parent_idx: 親キーポイントのインデックス
// parent_idxがnumPoints以上になった->自分が大元の親->相対角なし

//	float parent_orientation	= parent_idx < numPoints ? (g_ipoints + parent_idx)->orientation : 0.0f;
//	float relative_orientation	= g_ipt->orientation - parent_orientation;// 親角度からの差分が分かっただけ. 



//	float rel_cosine	= cosf( relative_orientation );
//	float rel_sine		= sinf( relative_orientation );

//###################### テストコード 2013.09.13 ########################//
float final_orientation	= parent_idx < numPoints ? (g_ipoints + parent_idx)->orientation : g_ipt->orientation;
float rel_cosine	= cosf( final_orientation );
float rel_sine	= sinf( final_orientation );


//##############################################################################################//

	if( !bUpRight )
	{
		// TODO: 親キーポイント基準の相対角度にしたい時は、ぶんだけdx, dyを回転させる → オリエンテーションの相対角度
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex]	= -xResponse * rel_sine + yResponse * rel_cosine;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex]	= xResponse * rel_cosine + yResponse * rel_sine;
	}
	else
	{
		s_rx[subSquareId][samplePointXIndex][samplePointYIndex] = xResponse;
		s_ry[subSquareId][samplePointXIndex][samplePointYIndex] = yResponse;
	}

	// TODO: Can this be optimized? It waits for the results of ALL 400 threads, although they are
	// independent in blocks of 25! (Further work)
	__syncthreads(); // 全スレッドが応答値計算終わるまで待つ


	// 全スレッドで加算やっても無駄→ツリー構造みたいに、隣同士加算を何回か実行する
	__shared__ float s_sums[16][8][5];//__shared__ float s_sums[16][4][5]; // For each sub-square, for the four values (dx,dy,|dx|,|dy|), this contains the sum over five values.
	__shared__ float s_outDesc[16][8];	//__shared__ float s_outDesc[16][4]; // The output descriptor partitioned into 16 bins (one for each subsquare)


	// Only five threads per sub-square sum up five values each
	if( threadIdx.y == 0 )
	{
		// Temporary sums
		float	tdx_p = 0.0f, tdx_n = 0.0f,
				tdy_p = 0.0f, tdy_n = 0.0f,
				tmdx_p = 0.0f, tmdx_n	= 0.0f,
				tmdy_p = 0.0f, tmdy_n = 0.0f;

		for( int sy = 0; sy < 5; ++sy )// 横方向5サンプル点分のdx,dyを累積する
		{
			float srx = s_rx[subSquareId][threadIdx.x][sy];
			float sry = s_ry[subSquareId][threadIdx.x][sy];

			//float s_rx_p = srx * float(sry>=0.0f);
			//float s_rx_n = srx * float(sry<0.0f);
			//float s_ry_p = sry * float(srx>=0.0f);
			//float s_ry_n = sry * float(srx<0.0f);

			//tdx_p += s_rx_p;// 正の場合だけ加算
			//tdx_n += s_rx_n;// 負の場合だけ加算

			//tdy_p += s_ry_p;// 正の場合だけ加算
			//tdy_n += s_ry_n;// 負の場合だけ加算

			//tmdx_p += fabsf(s_rx_p);
			//tmdx_n += fabsf(s_rx_n);

			//tmdy_p += fabsf(s_ry_p);
			//tmdy_n += fabsf(s_ry_n);



			if( sry >= 0 )
			{
				tdx_p	+= srx;
				tmdx_p	+= fabsf( srx );
			}
			else
			{
				tdx_n	+= srx;
				tmdx_n	+= fabsf( srx );
			}

			if( srx >= 0 )
			{
				tdy_p	+= sry;
				tmdy_p	+= fabsf( sry );
			}
			else
			{
				tdy_n	+= sry;
				tmdy_n	+= fabsf( sry );
			}




		}

		// Write out the four sums to the shared memory
		s_sums[subSquareId][0][threadIdx.x] = tdx_p;
		s_sums[subSquareId][1][threadIdx.x] = tdy_p;
		s_sums[subSquareId][2][threadIdx.x] = tmdx_p;
		s_sums[subSquareId][3][threadIdx.x] = tmdy_p;

		s_sums[subSquareId][4][threadIdx.x] = tdx_n;
		s_sums[subSquareId][5][threadIdx.x] = tdy_n;
		s_sums[subSquareId][6][threadIdx.x] = tmdx_n;
		s_sums[subSquareId][7][threadIdx.x] = tmdy_n;

	}

	__syncthreads(); // Wait until all threads have summed their values



	// Only four threads per sub-square can now write out the descriptor
	if (threadIdx.x < 4 && threadIdx.y == 0)
	{
// [5]～[7]の処理が効いてない. threadIdx.xは5までしかないため
		float* s_src;
		float out;
		int subSquareOffset = (subSquareX + subSquareY * 4) * 8; // Calculate the offset in the descriptor for this sub-square

		// [0]～[3]
		s_src	= s_sums[subSquareId][threadIdx.x]; // Pointer to the sum this thread will write out
		out		= s_src[0] + s_src[1] + s_src[2] + s_src[3] + s_src[4]; // Build the last sum for the value this thread writes out
		g_desc[subSquareOffset + threadIdx.x]	= out; // Write the value to the descriptor
		s_outDesc[subSquareId][threadIdx.x]		= out; // Write the result to shared memory too, this will be used by the last thread to precompute parts of the length
	
		// [4]～[7]
		s_src	= s_sums[subSquareId][threadIdx.x+4]; // Pointer to the sum this thread will write out
		out		= s_src[0] + s_src[1] + s_src[2] + s_src[3] + s_src[4]; // Build the last sum for the value this thread writes out
		g_desc[subSquareOffset + threadIdx.x+4]	= out; // Write the value to the descriptor
		s_outDesc[subSquareId][threadIdx.x+4]	= out; // Write the result to shared memory too, this will be used by the last thread to precompute parts of the length
	}

	__syncthreads();


	// One thread per sub-square now computes the length of the description vector for a sub-square and writes it to global memory
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		g_ipt->lengths[subSquareX][subSquareY] =	s_outDesc[subSquareId][0] * s_outDesc[subSquareId][0] +
													s_outDesc[subSquareId][1] * s_outDesc[subSquareId][1] +
													s_outDesc[subSquareId][2] * s_outDesc[subSquareId][2] +
													s_outDesc[subSquareId][3] * s_outDesc[subSquareId][3] +

													s_outDesc[subSquareId][4] * s_outDesc[subSquareId][4] +
													s_outDesc[subSquareId][5] * s_outDesc[subSquareId][5] +
													s_outDesc[subSquareId][6] * s_outDesc[subSquareId][6] +
													s_outDesc[subSquareId][7] * s_outDesc[subSquareId][7];
	}
	
}




























//##########################################################################################################//
//												Device functions											//
//##########################################################################################################//

__device__ float getAngle( float X, float Y )
{
	float pi = M_PI;

	if (X >= 0.0f && Y >= 0.0f)
		return atanf(Y/X);

	if (X < 0.0f && Y >= 0.0f)
		return pi - atanf(-Y/X);

	if (X < 0.0f && Y < 0.0f)
		return pi + atanf(Y/X);

	if (X >= 0.0f && Y < 0.0f)
		return 2.0f*pi - atanf(-Y/X);

	return 0.0f;
}


//! Computes the sum of pixels within the rectangle specified by the top-left start
//! co-ordinate (row, col) and size (rows, cols).
__device__ float BoxIntegral( float *data, int width, int height, size_t widthStep, int row, int col, int rows, int cols )
{
	// The subtraction by one for row/col is because row/col is inclusive.
	int r1 = min( row, height ) - 1;
	int c1 = min( col, width ) - 1;
	int r2 = min( row + rows, height ) - 1;
	int c2 = min( col + cols, width ) - 1;

	float A, B, C, D;
	A = data[ r1 * widthStep + c1 ];
	B = data[ r1 * widthStep + c2 ];
	C = data[ r2 * widthStep + c1 ];
	D = data[ r2 * widthStep + c2 ];

	return max( 0.0f, A - B - C + D );
}






__device__  void haarXY( int x, int y, int roundedScale, float *xResponse, float *yResponse, float gauss )
{
	
	float4	leftTop, middleTop, rightTop,
			leftMiddle, rightMiddle,
			leftBottom, middleBottom, rightBottom;

	/*
	//######################### cudaSurfaceを使ったピクセルデータへのアクセス ###############################//
	int xmiddle	= min( max(x, 0), g_TexWidth-1 );
	int ymiddle	= min( max(y, 0), g_TexHeight-1 );
	int left	= max( xmiddle - roundedScale, 0);				// 左サンプル位置.スケールに応じて移動幅が変わる
	int right	= min( xmiddle + roundedScale, g_TexWidth-1 );	// 右サンプル位置.スケールに応じて移動幅が変わる
	int top		= max( ymiddle - roundedScale, 0 );				// 上サンプル位置.スケールに応じて移動幅が変わる
	int bottom	= min( ymiddle + roundedScale, g_TexHeight-1 );	// 下サンプル位置.スケールに応じて移動幅が変わる

	
	left		*= sizeof(float4);
	right		*= sizeof(float4);
	
// TODO: Out of Range Address発生！なんでだーーーー → xmiddle, ymiddleのクランプ忘れ(2013.06.24)
//printf( "  %d, %d\n", xmiddle, ymiddle );

	surf2Dread( &leftTop, surfIntegralImage, left, top );
	surf2Dread( &leftMiddle, surfIntegralImage, left, ymiddle );
	surf2Dread( &leftBottom, surfIntegralImage, left, bottom );
	surf2Dread( &rightTop, surfIntegralImage, right, top );
	surf2Dread( &rightMiddle, surfIntegralImage, right, ymiddle );
	surf2Dread( &rightBottom, surfIntegralImage, right, bottom );
	surf2Dread( &middleTop, surfIntegralImage, xmiddle*sizeof(float4), top );
	surf2Dread( &middleBottom, surfIntegralImage, xmiddle*sizeof(float4), bottom );
	//######################################################################################################//
	*/
	
	//######################### cudaTextureを使ったピクセルデータへのアクセス #############################//
	int xmiddle = x;
	int ymiddle = y;
	int left = xmiddle - roundedScale;
	int right = xmiddle + roundedScale;
	int top = ymiddle - roundedScale;
	int bottom = ymiddle + roundedScale;

	leftTop = tex2D(g_texIntegralImage,  left, top);
	leftMiddle = tex2D(g_texIntegralImage,  left, ymiddle);
	leftBottom = tex2D(g_texIntegralImage,  left, bottom);
	rightTop = tex2D(g_texIntegralImage,  right, top);
	rightMiddle = tex2D(g_texIntegralImage,  right, ymiddle);
	rightBottom = tex2D(g_texIntegralImage,  right, bottom);
	middleTop = tex2D(g_texIntegralImage,  xmiddle, top);
	middleBottom = tex2D(g_texIntegralImage,  xmiddle, bottom);
	//######################################################################################################//
	

	// y方向の応答を計算する
	float upperHalf	= leftTop.x - rightTop.x - leftMiddle.x + rightMiddle.x;
	float lowerHalf	= leftMiddle.x - rightMiddle.x - leftBottom.x + rightBottom.x;
	*yResponse		= gauss * (lowerHalf - upperHalf);

	// x方向の応答を計算する
	float rightHalf	= middleTop.x - rightTop.x - middleBottom.x + rightBottom.x;
	float leftHalf	= leftTop.x - middleTop.x - leftBottom.x + middleBottom.x;
	*xResponse		= gauss * (rightHalf - leftHalf);
}





__device__  void haarXY_float( float x, float y, float Scale, float *xResponse, float *yResponse, float gauss )
{
	
	float4	leftTop, middleTop, rightTop,
			leftMiddle, rightMiddle,
			leftBottom, middleBottom, rightBottom;
	
	//######################### cudaTextureを使ったピクセルデータへのアクセス #############################//
	float xmiddle = x;
	float ymiddle = y;
	float left = xmiddle - Scale;
	float right = xmiddle + Scale;
	float top = ymiddle - Scale;
	float bottom = ymiddle + Scale;

	leftTop		= tex2D( g_texIntegralImage,  left, top );
	leftMiddle	= tex2D( g_texIntegralImage,  left, ymiddle );
	leftBottom	= tex2D( g_texIntegralImage,  left, bottom );
	rightTop	= tex2D( g_texIntegralImage,  right, top );
	rightMiddle	= tex2D( g_texIntegralImage,  right, ymiddle );
	rightBottom	= tex2D( g_texIntegralImage,  right, bottom );
	middleTop	= tex2D( g_texIntegralImage,  xmiddle, top );
	middleBottom = tex2D( g_texIntegralImage,  xmiddle, bottom );
	//######################################################################################################//
	

	// y方向の応答を計算する
	float upperHalf	= leftTop.x - rightTop.x - leftMiddle.x + rightMiddle.x;
	float lowerHalf	= leftMiddle.x - rightMiddle.x - leftBottom.x + rightBottom.x;
	*yResponse		= gauss * (lowerHalf - upperHalf);

	// x方向の応答を計算する
	float rightHalf	= middleTop.x - rightTop.x - middleBottom.x + rightBottom.x;
	float leftHalf	= leftTop.x - middleTop.x - leftBottom.x + middleBottom.x;
	*xResponse		= gauss * (rightHalf - leftHalf);
}

















// CalcSurfOrientationのオリジナルコード
// キーポイントの回転を検出する
// Grid内のBlock数をキーポイント数に設定
// Block内のThread数を42個に設定
//__global__ void CalcSurfOrientation( pKeypoint* g_ipoints, int numPoints )
//{
//	pKeypoint *g_ipt = g_ipoints + blockIdx.x; // Get a pointer to the interest point processed by this block
//
//	//======================= Step1.キーポイント周辺のサンプル点情報を計算する ======================//
//	int iscale	= fRound(g_ipt->scale);	//float scale	= g_ipt->scale;
//
//	int x	= fRound(g_ipt->x);
//	int y	= fRound(g_ipt->y);
//	
//	// キーポイント半径6s内の情報(スレッド間で共有する)
//	__shared__ float	s_resX[109],	// 各サンプル点のx方向HaarWavelet応答(x方向)
//						s_resY[109],	// 各サンプル点のy方向HaarWavelet応答(y方向)
//						s_ang[109];		// 各サンプル点の向き(HaarWavelet応答から計算)
//	
//	// 6*scale以内のサンプル点のHaar Wavelet応答を計算する. 42個にスレッドを分割しているので、キーポイントは42個ジャンプで
//	for( int index=threadIdx.x; index<109; index+=42 )
//	{
//		// Get X&Y offset of our sampling point (unscaled)
//		int xOffset = dc_coord_x[index];
//		int yOffset = dc_coord_y[index];
//		float gauss = dc_gauss_lin[index];
//		
//		//printf( "%d, %d\n", x+xOffset*s, y+yOffset*s );
//		// Take the sample
//		float haarXResult, haarYResult;
//		haarXY( x+xOffset*iscale, y+yOffset*iscale, 2*iscale, &haarXResult, &haarYResult, gauss );//haarXY_float( g_ipt->x+xOffset*scale, g_ipt->y+yOffset*scale, 2.0*scale, &haarXResult, &haarYResult, gauss );
//
//
//	//printf("%f, %f\n", haarXResult, haarYResult );
//	
//
//		// Store the sample and precomputed angle in shared memory
//		s_resX[index] = haarXResult;
//		s_resY[index] = haarYResult;
//		s_ang[index] = getAngle(haarXResult, haarYResult);
//	}
//	
//	// 42個のスレッド全てがHaarWavelet計算完了するのを待つ
//	__syncthreads();
//	
//	
//	// calculate the dominant direction
//	float sumX, sumY;
//	float ang1, ang2, ang;
//	float pi = M_PI;
//	//float pi_third = pi / 3.0f; // Size of the sliding window
//
//	
//
//	// Calculate ang1 at which this thread operates, 42 times at most
//	ang1 = threadIdx.x * 0.15f;// このスレッドの角度レンジを割り当てる
//
//	// Padded to 48 to allow efficient reduction by 24 threads without branching
//	__shared__ float s_metrics[48];
//	__shared__ float s_orientations[48];
//
//	// Set the padding to 0, so it doesnt interfere.
//	if( threadIdx.x < 6 )
//	{
//		s_metrics[42 + threadIdx.x] = 0.0f;
//	}
//	//s_metrics[48] = ********** ********** ********** ********* **000000<-padding
//
//	// Each thread now computes one of the windows
//	ang2 = ang1+ORI_WINDOW_RNGE > 2.0f*pi ? ang1-5.0f*ORI_WINDOW_RNGE : ang1+ORI_WINDOW_RNGE;// 角度レンジの終端が360度越えないよう補正する
//	sumX = sumY = 0.0f;
//
//	// Find all the points that are inside the window
//	// The x,y results computed above are now interpreted as points
//	for( unsigned int k = 0; k < 109; k++ )
//	{
//		ang = s_ang[k]; // HaarWavelet応答から計算したキーポイント角度を取り出す
//
//		// キーポイントの傾きが、レンジの中に納まっているかどうか調べる
//		if( ang1 < ang2 && ang1 < ang && ang < ang2 )
//		{
//			sumX += s_resX[k];
//			sumY += s_resY[k];
//		}// end of if
//		else if( ang2 < ang1 && ( (ang > 0.0f && ang < ang2) || (ang > ang1 && ang < 2.0f*pi) ) )
//		{
//			sumX += s_resX[k];
//			sumY += s_resY[k];
//		}// end of else
//	}// end of k loop
//
//	// if the vector produced from this window is longer than all
//	// previous vectors then this forms the new dominant direction
//	s_metrics[ threadIdx.x ]		= sumX*sumX + sumY*sumY;	// 当該角度レンジのメトリックを、サンプル点109個から計算する
//	s_orientations[ threadIdx.x ]	= getAngle(sumX, sumY);		// 当該角度レンジの角度を、サンプル点109個から計算する
//	
//	__syncthreads();// 並列する48スレッド(有効な答え入ってるのは42個)が、s_metrics, s_orientationsに値を格納し終わるのを待つ
//
//	//##############################################################//
//	// The rest of this function finds the longest vector.
//	// The vector length is stored in metrics, while the
//	// corresponding orientation is stored in orientations
//	// with the same index.
//	//##############################################################//
//	// ParellelReductionで最大値を探す. 42スレッド→2で割っていくと21で割り切れなくなる→48にすれば3までいける
//#pragma unroll 4
//	for( int threadCount=24; threadCount>=3; threadCount/=2 )// 最大値比較. スレッド0～スレッド3のメモリ領域に、最大値候補を固めて再配置する
//	{
//		if( threadIdx.x < threadCount )
//		{
//			if( s_metrics[threadIdx.x] < s_metrics[threadIdx.x + threadCount] )
//			{
//				s_metrics[threadIdx.x]		= s_metrics[threadIdx.x + threadCount];
//				s_orientations[threadIdx.x] = s_orientations[threadIdx.x + threadCount];
//			}
//		}
//		__syncthreads();// 各スレッドで大小比較を行った直後に同期をとる
//	}
//
//	if( threadIdx.x==0 )// 最大値候補(3つ)から最大値を取得する
//	{
//		float max = 0.0f, maxOrientation = 0.0f;
//#pragma unroll 3
//		for( int i=0; i<3; ++i )
//		{
//			if( s_metrics[i] > max )
//			{
//				max = s_metrics[i];
//				maxOrientation = s_orientations[i];
//			}
//		}
//
//		// assign orientation of the dominant response vector
//		g_ipt->orientation = maxOrientation;
//	}
//	
//}