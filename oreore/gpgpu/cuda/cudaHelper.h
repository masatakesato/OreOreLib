#ifndef CUDA_HELPER_H
#define	CUDA_HELPER_H


//######################################################################################//
//								Cuda library include									//
//######################################################################################//

#include	<cuda.h>
#include	<cuda_runtime.h>
#include	<device_launch_parameters.h>
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")


//######################################################################################//
//									My library include									//
//######################################################################################//

#include	<oreore/common/TString.h>


//######################################################################################//
//										Macro											//
//######################################################################################//

#define CUDA_ERROR_CHECK

#ifndef CudaSafeCall
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#endif

#ifndef CudaCheckError
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
#endif

#ifndef fRound
#define fRound(x) (int)rintf(x)
#endif

#ifndef DivUp
#define DivUp(a, b)( ((a%b)==0)?(a/b):(a/b+1) )
#endif



//######################################################################################//
//									Helper functions									//
//######################################################################################//

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
	{
		tcout<< "cudaSafeCall() failed at " << file << ":" << line << " : " << cudaGetErrorString( err ) << tendl;
		//fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif

	return;
}


inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		tcout<< "cudaCheckError() failed at " << file << ":" << line << " : " << cudaGetErrorString( err ) << tendl;
		//fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		tcout<< "cudaCheckError() with sync failed at " << file << ":" << line << " : " << cudaGetErrorString( err ) << tendl;
		//fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif

	return;
}


static bool resetCuda()
{
	cudaDeviceReset();
	return 0;
}




#endif // !CUDA_HELPER_H