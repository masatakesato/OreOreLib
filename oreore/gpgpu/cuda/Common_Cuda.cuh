#ifndef	COMMOMN_CUDA_H
#define	COMMOMN_CUDA_H


//#include	<stdio.h>
#include	<iostream>

#ifndef fRound
#define fRound(x) (int)rintf(x)
#endif


#ifndef DivUp
#define DivUp(a, b)( ((a%b)==0)?(a/b):(a/b+1) )
#endif




// CUDA ErrorCheck Function
inline int MyCheckCudaErrors( cudaError_t	err )
{
	if( err != cudaSuccess )
	{
		std::cout << "CUDA error : \n" << cudaGetErrorString(err) << std::endl;
		//printf( "CUDA error : %s\n", cudaGetErrorString(err) );
		return 1;
	}

	return 0;
}




#endif	// COMMOMN_CUDA_H //