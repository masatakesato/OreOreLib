#ifndef ND_ARRAY_BASE_H
#define	ND_ARRAY_BASE_H


//#include	<oreore/container/ArrayBase.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/StaticArray.h>
#include	<oreore/container/ArrayView.h>
#include	<oreore/container/NDShape.h>



namespace OreOreLib
{

	namespace detail
	{
		// struct for NDArrayView specialization
		template< typename T >	struct NDARRVIEW{ using Type = typename T; };

		// struct for NDStaticArray specialization
		template< typename T >	struct NDSTATICARR{ using Type = typename T; };
	}


	// NDArrayBase declaration
	template< typename T, int64 ...Ns > class NDArrayBase; 


	// NDArray specialization
	template< typename T, int64 N >
	using NDArray = NDArrayBase< T, N >;


	// NDStaticArray specialization
	template< typename T, int64 ... Args >
	using NDStaticArray = NDArrayBase< detail::NDSTATICARR<T>, Args... >;


	// NDArrayView specialization
	template< typename T, int64 N >
	using NDArrayView = NDArrayBase< detail::NDARRVIEW<T>, N >;
	


}


#endif // !ND_ARRAY_BASE_H
