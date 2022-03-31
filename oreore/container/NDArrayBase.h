#ifndef ND_ARRAY_BASE_H
#define	ND_ARRAY_BASE_H


//#include	<oreore/container/ArrayBase.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/StaticArray.h>
#include	<oreore/container/ArrayView.h>
#include	<oreore/container/NDShape.h>



namespace OreOreLib
{

	//##########################################################################//
	//																			//
	//							NDArrayBase templates							//
	//																			//
	//##########################################################################//

	namespace detail
	{
		// struct for NDArrayView specialization
		template< typename T >	struct NDARRVIEW{ using Type = typename T; };

		// struct for NDStaticArray specialization
		template< typename T >	struct NDSTATICARR{	using Type = typename T; };
	}


	// NDArrayBase declaration
	template< typename T, typename IndexType, IndexType ...Ns > class NDArrayBase; 




	//##########################################################################//
	//																			//
	//						NDArrayBase partial specialization					//
	//																			//
	//##########################################################################//

	// NDStaticArray
	template< typename T, typename IndexType, IndexType N >
	using NDArrayImpl = NDArrayBase< T, IndexType, N >;
		
	// NDArrayView
	template< typename T, typename IndexType, IndexType ...Ns >
	using NDStaticArrayImpl = NDArrayBase< detail::NDSTATICARR<T>, IndexType, Ns... >;

	// NDArrayView
	template< typename T, typename IndexType, IndexType N >
	using NDArrayViewImpl = NDArrayBase< detail::NDARRVIEW<T>, IndexType, N >;




	//##########################################################################//
	//																			//
	//						NDArrayBase full specialization						//
	//																			//
	//##########################################################################//

	// NDArray
	template< typename T, MemSizeType N >
	using NDArray = NDArrayImpl< T, MemSizeType, N >;

	// NDStaticArray
	template< typename T, MemSizeType ...Ns >
	using NDStaticArray = NDStaticArrayImpl< T, MemSizeType, Ns... >;

	// NDArrayView
	template< typename T, MemSizeType N >
	using NDArrayView = NDArrayViewImpl< T, MemSizeType, N >;


}


#endif // !ND_ARRAY_BASE_H
