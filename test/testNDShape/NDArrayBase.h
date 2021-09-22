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

		//================= NDArrayView specific structs =================//

		template< typename T >	struct NDARRVIEW{ using Type = typename T; };

		//template< typename >
		//struct is_ndarrview : std::false_type{};

		//template< typename T >
		//struct is_ndarrview< NDARRVIEW<T> > : std::true_type{};

		//template< typename T >
		//constexpr bool is_ndarrview_v = is_ndarrview<T>::value;


		template< typename T >	struct NDSTATICARR{ using Type = typename T; };


		//==============================================================//

	}


	template< typename T, uint64 ...Ns > class NDArrayBase; 


	// NDArray
	template< typename T, uint64 N >
	using NDArray_proto = NDArrayBase< T, N >;


	// NDStaticArray
	template< typename T, uint64 ... Args >
	using NDStaticArray_proto = NDArrayBase< detail::NDSTATICARR<T>, Args... >;


	// NDArrayView
	template< typename T, uint64 N >
	using NDArrayView_proto = NDArrayBase< detail::NDARRVIEW<T>, N >;
	

}


#endif // !ND_ARRAY_BASE_H
