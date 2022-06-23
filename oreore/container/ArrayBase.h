#ifndef ARRAY_BASE_H
#define	ARRAY_BASE_H


#include	"../memory/Memory.h"


namespace OreOreLib
{
	namespace detail
	{

	//##########################################################################//
	//																			//
	//							ArrayBase templates								//
	//																			//
	//##########################################################################//

		const sizeType DynamicSize = (~0u);
		//-1;


		//================= ArrayView specific structs =================//

		template< typename T >	struct ARRVIEW{ using Type = typename T; };

		template< typename >
		struct is_arrview : std::false_type{};

		template< typename T >
		struct is_arrview< ARRVIEW<T> > : std::true_type{};

		template< typename T >
		constexpr bool is_arrview_v = is_arrview<T>::value;

		//==============================================================//

	}


	// ArrayBase declaration
	template< typename T, sizeType Size, typename IndexType, typename enable=void > class ArrayBase; 




	//##########################################################################//
	//																			//
	//						ArrayBase partial specialization					//
	//																			//
	//##########################################################################//

	// Dynamic array
	template< typename T, typename IndexType >
	using ArrayImpl = ArrayBase< T, detail::DynamicSize, IndexType >;

	// Static array
	template< typename T, sizeType Size, typename IndexType >
	using StaticArrayImpl = ArrayBase< T, Size, IndexType, std::enable_if_t< Size!=detail::DynamicSize > >;

	// Array view
	template< typename T, typename IndexType >
	using ArrayViewImpl = ArrayBase< detail::ARRVIEW<T>, detail::DynamicSize, IndexType >;




	//##########################################################################//
	//																			//
	//						ArrayBase full specialization						//
	//																			//
	//##########################################################################//

	// Dynamic array
	template< typename T >
	using Array = ArrayImpl< T, MemSizeType >;

	// Static array
	template< typename T, sizeType Size >
	using StaticArray = StaticArrayImpl< T, Size, MemSizeType >;

	// Array view
	template< typename T >
	using ArrayView = ArrayViewImpl< T, MemSizeType >;


}



#endif // !ARRAY_BASE_H
