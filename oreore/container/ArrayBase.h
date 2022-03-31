#ifndef ARRAY_BASE_H
#define	ARRAY_BASE_H


#include	"../memory/Memory.h"


namespace OreOreLib
{
	namespace detail
	{
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


	template< typename T, sizeType Size, typename SizeType, typename enable=void > class ArrayBase; 


	// Dynamic array
	template< typename T, typename SizeType=MemSizeType >
	using Array = ArrayBase< T, detail::DynamicSize, SizeType >;


	// Static array
	template< typename T, sizeType Size, typename SizeType=MemSizeType >
	using StaticArray = ArrayBase< T, Size, SizeType, std::enable_if_t< Size!=detail::DynamicSize > >;


	// Array view
	template< typename T, typename SizeType=MemSizeType >
	using ArrayView = ArrayBase< detail::ARRVIEW<T>, detail::DynamicSize, SizeType >;


}



#endif // !ARRAY_BASE_H
