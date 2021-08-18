﻿#ifndef TYPE_TRAITS_H
#define	TYPE_TRAITS_H

#include	<type_traits>


namespace OreOreLib
{
	namespace TypeTraits
	{

		namespace detail
		{
			template< bool... > struct bool_pack;
			template< bool... bs >
			using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;
		}


		template < typename... Ts >
		using all_true = detail::all_true< Ts::value... >;


		template < typename T, typename... Ts >
		using all_same = all_true< std::is_same<T, Ts>... >;

	}

}


#endif // !TYPE_TRAITS_H
