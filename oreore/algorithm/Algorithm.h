#ifndef ALGORITHMS_H
#define	ALGORITHMS_H

#include	<iterator>

#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"



namespace OreOreLib
{


	//##############################################################################################################//
	//																												//
	//												Min/Max elements												//
	//																												//
	//##############################################################################################################//

	template< typename ForwardIterator >
	inline ForwardIterator MinElement( ForwardIterator first, ForwardIterator last )
	{
		if( first==last )
			return first;

		ForwardIterator result = first++;

		for(; first != last; ++first )
		{
			if( *first < *result )
				result = first;
		}
		return result;
	}



	template< typename ForwardIterator, class Compare >
	inline ForwardIterator MinElement( ForwardIterator first, ForwardIterator last, Compare comp )
	{
		if( first==last )
			return first;

		ForwardIterator result = first++;

		for(; first != last; ++first )
		{
			if( comp(*first, *result) )
				result = first;
		}
		return result;
	}



	template< typename ForwardIterator >
	inline ForwardIterator MaxElement( ForwardIterator first, ForwardIterator last )
	{
		if( first==last )
			return first;

		ForwardIterator result = first++;

		for(; first != last; ++first )
		{
			if( *first > *result )
				result = first;
		}
		return result;
	}



	template< typename ForwardIterator, class Compare >
	inline ForwardIterator MaxElement( ForwardIterator first, ForwardIterator last, Compare comp )
	{
		if( first==last )
			return first;

		ForwardIterator result = first++;

		for(; first != last; ++first )
		{
			if( comp(*result, *first) )
				result = first;
		}
		return result;
	}



	//##############################################################################################################//
	//																												//
	//													Distance													//
	//																												//
	//##############################################################################################################//

	namespace detail
	{
		template < class InputIterator >
		typename std::iterator_traits<InputIterator>::difference_type
			distance_impl( InputIterator first, InputIterator last, std::input_iterator_tag )
		{
			using result_type = typename std::iterator_traits<InputIterator>::difference_type;

			result_type n = 0;
			for(; first != last; ++first )
				++n;

			return n;
		}
	

		template < class RandomAccessIterator >
		typename std::iterator_traits<RandomAccessIterator>::difference_type
			distance_impl( RandomAccessIterator first, RandomAccessIterator last, std::random_access_iterator_tag )
		{
			return last - first;
		}

	}// end of namespace detail



	template < class InputIterator >
	typename std::iterator_traits<InputIterator>::difference_type 
		Distance( InputIterator first, InputIterator last )
	{
		return detail::distance_impl( first, last, typename std::iterator_traits<InputIterator>::iterator_category() );
	}



	//##############################################################################################################//
	//																												//
	//												ArgMin/ArgMax													//
	//																												//
	//##############################################################################################################//


	template< typename ForwardIterator >
	inline sizeType ArgMin( ForwardIterator first, ForwardIterator last )
	{
		return Distance( first, MinElement( first, last ) );////std::distance( first, std::min_element( first, last ) );
	}



	template< typename ForwardIterator, class Compare >
	inline sizeType ArgMin( ForwardIterator first, ForwardIterator last, Compare comp )
	{
		return Distance( first, MinElement( first, last, comp ) );//std::distance( first, std::min_element( first, last, comp ) );
	}



	template< typename ForwardIterator >
	inline sizeType ArgMax( ForwardIterator first, ForwardIterator last )
	{
		return Distance( first, /*std::max_element( first, last )*/MaxElement( first, last ) );
	}



	template< typename ForwardIterator, class Compare >
	inline sizeType ArgMax( ForwardIterator first, ForwardIterator last, Compare comp )
	{
		return Distance( first, /*std::max_element( first, last )*/MaxElement( first, last, comp ) );
	}




	//##############################################################################################################//
	//																												//
	//												Element manipulaion												//
	//																												//
	//##############################################################################################################//


	template < class ForwardIter, class T >
	inline void Fill( ForwardIter first, ForwardIter last, const T& value )
	{
		while( first != last )
			*first++ = value;
	}



	// https://stackoverflow.com/questions/58598763/how-to-assign-variadic-template-arguments-to-stdarray
	template < typename ForwardIterator, typename T, typename ... Args >
	std::enable_if_t< TypeTraits::all_same<T, Args...>::value, void >
	SetValues( ForwardIterator first, ForwardIterator last, Args const & ... args )
	{
		for( const auto& val : std::initializer_list<T>{args...} )
		{
			*first++ = val;
			if( first == last )	break;
		}
	}



	//template < typename ForwardIterator, typename T >
	//void Clear( ForwardIterator first, ForwardIterator last )
	//{
	//	while( first != last )
	//	{
	//		first->~T();

	//	}
	//}





}// end of namespace OreOreLib

#endif // !ALGORITHMS_H
