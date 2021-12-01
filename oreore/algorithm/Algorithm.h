#ifndef ALGORITHMS_H
#define	ALGORITHMS_H

#include	<iterator>

#include	"../common/Utility.h"



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







}// end of namespace OreOreLib

#endif // !ALGORITHMS_H
