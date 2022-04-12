#ifndef TYPE_ID_H
#define	TYPE_ID_H

#include	"Utility.h"



namespace OreOreLib
{

	namespace detail
	{
		
		//##########################################################//
		//															//
		//						Forward declaration					//
		//															//
		//##########################################################//

		template < typename IndexType, typename enable=void >
		struct SeqID;

		template < typename T, typename IndexType, typename enable=void >
		struct TypeIDImpl;



		//##########################################################//
		//															//
		//					SeqID implementation					//
		//															//
		//##########################################################//

		template < typename IndexType >
		class SeqID< IndexType, typename std::enable_if_t< std::is_integral_v<IndexType> > >
		{
		private:

			static IndexType Generate()
			{
				static IndexType counter = 0;
				return counter++;
			}

			template < typename T, typename IndexType, typename enable >
			friend struct TypeIDImpl;
		};




		//##########################################################//
		//															//
		//				TypeIDImpl implementation					//
		//															//
		//##########################################################//

		template < typename T, typename IndexType >
		struct TypeIDImpl< T, IndexType, typename std::enable_if_t< std::is_integral_v<IndexType> > >
		{
			static const IndexType value;
		};

		template < typename T, typename IndexType >
		const IndexType TypeIDImpl< T, IndexType, typename std::enable_if_t<std::is_integral_v<IndexType>> >::value = SeqID<IndexType>::Generate();

	}// end of namespace detail




	//##########################################################//
	//															//
	//						Specialization						//
	//															//
	//##########################################################//

	template < typename T >
	using TypeID = detail::TypeIDImpl<T, uint32>;


	template < typename T >
	using TypeID8 = detail::TypeIDImpl<T, uint8>;

	template < typename T >
	using TypeID16 = detail::TypeIDImpl<T, uint16>;

	template < typename T >
	using TypeID64 = detail::TypeIDImpl<T, uint64>;



}// end of namespace


#endif // !TYPE_ID_H