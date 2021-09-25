﻿#ifndef ND_SHAPE_H
#define	ND_SHAPE_H

#include	"../common/TString.h"
#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"



//###############################################################################################################//
//
//  C-style array and initializer_list performance comparison using NDShape::ToND and NDShape::To1D method.
//
//	+-------------------+------------+-------------+
//	|					|    read    |    write    |
//	+-------------------+------------+-------------+
//	|   C-style array	|    slower  |    faster   |
//	+-------------------+------------+-------------+
//	| initializer_list  |    faster  |    slower   |
//	+-------------------+------------+-------------+
//
//   2021.09.21. VS2017(v141)
//
//###############################################################################################################//




namespace OreOreLib
{

	// https://stackoverflow.com/questions/32921192/c-variadic-template-limit-number-of-args

	template < int64 N >
	class NDShape
	{
	public:

		using SHAPE_TYPE = uint32;


		// Default constructor
		NDShape()
			: m_Shape{ 0 } 
			, m_Strides{ 0 } 
		{
		
		}


		// Constructor(variadic tempaltes)
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<SHAPE_TYPE, Args...>::value >* = nullptr >
		NDShape( const Args& ... args )// ...w, z, y, x
			: m_Shape{ SHAPE_TYPE(args)... }
		{
			InitStrides();
		}


		// Constructor(initializer list)
		template < typename T, std::enable_if_t< std::is_convertible<T, SHAPE_TYPE>::value >* = nullptr >
		NDShape( std::initializer_list<T> indexND )// ...w, z, y, x
		{
			Init( indexND );
		}


		// Destructor
		~NDShape()
		{
			Release();
		}


		// Copy constructor
		NDShape( const NDShape& obj )
		{
			MemCopy( m_Shape, obj.m_Shape, N );
			MemCopy( m_Strides, obj.m_Strides, N );
		}


		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<SHAPE_TYPE, Args...>::value, void >
		Init( const Args& ... args )// ...w, z, y, x
		{
			auto p = m_Shape;
			for( const auto& val : { args... } )
				*(p++) = (SHAPE_TYPE)val;

			InitStrides();
		}


		template < typename T >
		std::enable_if_t< std::is_convertible<T, SHAPE_TYPE>::value, void >
		Init( std::initializer_list<T> indexND )// ...w, z, y, x
		{
			auto p = m_Shape;
			for( const auto& val : indexND )
				*(p++) = (SHAPE_TYPE)val;

			InitStrides();
		}


		template < typename T >
		std::enable_if_t< std::is_convertible<T, SHAPE_TYPE>::value, void >
		Init( const T indexND[] ) 
		{
			MemCopy( m_Shape, indexND, N );
			InitStrides();
		}


		void Release()
		{
			for( int i=1; i<N; ++i )
				m_Shape[i] = m_Strides[i] = 0;
		}



		//============== ND to 1D index conversion ===============//
		// ...
		// indexND[0] * m_Strides[2];	// w
		// indexND[1] * m_Strides[1] +	// z
		// indexND[2] * m_Strides[0] +	// y
		// indexND[3] +					// x

		template < typename T=SHAPE_TYPE, typename ... Args >// variadic template version
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<T, Args...>::value, T >
		To1D( const Args& ... args ) const// ...w, z, y, x
		{
			//using T = typename TypeTraits::first_type<Args...>::type;
			auto indexND = { &args... };
			auto itr = std::rbegin( indexND );
			
			T index = **itr++;
			auto offset = m_Strides;

			while( itr !=std::rend( indexND ) )
				index += (T)**(itr++) * (T)*(offset++);

			return index;
		}


		template < typename T=SHAPE_TYPE >// initializer_list version
		std::enable_if_t< std::is_convertible<SHAPE_TYPE, T>::value, T >
		To1D( std::initializer_list<T> indexND ) const// ...w, z, y, x
		{
			auto itr = std::rbegin( indexND );

			T index = (T)*(itr++);
			auto offset = m_Strides;

			while( itr !=std::rend( indexND ) )
				index += (T)*(itr++) * (T)*(offset++);

			return index;
		}


		template < typename T=SHAPE_TYPE >
		std::enable_if_t< std::is_convertible_v<T, SHAPE_TYPE>, T >
		To1D( const T indexND[] ) const
		{
			T index = indexND[N-1];
			for( int i=N-2; i>=0; --i )	index += indexND[i] * (T)m_Strides[N-2-i];
			return index;

			//indexND[0] * m_Strides[2];	// w
			//indexND[1] * m_Strides[1] +	// z
			//indexND[2] * m_Strides[0] +	// y
			//indexND[3] +					// x
		}


		template < typename T >
		std::enable_if_t< std::is_convertible_v<T, uint64>, int64 >
		From3DTo1D( const T& z, const T& y, const T& x ) const
		{
			return z * m_Strides[1] + y * m_Strides[0] + x;
		}



		//=============== 1D to ND index conversion ===============//
		// indexND[0]	= ( id / m_Strides[2] );
		// indexND[1]	= ( id % m_Strides[2] ) / m_Strides[1];
		// indexND[2]	= ( id % m_Strides[2] ) % m_Strides[1] / m_Strides[0];
		// indexND[3]	= ( id % m_Strides[2] ) % m_Strides[1] % m_Strides[0];
		// ...

		template < typename T_INDEX=SHAPE_TYPE, typename T=SHAPE_TYPE >
		std::enable_if_t< std::is_convertible_v<T_INDEX, T>, T* >
		ToND( T_INDEX indexd1D, T indexND[] ) const
		{
			indexND[ 0 ] = (T)indexd1D;
			for( int i=0; i<N-1; ++i )
			{
				T str = (T)m_Strides[ N-2-i ];
				indexND[ i+1 ] = indexND[ i ] % str;
				indexND[ i ] /= str;
			}

			return indexND;
		}


		//　Single dim only
		template < typename T_INDEX=SHAPE_TYPE, typename T=SHAPE_TYPE >
		std::enable_if_t< std::is_convertible_v<T_INDEX, T>, T >
		ToND( T_INDEX indexd1D, uint16 dim ) const
		{
			T index = (T)indexd1D;
			for( int i=1; i<=dim; ++i )
				index = index % (T)m_Strides[ N-1-i ];

			if( dim < N-1 )
				index /= (T)m_Strides[ N-2-dim ];

			return index;
		}


		template < typename T_INDEX=SHAPE_TYPE, typename ... Args >// C-style array version
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<T_INDEX, Args...>::value, void >
		ToND( T_INDEX index1D, Args& ... args ) const
		{
			using T = typename TypeTraits::first_type<Args...>::type;
			T* indexND[ N ] = { &args... };

			*indexND[ 0 ] = (T)index1D;
			for( int i=0; i<N-1; ++i )
			{
				T& stride = (T&)m_Strides[ N-2-i ];
				*indexND[ i+1 ] = *indexND[ i ] % stride;
				*indexND[ i ] /= stride;
			}
		}
		//template < typename ... Args >// initializer_list version. slower than above implementation. 
		//std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		//ToND( uint64 index1D, Args& ... args ) const
		//{
		//	using T = typename TypeTraits::first_type<Args...>::type;

		//	auto indexND = { &args... };

		//	auto iter_i = (T**)indexND.begin();
		//	**iter_i = (T)index1D;//indexND[N-1] = (T)indexd1D;

		//	auto iter_i_1 = iter_i;
		//	auto iter_stride = &m_Strides[N-2];
		//	while( ++iter_i != indexND.end() )
		//		**iter_i = **iter_i_1++ % *iter_stride--;
		//	
		//	iter_i = (T**)indexND.begin();
		//	iter_stride = &m_Strides[N-2];
		//	while( iter_stride >= m_Strides )
		//		**iter_i++ /= (T)*iter_stride--;
		//}


		template < typename T=SHAPE_TYPE >
		/*uint64*/T NumDims() const
		{
			return (T)N;
		}


		template < typename T=SHAPE_TYPE >
		T Dim( int32 i ) const
		{
			return (T)m_Shape[i];//i<N ? m_Shape[ i ] : 0;
		}


		template < typename T=SHAPE_TYPE >
		T Size() const
		{
			return (T)m_Strides[ N-1 ];
		}


		void Disiplay()
		{
			tcout << typeid(*this).name() << _T(":\n");
			tcout << _T("  Shape = [");
			for( int i=0; i<N; ++i )
				tcout << m_Shape[i] << ( i==N-1 ? _T("];\n") : _T(", ") );
		}



	private:

		SHAPE_TYPE	m_Shape[ N ];
		SHAPE_TYPE	m_Strides[ N ];// strides for multidimensional element access.


		const void InitStrides()
		{
			m_Strides[0] = m_Shape[N-1];
			for( int i=0; i<N-1; ++i )
				m_Strides[ i+1 ] = m_Strides[ i ] * m_Shape[ N-2-i ];

			//m_Strides[0]	= m_Shape[3];
			//m_Strides[1]	= m_Strides[0] * m_Shape[2];
			//m_Strides[2]	= m_Strides[1] * m_Shape[1];
			//m_Strides[3]	= m_Strides[2] * m_Shape[0];
			//...
		}


	};


}// end of namespace



#endif // !ND_SHAPE_H