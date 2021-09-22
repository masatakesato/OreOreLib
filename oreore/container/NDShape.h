#ifndef ND_SHAPE_H
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

	template < uint64 N >
	class NDShape
	{
	public:

		// Default constructor
		NDShape()
			: m_Shape{ 0 } 
			, m_Strides{ 0 } 
		{
		
		}


		// Constructor(variadic tempaltes)
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		NDShape( const Args& ... args )// x, y, z, w...
			: m_Shape{ uint64(args)... }
		{
			InitStrides();
		}


		// Constructor(initializer list)
		template < typename T, std::enable_if_t< std::is_convertible<uint64, T>::value >* = nullptr >
		NDShape( std::initializer_list<T> indexND )// x, y, z, w...
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
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		Init( const Args& ... args )
		{
			auto p = m_Shape;
			for( const auto& val : { args... } )
				*(p++) = val;

			InitStrides();
		}


		template < typename T >
		std::enable_if_t< std::is_convertible<uint64, T>::value, void >
		Init( std::initializer_list<T> indexND )
		{
			auto p = m_Shape;
			for( const auto& val : indexND )
				*(p++) = val;

			InitStrides();
		}


template < typename T >
std::enable_if_t< std::is_convertible<uint64, T>::value, void >
Init( const T indexND[] ) 
{
	MemCopy( m_Shape, indexND, N );
//	for( int i=1; i<N; ++i )
//		m_Shape[i] = indexND[i];

	InitStrides();
}


		void Release()
		{
			for( int i=1; i<N; ++i )
				m_Shape[i] = m_Strides[i] = 0;
		}



		//============== ND to 1D index conversion ===============//
		// indexND[0] +				// x
		// indexND[1] * m_Strides[0] +	// y
		// indexND[2] * m_Strides[1] +	// z
		// indexND[3] * m_Strides[2];	// w
		// ...

		template < typename ... Args >// initializer_list version
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, int64 >
		To1D( const Args& ... args ) const// x, y, z, w...
		{
			auto indexND = { args... };
			auto itr = std::begin( indexND );

			uint64 index = *(itr++);
			auto offset = m_Strides;

			while( itr !=std::end( indexND ) )
				index += *(itr++) * *(offset++);

			return index;
		}
		//template < typename ... Args >// C-style array version. slower than above implementation. 
		//std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, int64 >
		//To1D( const Args& ... args ) const// x, y, z, w...
		//{
		//	using T = typename TypeTraits::first_type<Args...>::type;
		//	const T* indexND[N] = { &args... };
		//	
		//	int64 index = (T)*indexND[0];
		//	for( int i=1; i<N; ++i )	index += *indexND[i] * m_Strides[i-1];
		//	return index;
		//}


		template < typename T >
		std::enable_if_t< std::is_convertible<uint64, T>::value, int64 >
		To1D( std::initializer_list<T> indexND ) const// x, y, z, w...
		{
			auto itr = std::begin( indexND );

			uint64 index = (uint64)*(itr++);
			auto offset = m_Strides;

			while( itr !=std::end( indexND ) )
				index += (uint64)*(itr++) * *(offset++);

			return index;
		}


		template < typename T >
		std::enable_if_t< std::is_convertible_v<T, uint64>, int64 >
		To1D( const T indexND[] ) const
		{
			int64 index = indexND[0];
			for( int i=1; i<N; ++i )	index += indexND[i] * m_Strides[i-1];
			return index;

			//indexND[0] +					// x
			//indexND[1] * m_Strides[0] +	// y
			//indexND[2] * m_Strides[1] +	// z
			//indexND[3] * m_Strides[2];	// w
		}


		template < typename T >
		std::enable_if_t< std::is_convertible_v<T, uint64>, int64 >
		From3DTo1D( const T& x, const T& y, const T& z ) const
		{
			return z * m_Strides[1] + y * m_Strides[0] + x;
		}



		//=============== 1D to ND index conversion ===============//
		// indexND[3]	= ( id / m_Strides[2] );
		// indexND[2]	= ( id % m_Strides[2] ) / m_Strides[1];
		// indexND[1]	= ( id % m_Strides[2] ) % m_Strides[1] / m_Strides[0];
		// indexND[0]	= ( id % m_Strides[2] ) % m_Strides[1] % m_Strides[0];
		// ...

		template < typename T >
		std::enable_if_t< std::is_convertible_v<T, uint64>, void >
		ToND( uint64 indexd1D, T indexND[] ) const
		{
			indexND[N-1] = (T)indexd1D;
			for( int i=N-2; i>=0; --i )
				indexND[i] = indexND[i+1] % (T)m_Strides[i];

			for( int i=N-1; i>=1; --i )
				indexND[i] /= (T)m_Strides[i-1];
		}


		//　Single dim only
		uint64 ToND( uint64 indexd1D, int dim ) const
		{
			uint64 index = indexd1D;
			for( int i=N-2; i>=dim; --i )
				index = index % m_Strides[i];

			if( dim > 0 )
				index /= m_Strides[dim-1];

			return index;
		}


		template < typename ... Args >// C-style array version
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		ToND( uint64 index1D, Args& ... args ) const
		{
			using T = typename TypeTraits::first_type<Args...>::type;

			T* indexND[N] = { &args... };

			*indexND[N-1] = (T)index1D;
			for( int i=N-2; i>=0; --i )
				*indexND[i] = *indexND[i+1] % (T)m_Strides[i];
			
			for( int i=N-1; i>=1; --i )
				*indexND[i] /= (T)m_Strides[i-1];
		}
		//template < typename ... Args >// initializer_list version. slower than above implementation. 
		//std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		//ToND( uint64 index1D, Args& ... args ) const
		//{
		//	using T = typename TypeTraits::first_type<Args...>::type;

		//	auto indexND = { &args... };

		//	auto iter_i = (T**)indexND.end();	iter_i--;
		//	**iter_i = (T)index1D;//indexND[N-1] = (T)indexd1D;

		//	auto iter_i_1 = iter_i;
		//	auto iter_stride = &m_Strides[N-2];
		//	while( --iter_i >= indexND.begin() )
		//		**iter_i = **iter_i_1-- % *iter_stride--;
		//	
		//	iter_i = (T**)indexND.end();
		//	iter_stride = &m_Strides[N-2];
		//	while( --iter_i > indexND.begin() )
		//		**iter_i /= (T)*iter_stride--;
		//}



		uint64 NumDims() const
		{
			return N;
		}


		uint64 Dim( uint64 i ) const
		{
			return i<N ? m_Shape[ i ] : 0;
		}


		uint64 Size() const
		{
			return m_Strides[ N-1 ];
		}


		void Disiplay()
		{
			tcout << typeid(*this).name() << _T(":\n");
			tcout << _T("  Shape = [");
			for( int i=0; i<N; ++i )
				tcout << m_Shape[i] << ( i==N-1 ? _T("];\n") : _T(", ") );
		}



	private:

		uint64	m_Shape[ N ];
		uint64	m_Strides[ N ];// strides for multidimensional element access.


		const void InitStrides()
		{
			m_Strides[0] = m_Shape[0];
			for( int i=1; i<N; ++i )	m_Strides[i] = m_Shape[i] * m_Strides[i-1];

			//m_Strides[0]	= m_Shape[0];
			//m_Strides[1]	= m_Shape[0] * m_Shape[1];
			//m_Strides[2]	= m_Shape[0] * m_Shape[1] * m_Shape[2];
			//m_Strides[3]	= m_Shape[0] * m_Shape[1] * m_Shape[2] * m_Shape[3];
			//...
		}


	};


}// end of namespace



#endif // !ND_SHAPE_H