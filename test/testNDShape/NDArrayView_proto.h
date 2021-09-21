#ifndef ND_ARRAY_VIEW_PROTO_H
#define	ND_ARRAY_VIEW_PROTO_H

#include	<oreore/common/TString.h>

//#include	<oreore/container/ArrayView.h>
//#include	<oreore/container/NDShape.h>

#include	"NDArrayBase.h"


// https://www.codeproject.com/Articles/848746/ArrayView-StringView


//TODO: Disable Subscript operator

namespace OreOreLib
{


	template< typename T, uint64 N >
	class NDArrayBase< detail::NDARRVIEW<T>, N > : public ArrayView<T>
	{
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		// Default constructor
		NDArrayBase()
			: ArrayView<T>()
		{

		}


		// Constructor
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		NDArrayBase( ConstPtr const pdata, const Args& ... args )
			: m_Shape( mult_<args...>::value )
			, m_SrcShape( m_Shape )
		{
			Init( pdata, args... );
		}


		// Constructor
		NDArrayBase( const Memory<T>& obj )
			: m_Shape( obj.Length() )
			, m_SrcShape( m_Shape )
		{
			Init( obj );
		}





// Constructor using NDArray_proto(variadic template ver)
template < uint64 N, typename ... Args, std::enable_if_t< (sizeof...(Args)==2*N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
NDArrayBase( const NDArray_proto<T, N>& obj, const Args& ... args )
{
	Init( obj.begin(), obj.Shape(), args... );
}


		// Constructor using NDArray_proto(initializer_list ver)
		template < uint64 N, typename T_INDEX, std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value>* = nullptr >
		NDArrayBase( const NDArray_proto<T, N>& obj, std::initializer_list<T_INDEX> offset, std::initializer_list<T_INDEX> indexND )
			: m_Shape( indexND )
			, m_SrcShape( obj.Shape() )
		{
			//ArrayView<T>::Init( obj.begin() + m_SrcShape.To1D( offset ), int(m_Shape.Size()) );
		}



// Constructor using NDStaticArray_proto
template < uint64 ... Args, typename ... Args2, std::enable_if_t< (sizeof...(Args)==2*N) && TypeTraits::all_convertible<uint64, Args2...>::value >* = nullptr >
NDArrayBase( const NDStaticArray_proto<T, Args...>& obj, const Args2& ... args )
{
	Init( obj.begin(), obj.Shape(), args... );
}


// Constructor using NDStaticArray_proto
template < uint64 ... Args, typename T_INDEX, std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value>* = nullptr >
NDArrayBase( const NDStaticArray_proto<T, Args...>& obj, std::initializer_list<T_INDEX> offset, std::initializer_list<T_INDEX> indexND )
	//: m_Shape( indexND )
	//, m_SrcShape( obj.Shape() )
{
	Init( obj.begin(), obj.Shape(), offset, indexND );
	//ArrayView<T>::Init( obj.begin() + m_SrcShape.To1D( offset ), int(m_Shape.Size()) );
}




		// Destructor
		~NDArrayBase()
		{
			Release();
		}


		// Copy constructor
		NDArrayBase( const NDArrayBase& obj )
			: ArrayView( obj )
			, m_Shape( obj.m_Shape )
		{

		}


		//================= Element access operators(variadic templates) ===================//

		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( args... ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( args... ) ];
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_SrcShape.To1D( args... ) ];
		}



		//================= Element acces operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}



		void Init( const Memory<T>& obj )
		{
			ArrayView<T>::Init( obj );
			m_Shape.Init( obj.Length() );
		}






template < uint64 N, typename ... Args >
std::enable_if_t< (sizeof...(Args)==2*N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
Init( const T* ptr, const NDShape<N>& srcShape, const Args& ... args )
{
	uint64 offset[N], indexND[N];

	auto itr = std::begin( {args...} );

	for( int i=0; i<N; ++i )
		offset[i] = *itr++;

	for( int i=0; i<N; ++i )
		indexND[i] = *itr;

	m_SrcShape = srcShape;
	m_Shape.Init( indexND );
	ArrayView<T>::Init( ptr + m_SrcShape.To1D( offset ), int(m_Shape.Size()) );
}


template < uint64 N, typename T_INDEX >
std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, void >
Init( const T* ptr, const NDShape<N>& srcShape, std::initializer_list<T_INDEX> offset, std::initializer_list<T_INDEX> indexND )
{
	m_SrcShape = srcShape;
	m_Shape.Init( indexND );			
	ArrayView<T>::Init( ptr + m_SrcShape.To1D( offset ), int(m_Shape.Size()) );
}






		//template < typename ... Args >
		//std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		//Init( ConstPtr const pdata, const Args& ... args )
		//{
		//	m_Shape.Init( args... );
		//	ArrayView<T>::Init( pdata, m_Shape.Size() );
		//}


		//template < typename T_INDEX >
		//std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, void >
		//Init( ConstPtr const pdata, std::initializer_list<T_INDEX> indexND )
		//{
		//	m_Shape.Init( indexND );
		//	ArrayView<T>::Init( pdata, int(m_Shape.Size()) );
		//}


		void Release()
		{
			ArrayView<T>::Release();
			m_Shape.Release();
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			uint64 dims[N];

			for( int i=0; i<this->m_Length; ++i )
			{
				m_Shape.ToND( i, dims );

				tcout << _T("  ");
				for( int j=N-1; j>=0; --j )	tcout << _T("[") << dims[j] << _T("]");

				uint64 idx = m_SrcShape.To1D( dims );

				tcout << _T(": ") << this->m_pData[ idx ] << tendl;
			}

			tcout << tendl;
		}		


		// Disabled subscript operators
		const T& operator[]( std::size_t n ) const& = delete;
		T& operator[]( std::size_t n ) & = delete;
		T operator[]( std::size_t n ) const&& = delete;


		const NDShape<N>& Shape() const { return m_Shape; }


	private:

		NDShape<N>	m_Shape;
		NDShape<N>	m_SrcShape;


		using Memory<T>::operator[];
		using Memory<T>::begin;
		using Memory<T>::end;
	};


}// end of namespace


#endif // !ND_ARRAY_VIEW_PROTO_H
