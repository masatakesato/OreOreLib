#ifndef ND_ARRAY_PROTO_H
#define ND_ARRAY_PROTO_H

#include	<math.h>
#include	<limits>

#include	<oreore/common/TString.h>
//#include	<oreore/mathlib/Random.h>

//#include	<oreore/container/Array.h>
//#include	<oreore/container/NDShape.h>
#include	"NDArrayBase.h"


//TODO: Disable subscript operator

namespace OreOreLib
{

	//######################################################################//
	//																		//
	//						Array class implementation						//
	//																		//
	//######################################################################//


	template< typename T, uint64 N >
	class /*NDArray_proto*/NDArrayBase< T, N > : public Array<T>
	{
	public:

		// Default constructor
		/*NDArray_proto*/NDArrayBase()
			: m_Shape()
			, Array<T>()
		{
		
		}


		// Constructor
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		/*NDArray_proto*/NDArrayBase( Args const & ... args )
			: m_Shape( args... )
		{
			Memory<T>::Init( int(m_Shape.Size()) );
		}


		// Constructor with initializer list
		template < typename T_INDEX, std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value >* = nullptr >
		/*NDArray_proto*/NDArrayBase( std::initializer_list<T_INDEX> ilist )
			: m_Shape( ilist )
		{
			Memory<T>::Init( int(m_Shape.Size()) );
		}


		// Constructor using Memory
		/*NDArray_proto*/NDArrayBase( const Memory<T>& obj )
			: Array<T>( obj )
			, m_Shape( obj.Length() )
		{
		
		}


		// Copy constructor
		/*NDArray_proto*/NDArrayBase( const /*NDArray_proto*/NDArrayBase& obj )
			: Array<T>( obj )
			, m_Shape( obj.m_Shape )
		{
		
		}


		// Move constructor
		/*NDArray_proto*/NDArrayBase( /*NDArray_proto*/NDArrayBase&& obj )
			: Array<T>( obj )
			, m_Shape( obj.m_Shape )
		{
		
		}


		// Copy Assignment opertor =
		inline /*NDArray_proto*/NDArrayBase& operator=( const /*NDArray_proto*/NDArrayBase& obj )
		{
			Memory<T>::operator=( obj );
			m_Shape = obj.m_Shape;
			return *this;
		}


		inline /*NDArray_proto*/NDArrayBase& operator=( const Memory<T>& obj )
		{
			Memory<T>::operator=( obj );
			m_Shape.Init( obj.Length() );
			return *this;
		}


		// Move assignment opertor =
		inline /*NDArray_proto*/NDArrayBase& operator=( /*NDArray_proto*/NDArrayBase&& obj )
		{
			Memory<T>::operator=( (/*NDArray_proto*/NDArrayBase&&)obj );
			m_Shape = obj.m_Shape;
			return *this;
		}



		void Init( const Memory<T>& obj )
		{
			Array<T>::Init( obj );
			m_Shape.Init( obj.Length() );
		}


		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		Init( const T* const pdata, const Args& ... args )
		{
			m_Shape.Init( args... );
			Memory<T>::Init( m_Shape.Size(), pdata );
		}


		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, void >
		Init( std::initializer_list<T_INDEX> ilist )
		{
			m_Shape.Init( ilist );
			Memory<T>::Init( int(m_Shape.Size()) );
		}


		void Release()
		{
			Array<T>::Release();
			m_Shape.Release();
		}


		inline void Swap( int i, int j )
		{
			assert( i>=0 && i<this->m_Length && j>=0 && j<this->m_Length );

			if( i==j ) return;

			T tmp = this->m_pData[i];
			this->m_pData[i] = this->m_pData[j];
			this->m_pData[j] = tmp;
		}


		//================= Subscript operators(variadic templates) ===================//

		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( args... ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( args... ) ];
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( args... ) ];
		}


		//================= Subscript operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( indexND ) ];
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

				tcout << _T(": ") << this->m_pData[i] << tendl;
			}

			tcout << tendl;
		}


		// Disable subscript operators
		//const T& operator[]( std::size_t n ) const& = delete;
		//T& operator[]( std::size_t n ) & = delete;
		//T operator[]( std::size_t n ) const&& = delete;


		const NDShape<N>& Shape() const { return m_Shape; }


	private:

		NDShape<N> m_Shape;


		//using Memory<T>::operator[];
		//using Memory<T>::begin;
		//using Memory<T>::end;

	};



}// end of namespace


#endif /* ND_ARRAY_PROTO_H */