#ifndef NDARRAY_PROTO_H
#define NDARRAY_PROTO_H

#include	<math.h>
#include	<limits>

#include	<oreore/common/TString.h>
//#include	<oreore/mathlib/Random.h>

#include	<oreore/container/Array.h>
#include	<oreore/container/NDShape.h>




namespace OreOreLib
{

	//######################################################################//
	//																		//
	//						Array class implementation						//
	//																		//
	//######################################################################//


	template< typename T, uint64 N >
	class NDArray_proto : public Array<T>
	{
	public:

		// Default constructor
		NDArray_proto()
			: m_Shape()
			, Array<T>()
		{
		
		}

		// Constructor
//		NDArray_proto( int len ) : Array<T>(len) {}
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		NDArray_proto( Args const & ... args )
			: m_Shape( args... )
		{
			this->Init( int(m_Shape.Size()) );
		}


		// Constructor
//		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
//		NDArray_proto( Args const & ... args ) : Memory<T>( args ...) {}

		// Constructor with initializer list
//		NDArray_proto( std::initializer_list<T> ilist ) : Array<T>( ilist ) {}



		// Constructor with external buffer
//		NDArray_proto( int len, T* pdata ): Array<T>( len, pdata ) {}

		// Constructor using Memory
		NDArray_proto( const Memory<T>& obj )
			: Array<T>( obj )
			, m_Shape( obj.Length() )
		{
		
		}


		// Copy constructor
		NDArray_proto( const NDArray_proto& obj )
			: Array<T>( obj )
			, m_Shape( obj.m_Shape )
		{
		
		}


		// Move constructor
		NDArray_proto( NDArray_proto&& obj )
			: Array<T>( obj )
			, m_Shape( obj.m_Shape )
		{
		
		}


		// Copy Assignment opertor =
		inline NDArray_proto& operator=( const NDArray_proto& obj )
		{
			Memory<T>::operator=( obj );
			m_Shape = obj.m_Shape;
			return *this;
		}


		inline NDArray_proto& operator=( const Memory<T>& obj )
		{
			Memory<T>::operator=( obj );
			m_Shape.Init( obj.Length() );
			return *this;
		}


		// Move assignment opertor =
		inline NDArray_proto& operator=( NDArray_proto&& obj )
		{
			Memory<T>::operator=( (NDArray_proto&&)obj );
			m_Shape = obj.m_Shape;
			return *this;
		}


		inline void Swap( int i, int j )
		{
			assert( i>=0 && i<this->m_Length && j>=0 && j<this->m_Length );

			if( i==j ) return;

			T tmp = this->m_pData[i];
			this->m_pData[i] = this->m_pData[j];
			this->m_pData[j] = tmp;
		}


		//================= Subscription operators(variadic templates) ===================//

		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, const T& >
		operator()( Args ... args ) const&// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( args... ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T& >
		operator()( Args ... args ) &// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( args... ) ];
		}


		// Subscription operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T >
		operator()( Args ... args ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( args... ) ];
		}


		//================= Subscription operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		std::enable_if_t< std::is_convertible<uint64, T>::value, const T& >
		operator()( std::initializer_list<T> indexND ) const&// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		std::enable_if_t< std::is_convertible<uint64, T>::value, T& >
		operator()( std::initializer_list<T> indexND ) &// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Subscription operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		std::enable_if_t< std::is_convertible<uint64, T>::value, T >
		operator()( std::initializer_list<T> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( indexND ) ];
		}




		void Display() const
		{
			tcout << typeid(*this).name() << _T("[ ") << this->m_Length << _T(" ]:\n" );

			for( int i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}



	private:

		NDShape<N> m_Shape;

	};



}// end of namespace


#endif /* NDARRAY_PROTO_H */