#ifndef ND_ARRAY_PROTO_H
#define ND_ARRAY_PROTO_H

#include	<math.h>
#include	<limits>

#include	<oreore/common/TString.h>
//#include	<oreore/mathlib/Random.h>

#include	<oreore/container/Array.h>
#include	<oreore/container/NDShape.h>


//TODO: Disable subscript operator

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
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		NDArray_proto( Args const & ... args )
			: m_Shape( args... )
		{
			Memory<T>::Init( int(m_Shape.Size()) );
		}


		// Constructor with initializer list
		template < typename T_INDEX, std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value >* = nullptr >
		NDArray_proto( std::initializer_list<T_INDEX> ilist )
			: m_Shape( ilist )
		{
			Memory<T>::Init( int(m_Shape.Size()) );
		}


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

			for( int i=0; i<this->m_Length; ++i )
			{
				tcout << _T("  ");
				for( int dim=(int)m_Shape.NumDims()-1; dim>=0; --dim )
					tcout << _T("[") << m_Shape.ToND(i, dim) << _T("]");

				tcout << _T(": ") << *(this->begin() + i) << tendl;
			}

			tcout << tendl;
		}


		// Disable subscript operators
		const T& operator[]( std::size_t n ) const& = delete;
		T& operator[]( std::size_t n ) & = delete;
		T operator[]( std::size_t n ) const&& = delete;



	private:

		NDShape<N> m_Shape;


		using Memory<T>::operator[];

	};



}// end of namespace


#endif /* ND_ARRAY_PROTO_H */