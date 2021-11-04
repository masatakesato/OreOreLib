#ifndef ND_ARRAY_H
#define ND_ARRAY_H

#include	<oreore/common/TString.h>
#include	"NDArrayBase.h"



namespace OreOreLib
{

	template< typename T, int64 N >
	class NDArrayBase< T, N > : public Array<T>
	{
	public:

		// Default constructor
		NDArrayBase()
			: Array<T>()
		{
		
		}


		// Constructor
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value >* = nullptr >
		NDArrayBase( Args const & ... args )
			: m_Shape( args... )
		{
			Memory<T>::Init( int(m_Shape.Size()) );
		}


		// Constructor with initializer list
		template < typename T_INDEX, std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value >* = nullptr >
		NDArrayBase( std::initializer_list<T_INDEX> ilist )
			: m_Shape( ilist )
		{
			Memory<T>::Init( int(m_Shape.Size()) );
		}


		// Constructor using NDArrayBase
		template< typename Type, int64 ... Ns, std::enable_if_t< (sizeof...(Ns)==N) >* = nullptr >
		NDArrayBase( const NDArrayBase<Type, Ns...>& obj )
			: Array<T>( obj )
			, m_Shape( obj.Shape() )
		{
			//tcout << _T( "NDArray::NDArray( const NDArrayBase<Type, Ns...>& obj )...\n" );
		}


		// Constructor( NDArrayView specific )
		NDArrayBase( const NDArrayView<T, N>& obj )
			: Array<T>( (int)obj.Shape().Size() )
			, m_Shape( obj.Shape() )
		{
			//tcout << _T( "NDArray::NDArray( const NDArrayView<T, N>& obj )...\n" );

			for( int i=0; i<this->m_Length; ++i )
				this->m_pData[i] = obj[i];
		}



		// Copy constructor. 
		NDArrayBase( const NDArrayBase& obj )
			: Array<T>( obj )
			, m_Shape( obj.m_Shape )
		{
		
		}


		// Move constructor
		NDArrayBase( NDArrayBase&& obj )
			: Array<T>( obj )
			, m_Shape( obj.m_Shape )
		{
		
		}


		// Copy Assignment opertor =
		inline NDArrayBase& operator=( const NDArrayBase& obj )
		{
			Memory<T>::operator=( obj );
			m_Shape = obj.m_Shape;
			return *this;
		}


		inline NDArrayBase& operator=( const Memory<T>& obj )
		{
			Memory<T>::operator=( obj );
			m_Shape.Init( obj.Length() );
			return *this;
		}


		// Move assignment opertor =
		inline NDArrayBase& operator=( NDArrayBase&& obj )
		{
			Memory<T>::operator=( (NDArrayBase&&)obj );
			m_Shape = obj.m_Shape;
			return *this;
		}


		void Init( const Memory<T>& obj )
		{
			Array<T>::Init( obj );
			m_Shape.Init( obj.Length() );
		}


		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, void >
		Init( const Args& ... args )
		{
			m_Shape.Init( args... );
			Memory<T>::Init( int(m_Shape.Size()) );
		}


		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, void >
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
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( {args...} ) ];// faster
			// return this->m_pData[ m_Shape.To1D( args... ) ];// slower
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( {args...} ) ];// faster
			//return this->m_pData[ m_Shape.To1D( args... ) ];// slower
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( {args...} ) ];// faster
			//return (T&&)this->m_pData[ m_Shape.To1D( args... ) ];// slower
		}


		//================= Subscript operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		const NDShape<N>& Shape() const
		{
			return m_Shape;
		}


		template < typename T_INDEX=NDShape<N>::SHAPE_TYPE >
		std::enable_if_t< std::is_convertible_v< T_INDEX, detail::ShapeType<N> >, T_INDEX >
		Dim( int i ) const
		{
			return m_Shape.Dim<T_INDEX>(i);
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			uint32 dims[N];

			for( int i=0; i<this->m_Length; ++i )
			{
				m_Shape.ToND( i, dims );

				tcout << _T("  ");
				for( int j=0; j<N; ++j )	tcout << _T("[") << dims[j] << _T("]");

				tcout << _T(": ") << this->m_pData[i] << tendl;
			}

			tcout << tendl;
		}


		// Disable subscript operators
		//const T& operator[]( std::size_t n ) const& = delete;
		//T& operator[]( std::size_t n ) & = delete;
		//T operator[]( std::size_t n ) const&& = delete;



	private:

		NDShape<N> m_Shape;


		using Memory<T>::Init;
		using Memory<T>::Resize;
		using Memory<T>::Extend;
		using Memory<T>::Shrink;
		using Memory<T>::operator[];
		//using Memory<T>::begin;
		//using Memory<T>::end;

	};



}// end of namespace


#endif /* ND_ARRAY_H */