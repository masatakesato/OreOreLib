#ifndef ND_ARRAY_H
#define ND_ARRAY_H

#include	<oreore/common/TString.h>
#include	"NDArrayBase.h"



namespace OreOreLib
{

	template< typename T, typename IndexType, IndexType N >
	class NDArrayBase< T, IndexType, N > : public ArrayImpl<T, IndexType>
	{
	public:

		// Default constructor
		NDArrayBase()
			: ArrayImpl<T, IndexType>()
		{
		
		}


		// Constructor
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value >* = nullptr >
		NDArrayBase( Args const & ... args )
			: m_Shape( args... )
		{
			MemoryBase<T, IndexType>::Init( m_Shape.Size() );
		}


		// Constructor with initializer list
		template < typename T_INDEX, std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value >* = nullptr >
		NDArrayBase( std::initializer_list<T_INDEX> ilist )
			: m_Shape( ilist )
		{
			MemoryBase<T, IndexType>::Init( m_Shape.Size() );
		}


		// Constructor using NDArrayBase
		template< typename Type, typename IDTYPE, IDTYPE ... Ns, std::enable_if_t< (sizeof...(Ns)==N) >* = nullptr >
		NDArrayBase( const NDArrayBase<Type, IDTYPE, Ns...>& obj )
			: ArrayImpl<T, IndexType>( obj )
			, m_Shape( obj.Shape() )
		{
			//tcout << _T( "NDArray::NDArray( const NDArrayBase<Type, Ns...>& obj )...\n" );
		}


		// Constructor( NDArrayView specific )
		//NDArrayBase( const NDArrayView<T, N>& obj )
		//	: ArrayImpl<T, IndexType>( obj.Shape().Size() )
		//	, m_Shape( obj.Shape() )
		//{
		//	//tcout << _T( "NDArray::NDArray( const NDArrayView<T, N>& obj )...\n" );

		//	for( IndexType i=0; i<this->m_Length; ++i )
		//		this->m_pData[i] = obj[i];
		//}


		// Destructor
		~NDArrayBase()
		{
			Release();
		}


		// Copy constructor. 
		NDArrayBase( const NDArrayBase& obj )
			: ArrayImpl<T, IndexType>( obj )
			, m_Shape( obj.m_Shape )
		{
		
		}


		// Move constructor
		NDArrayBase( NDArrayBase&& obj )
			: ArrayImpl<T, IndexType>( obj )
			, m_Shape( obj.m_Shape )
		{
		
		}


		// Copy Assignment opertor =
		inline NDArrayBase& operator=( const NDArrayBase& obj )
		{
			MemoryBase<T, IndexType>::operator=( obj );
			m_Shape = obj.m_Shape;
			return *this;
		}


		inline NDArrayBase& operator=( const MemoryBase<T, IndexType>& obj )
		{
			MemoryBase<T, IndexType>::operator=( obj );
			m_Shape.Init( obj.Length() );
			return *this;
		}


		// Move assignment opertor =
		inline NDArrayBase& operator=( NDArrayBase&& obj )
		{
			MemoryBase<T, IndexType>::operator=( (NDArrayBase&&)obj );
			m_Shape = obj.m_Shape;
			return *this;
		}


		void Init( const MemoryBase<T, IndexType>& obj )
		{
			ArrayImpl<T, IndexType>::Init( obj );
			m_Shape.Init( obj.Length() );
		}


		//template < typename ... Args >
		//std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, void >
		//Init( const Args& ... args )
		//{
		//	m_Shape.Init( args... );
		//	MemoryBase<T, IndexType>::Init( m_Shape.Size() );
		//}


		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, void >
		Init( std::initializer_list<T_INDEX> ilist )
		{
			m_Shape.Init( ilist );
			MemoryBase<T, IndexType>::Init( m_Shape.Size() );
		}


		void Release()
		{
			ArrayImpl<T, IndexType>::Release();
			m_Shape.Release();
		}


		inline void Swap( IndexType i, IndexType j )
		{
			ASSERT( i<this->m_Length && j<this->m_Length );

			if( i==j ) return;

			T tmp = this->m_pData[i];
			this->m_pData[i] = this->m_pData[j];
			this->m_pData[j] = tmp;
		}


		//================= Subscript operators(variadic templates) ===================//

		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( { static_cast<IndexType>(args)... }/*{ args... }*/ ) ];// faster
			// return this->m_pData[ m_Shape.To1D( args... ) ];// slower
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( { static_cast<IndexType>(args)... }/*{ args... }*/ ) ];// faster
			//return this->m_pData[ m_Shape.To1D( args... ) ];// slower
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( { static_cast<IndexType>(args)... }/*{ args... }*/ ) ];// faster
			//return (T&&)this->m_pData[ m_Shape.To1D( args... ) ];// slower
		}


		//================= Subscript operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		const NDShape<N, IndexType>& Shape() const
		{
			return m_Shape;
		}


		template < typename T_INDEX=IndexType >
		std::enable_if_t< std::is_convertible_v< T_INDEX, IndexType >, T_INDEX >
		Dim( IndexType i ) const
		{
			return m_Shape.Dim<T_INDEX>(i);
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			uint32 dims[N];

			for( IndexType i=0; i<this->m_Length; ++i )
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

		NDShape<N, IndexType> m_Shape;


		using MemoryBase<T, IndexType>::Init;
		using MemoryBase<T, IndexType>::Reserve;
		using MemoryBase<T, IndexType>::Reallocate;
		//using MemoryBase<T, IndexType>::Resize;
		//using MemoryBase<T, IndexType>::Extend;
		//using MemoryBase<T, IndexType>::Shrink;
		using MemoryBase<T, IndexType>::operator[];
		//using MemoryBase<T, IndexType>::begin;
		//using MemoryBase<T, IndexType>::end;

	};



}// end of namespace


#endif /* ND_ARRAY_H */