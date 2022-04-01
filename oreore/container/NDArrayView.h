#ifndef ND_ARRAY_VIEW_PROTO_H
#define	ND_ARRAY_VIEW_PROTO_H

#include	<oreore/common/TString.h>

//#include	<oreore/container/ArrayViewImpl.h>
//#include	<oreore/container/NDShape.h>

#include	"NDArrayBase.h"


// https://www.codeproject.com/Articles/848746/ArrayViewImpl-StringView



namespace OreOreLib
{

	template< typename T, typename IndexType, IndexType N >
	class NDArrayBase< detail::NDARRVIEW<T>, IndexType, N > : public ArrayViewImpl<T, IndexType>
	{
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		// Default constructor
		NDArrayBase()
			: ArrayViewImpl<T, IndexType>()
		{

		}


		//============== Constructor using raw pointer ==================//

		// variadic template
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value >* = nullptr >
		NDArrayBase( const Args& ... args )
			: m_Shape( args... )
			, m_SrcShape( m_Shape )
		{
			ArrayViewImpl<T, IndexType>::Init( m_Shape.Size() );
		}

		// initializer_list
		template < typename T_INDEX, std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value >* = nullptr >
		NDArrayBase( std::initializer_list<T_INDEX> indexND )
		{
			m_Shape.Init( indexND );
			ArrayViewImpl<T, IndexType>::Init( int(m_Shape.Size()) );
		}


		//============ Constructor using NDArrayBase ============//

		// variadic template
		template< typename Type, typename IType, IType ... Ns, typename ... Args, std::enable_if_t< (sizeof...(Args)==2*N) && TypeTraits::all_convertible< IndexType, Args... >::value >* = nullptr >
		NDArrayBase( const NDArrayBase<Type, IType, Ns...>& obj, const Args& ... args )
		{
			Init( obj, args... );
		}

		// initializer_list
		template< typename Type, typename IType, IType ...Ns, typename T_INDEX, std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value>* = nullptr >
		NDArrayBase( const NDArrayBase<Type, IType, Ns...>& obj, std::initializer_list<T_INDEX> offset, std::initializer_list<T_INDEX> indexND )
		{
			Init( obj, offset, indexND );
		}


		// Destructor
		~NDArrayBase()
		{
			Release();
		}


		// Copy constructor
		NDArrayBase( const NDArrayBase& obj )
			: ArrayViewImpl( obj )
			, m_Shape( obj.m_Shape )
			, m_SrcShape( obj.m_SrcShape )
		{

		}



		//================= Element access operators(variadic templates) ===================//

		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( { static_cast<IndexType>(args)... } ) ];
			//return this->m_pData[ m_SrcShape.To1D( args... ) ];// slower
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( { static_cast<IndexType>(args)... } ) ];
			//return this->m_pData[ m_SrcShape.To1D( args... ) ];//slower
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_SrcShape.To1D( { static_cast<IndexType>(args)... } ) ];
			//return (T&&)this->m_pData[ m_SrcShape.To1D( args... ) ];// slower
		}



		//================= Element acces operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}



		//================ Init ===================//

		// raw pointer with variadic template
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, void >
		Init( const Args& ... args )
		{
			m_Shape.Init( args... );
			m_SrcShape = m_Shape;
			ArrayViewImpl<T, IndexType>::Init( (int)m_Shape.Size() );
		}

		// raw pointer with initializer list
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, void >
		Init( std::initializer_list<T_INDEX> indexND )
		{
			m_Shape.Init( indexND );
			ArrayViewImpl<T, IndexType>::Init( (int)m_Shape.Size() );
		}


		// NDArrayBase with variadic template
		template< typename Type, typename IType, IType ... Ns, typename ... Args >
		std::enable_if_t< (sizeof...(Args)==2*N) && TypeTraits::all_convertible< IndexType, Args... >::value, bool >
		Init( const NDArrayBase<Type, IType, Ns...>& obj, const Args& ... args )
		{
			IndexType offset[N], indexND[N];

			auto itr = std::begin( { static_cast<IndexType>(args)... } );
			for( int i=0; i<N; ++i )	offset[i] = *itr++;
			for( int i=0; i<N; ++i )	indexND[i] = *itr++;

			if( !IsValidViewRange( std::begin(indexND), std::begin(offset), obj.Shape() ) )
				return false;

			m_SrcShape = obj.Shape();
			m_Shape.Init( indexND );
			ArrayViewImpl<T, IndexType>::Init( obj.begin() + m_SrcShape.To1D( offset ), int(m_Shape.Size()) );

			return true;
		}

		// NDArrayBase with initializer list
		template< typename Type, typename IType, IType ...Ns, typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, bool >
		Init( const NDArrayBase<Type, IType, Ns...>& obj, std::initializer_list<T_INDEX> offset, std::initializer_list<T_INDEX> indexND )
		{
			m_SrcShape = obj.Shape();
			m_Shape.Init( indexND );

			if( !IsValidViewRange( std::begin(indexND), std::begin(offset), obj.Shape() ) )
				return false;

			ArrayViewImpl<T, IndexType>::Init( obj.begin() + m_SrcShape.To1D( offset ), int(m_Shape.Size()) );

			return true;
		}


		void Release()
		{
			ArrayViewImpl<T, IndexType>::Release();
			m_Shape.Release();
		}


		template < typename ... Args >
		std::enable_if_t< TypeTraits::all_convertible<T, Args...>::value, void >
		SetValues( const Args& ... args )
		{
			IndexType count = Min( (IndexType)( sizeof...(Args) ), this->m_Length );
			auto src = std::begin( { static_cast<T>(args)... } );
			for( IndexType i=0; i<count; ++i )
				(*this)[i] = *src++;
		}


		template < typename Type >
		std::enable_if_t< std::is_convertible_v<Type, T>, void >
		SetValues( std::initializer_list<Type> ilist )
		{
			IndexType count = Min( (IndexType)ilist.size(), this->m_Length );
			auto src = ilist.begin();
			for( IndexType i=0; i<count; ++i )
				(*this)[i] = *src++;
		}


		const NDShape<N, IndexType>& Shape() const
		{
			return m_Shape;
		}


		template < typename T_INDEX=IndexType >
		std::enable_if_t< std::is_convertible_v< T_INDEX, IndexType >, T_INDEX >
		Dim( int i ) const
		{
			return m_Shape.Dim<T_INDEX>(i);
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			IndexType dims[N];

			for( IndexType i=0; i<this->m_Length; ++i )
			{
				m_Shape.ToND( i, dims );

				tcout << _T("  ");
				for( int j=0; j<N; ++j )	tcout << _T("[") << dims[j] << _T("]");

				//IndexType idx = m_SrcShape.To1D( dims );
				tcout << _T(": ") << this->m_pData[ i ] << tendl;
			}

			tcout << tendl;
		}


		// Disabled subscript operators
		//const T& operator[]( std::size_t n ) const& = delete;
		//T& operator[]( std::size_t n ) & = delete;
		T operator[]( IndexType ) const&& = delete;



		// Subscript operator for read only.( called if Memory is const )
		inline const T& operator[]( IndexType n ) const&
		{
			IndexType indexND[N];
			return this->m_pData[ m_SrcShape.To1D( m_Shape.ToND( n, indexND ) ) ];
		}


		// Subscript operator for read-write.( called if Memory is non-const )
		inline T& operator[]( IndexType n ) &
		{
			IndexType indexND[N];
			return this->m_pData[ m_SrcShape.To1D( m_Shape.ToND( n, indexND ) ) ];
		}


		// Subscript operator. ( called by following cases: "T a = Memory<T, IndexType>(10)[n]", "auto&& a = Memory<T, IndexType>(20)[n]" )
		//inline T operator[]( IndexType n ) const&&
		//{
		//	return std::move(this->m_pData[n]);// return object
		//}



	private:

		NDShape<N, IndexType>	m_Shape;
		NDShape<N, IndexType>	m_SrcShape;


		template < typename Iter >
		bool IsValidViewRange( Iter range, Iter offset, const NDShape<N, IndexType>& shape )
		{
			for( int i=0; i<N; ++i )
			{
				if( IndexType( *( range++ ) + *( offset++ ) ) > shape.Dim(i) )
				{
					tcout << "TOO LARGE DIMENSIONS SPECIFIED...\n";
					return false;
				}
			}

			return true;
		}



		using Memory<T, IndexType>::SetValues;
		using Memory<T, IndexType>::operator[];
		//using Memory<T, IndexType>::begin;
		//using Memory<T, IndexType>::end;
	};


}// end of namespace


#endif // !ND_ARRAY_VIEW_PROTO_H
