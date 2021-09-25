#ifndef ND_ARRAY_VIEW_PROTO_H
#define	ND_ARRAY_VIEW_PROTO_H

#include	<oreore/common/TString.h>

//#include	<oreore/container/ArrayView.h>
//#include	<oreore/container/NDShape.h>

#include	"NDArrayBase.h"


// https://www.codeproject.com/Articles/848746/ArrayView-StringView



namespace OreOreLib
{

	template< typename T, int64 N >
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


		//============== Constructor using raw pointer ==================//

		// variadic template
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value >* = nullptr >
		NDArrayBase( const Args& ... args )
			: m_Shape( args... )
			, m_SrcShape( m_Shape )
		{
			ArrayView<T>::Init( m_Shape.Size() );
		}

		// initializer_list
		template < typename T_INDEX, std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value >* = nullptr >
		NDArrayBase( std::initializer_list<T_INDEX> indexND )
		{
			m_Shape.Init( indexND );
			ArrayView<T>::Init( int(m_Shape.Size()) );
		}


		//============ Constructor using NDArrayBase ============//

		// variadic template
		template< typename Type, int64 ... Ns, typename ... Args, std::enable_if_t< (sizeof...(Args)==2*N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value >* = nullptr >
		NDArrayBase( const NDArrayBase<Type, Ns...>& obj, const Args& ... args )
		{
			Init( obj, args... );
		}

		// initializer_list
		template< typename Type, int64 ...Ns, typename T_INDEX, std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value>* = nullptr >
		NDArrayBase( const NDArrayBase<Type, Ns...>& obj, std::initializer_list<T_INDEX> offset, std::initializer_list<T_INDEX> indexND )
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
			: ArrayView( obj )
			, m_Shape( obj.m_Shape )
			, m_SrcShape( obj.m_SrcShape )
		{

		}



		//================= Element access operators(variadic templates) ===================//

		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( {args...} ) ];
			//return this->m_pData[ m_SrcShape.To1D( args... ) ];// slower
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( {args...} ) ];
			//return this->m_pData[ m_SrcShape.To1D( args... ) ];//slower
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_SrcShape.To1D( {args...} ) ];
			//return (T&&)this->m_pData[ m_SrcShape.To1D( args... ) ];// slower
		}



		//================= Element acces operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
			return this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_SrcShape.To1D( indexND ) ];
		}



		//================ Init ===================//

		// raw pointer with variadic template
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, void >
		Init( const Args& ... args )
		{
			m_Shape.Init( args... );
			m_SrcShape = m_Shape;
			ArrayView<T>::Init( (int)m_Shape.Size() );
		}

		// raw pointer with initializer list
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, void >
		Init( std::initializer_list<T_INDEX> indexND )
		{
			m_Shape.Init( indexND );
			ArrayView<T>::Init( (int)m_Shape.Size() );
		}


		// NDArrayBase with variadic template
		template< typename Type, int64 ... Ns, typename ... Args >
		std::enable_if_t< (sizeof...(Args)==2*N) && TypeTraits::all_convertible< detail::ShapeType<N>, Args... >::value, bool >
		Init( const NDArrayBase<Type, Ns...>& obj, const Args& ... args )
		{
			detail::ShapeType<N> offset[N], indexND[N];

			auto itr = std::begin( {args...} );
			for( int i=0; i<N; ++i )	offset[i] = *itr++;
			for( int i=0; i<N; ++i )	indexND[i] = *itr++;

			if( !IsValidViewRange( std::begin(indexND), std::begin(offset), obj.Shape() ) )
				return false;

			m_SrcShape = obj.Shape();
			m_Shape.Init( indexND );
			ArrayView<T>::Init( obj.begin() + m_SrcShape.To1D( offset ), int(m_Shape.Size()) );

			return true;
		}

		// NDArrayBase with initializer list
		template< typename Type, int64 ...Ns, typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, detail::ShapeType<N> >::value, bool >
		Init( const NDArrayBase<Type, Ns...>& obj, std::initializer_list<T_INDEX> offset, std::initializer_list<T_INDEX> indexND )
		{
			m_SrcShape = obj.Shape();
			m_Shape.Init( indexND );

			if( !IsValidViewRange( std::begin(indexND), std::begin(offset), obj.Shape() ) )
				return false;

			ArrayView<T>::Init( obj.begin() + m_SrcShape.To1D( offset ), int(m_Shape.Size()) );

			return true;
		}


		void Release()
		{
			ArrayView<T>::Release();
			m_Shape.Release();
		}


		template < typename ... Args >
		std::enable_if_t< TypeTraits::all_convertible<T, Args...>::value, void >
		SetValues( const Args& ... args )
		{
			int64 count = (int64)Min( sizeof...(Args), (size_t)this->m_Length );
			auto src = std::begin( { (T)args... } );
			for( int64 i=0; i<count; ++i )
				(*this)[i] = *src++;
		}


		template < typename Type >
		std::enable_if_t< std::is_convertible_v<Type, T>, void >
		SetValues( std::initializer_list<Type> ilist )
		{
			int64 count = (int64)Min( ilist.size(), (size_t)this->m_Length );
			auto src = ilist.begin();
			for( int64 i=0; i<count; ++i )
				(*this)[i] = *src++;
		}


		const NDShape<N>& Shape() const
		{
			return m_Shape;
		}


		template < typename T_INDEX=detail::ShapeType<N> >
		std::enable_if_t< std::is_convertible_v< T_INDEX, detail::ShapeType<N> >, T_INDEX >
		Dim( int i ) const
		{
			return m_Shape.Dim<T_INDEX>(i);
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			detail::ShapeType<N> dims[N];

			for( int i=0; i<this->m_Length; ++i )
			{
				m_Shape.ToND( i, dims );

				tcout << _T("  ");
				for( int j=0; j<N; ++j )	tcout << _T("[") << dims[j] << _T("]");

				//detail::ShapeType<N> idx = m_SrcShape.To1D( dims );
				tcout << _T(": ") << this->m_pData[ i ] << tendl;
			}

			tcout << tendl;
		}


		// Disabled subscript operators
		//const T& operator[]( std::size_t n ) const& = delete;
		//T& operator[]( std::size_t n ) & = delete;
		T operator[]( size_t n ) const&& = delete;



		// Subscript operator for read only.( called if Memory is const )
		inline const T& operator[]( size_t n ) const&
		{
			detail::ShapeType<N> indexND[N];
			return this->m_pData[ m_SrcShape.To1D( m_Shape.ToND( n, indexND ) ) ];
		}


		// Subscript operator for read-write.( called if Memory is non-const )
		inline T& operator[]( size_t n ) &
		{
			detail::ShapeType<N> indexND[N];
			return this->m_pData[ m_SrcShape.To1D( m_Shape.ToND( n, indexND ) ) ];
		}


		// Subscript operator. ( called by following cases: "T a = Memory<T>(10)[n]", "auto&& a = Memory<T>(20)[n]" )
		//inline T operator[]( size_t n ) const&&
		//{
		//	return std::move(this->m_pData[n]);// return object
		//}



	private:

		NDShape<N>	m_Shape;
		NDShape<N>	m_SrcShape;


		template < typename Iter >
		bool IsValidViewRange( Iter range, Iter offset, const NDShape<N>& shape )
		{
			for( int i=0; i<N; ++i )
			{
				if( detail::ShapeType<N>( *( range++ ) + *( offset++ ) ) > shape.Dim(i) )
				{
					tcout << "TOO LARGE DIMENSIONS SPECIFIED...\n";
					return false;
				}
			}

			return true;
		}



		using Memory<T>::SetValues;
		using Memory<T>::operator[];
		//using Memory<T>::begin;
		//using Memory<T>::end;
	};


}// end of namespace


#endif // !ND_ARRAY_VIEW_PROTO_H
