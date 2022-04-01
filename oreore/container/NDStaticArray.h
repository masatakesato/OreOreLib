#ifndef ND_STATIC_ARRAY_PROTO_H
#define ND_STATIC_ARRAY_PROTO_H

#include	<oreore/common/Utility.h>
#include	<oreore/common/TString.h>
#include	<oreore/meta/PeripheralTraits.h>

//#include	<oreore/container/NDShape.h>
//#include	<oreore/container/StaticArray.h>
#include	"NDArrayBase.h"



namespace OreOreLib
{

	template< typename T, typename IndexType, IndexType ... Args >
	class NDArrayBase< detail::NDSTATICARR<T>, IndexType, Args... > : public StaticArrayImpl<T, mult_<Args...>::value, IndexType >
	{
		static constexpr int64 N = sizeof...(Args);
		static constexpr int64 Size = mult_<Args...>::value;

	public:

		// Default constructor
		NDArrayBase()
			: StaticArrayImpl<T, Size, IndexType>()
		{
			//tcout << _T( "NDStaticArray::NDStaticArray()...\n" );
		}


		// Constructor with external buffer
		NDArrayBase( IndexType len, T* pdata )
			: StaticArrayImpl<T, Size, IndexType>( len, pdata )
		{
			//tcout << _T( "NDStaticArray::NDStaticArray( IndexType len, T* pdata )...\n" );
		}


		// Constructor with initial data( variadic tempalte )
		template < typename ... Vals, std::enable_if_t< (sizeof...(Vals)==Size) && TypeTraits::all_convertible<T, Vals...>::value >* = nullptr >
		NDArrayBase( const Vals& ... vals )
			: StaticArrayImpl<T, Size, IndexType>( {(T)vals...} )
		{
			//tcout << _T( "NDStaticArray::NDStaticArray( const Vals& ... vals )...\n" );
		}


		// Constructor with initial data( initializer list )
		NDArrayBase( std::initializer_list<T> ilist )
			: StaticArrayImpl<T, Size, IndexType>( ilist )
		{
			//tcout << _T( "NDStaticArray::NDStaticArray( std::initializer_list<Type> ilist )...\n" );
		}


		// Destructor
		~NDArrayBase()
		{
			this->m_pData = nullptr;
		}


		// Copy constructor
		NDArrayBase( const NDArrayBase& obj )
		{
			MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Move constructor.
		NDArrayBase( NDArrayBase&& obj )
		{
			MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Copy Assignment opertor =
		inline NDArrayBase& operator=( const NDArrayBase& obj )
		{
			if( this != &obj )
			{
				MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}
			return *this;
		}

		inline NDArrayBase& operator=( const Memory<T, IndexType>& obj )
		{
			if( this != &obj )
			{
				MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}

			return *this;
		}


		// Move assignment opertor.
		NDArrayBase& operator=( NDArrayBase&& obj )
		{
			if( this != &obj )
			{
				MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}

			return *this;
		}



		//================= Element access operators(variadic templates) ===================//

		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
			return this->m_Data[ m_Shape.To1D( { static_cast<IndexType>(args)... } ) ];
			//return this->m_Data[ m_Shape.To1D( args... ) ];// slower
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			return this->m_Data[ m_Shape.To1D( { static_cast<IndexType>(args)... } ) ];
			//return this->m_Data[ m_Shape.To1D( args... ) ];// slower
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		//template < typename ... Args >
		//std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible< IndexType, Args... >::value, T >
		//operator()( const Args& ... args ) const&&// x, y, z, w...
		//{
		//	return (T&&)this->m_pData[ m_Shape.To1D( { static_cast<IndexType>(args)... } ) ];
		//	//return (T&&)this->m_pData[ m_Shape.To1D( args... ) ];// slower
		//}


		//================= Element access operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
			return this->m_Data[ m_Shape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
			return this->m_Data[ m_Shape.To1D( indexND ) ];
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		//template < typename T_INDEX >
		//std::enable_if_t< std::is_convertible< T_INDEX, IndexType >::value, T >
		//operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		//{
		//	return (T&&)this->m_pData[ m_Shape.To1D( indexND ) ];
		//}


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

			IndexType dims[N];

			for( int i=0; i<Size; ++i )
			{
				m_Shape.ToND( i, dims );

				tcout << _T("  ");
				for( int j=0; j<N; ++j )	tcout << _T("[") << dims[j] << _T("]");

				tcout << _T(": ") << this->m_Data[i] << tendl;
			}

			tcout << tendl;
		}


		// Disable subscript operators
		//const T& operator[]( std::size_t n ) const& = delete;
		//T& operator[]( std::size_t n ) & = delete;
		//T operator[]( std::size_t n ) const&& = delete;



	private:

		const NDShape<N, IndexType> m_Shape = NDShape<N, IndexType>(Args...);


		//using Memory<T, IndexType>::operator[];
		//using Memory<T, IndexType>::begin;
		//using Memory<T, IndexType>::end;

	};





}// end of namespace


#endif // !ND_STATIC_ARRAY_PROTO_H