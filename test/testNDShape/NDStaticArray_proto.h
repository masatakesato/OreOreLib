#ifndef ND_STATIC_ARRAY_PROTO_H
#define ND_STATIC_ARRAY_PROTO_H

#include	<oreore/common/Utility.h>
#include	<oreore/common/TString.h>
#include	<oreore/meta/PeripheralTraits.h>

//#include	<oreore/container/NDShape.h>
//#include	<oreore/container/StaticArray.h>
#include	"NDArrayBase.h"

//TODO: Disable Subscript operator


namespace OreOreLib
{


	template< typename T, uint64 ... Args >
	class NDArrayBase< detail::NDSTATICARR<T>, Args... > : public StaticArray<T, mult_<Args...>::value >
	{
		static constexpr size_t N = sizeof...(Args);
		static constexpr size_t Size = mult_<Args...>::value;

	public:

		// Default constructor
		NDArrayBase()
			: StaticArray<T, Size>()
		{
TODO: Test			
		}


		// Constructor with external buffer
		NDArrayBase( int len, T* pdata )
			: StaticArray<T, Size>( len, pdata )
		{
TODO: Test
		}


		// Constructor with initial data( variadic tempalte )
		template < typename ... Vals, std::enable_if_t< (sizeof...(Vals)==Size) && TypeTraits::all_convertible<T, Vals...>::value >* = nullptr >
		NDArrayBase( const Vals& ... vals )
			: StaticArray<T, Size>( vals... )
		{
TODO: Test			
		}


		// Constructor with initial data( initializer list )
		template < typename Type, std::enable_if_t< std::is_convertible<T, Type>::value>* = nullptr >
		NDArrayBase( std::initializer_list<Type> ilist )
			: StaticArray<T, Size>( ilist )
		{
TODO: Test
		}



		// Constructor using NDArrayBase
		template< typename Type, uint64 ... Ns, std::enable_if_t< (sizeof...(Ns)==N) >* = nullptr >
		NDArrayBase( const NDArrayBase<Type, Ns...>& obj )
			: Array<T>( obj )
			, m_Shape( obj.m_Shape )
		{
TODO: Test
		}


		// Constructor( NDArrayView specific )
		NDArrayBase( const NDArrayView_proto<T, N>& obj )
			: Array<T>( (int)obj.Shape().Size() )
			, m_Shape( obj.Shape() )
		{
TODO: Test
			for( int i=0; i<this->m_Length; ++i )
				this->m_Data[i] = obj[i];
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

		inline NDArrayBase& operator=( const Memory<T>& obj )
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
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
			return this->m_Data + m_Shape.To1D( args... );
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			return this->m_Data + m_Shape.To1D( args... );
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		//template < typename ... Args >
		//std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T >
		//operator()( const Args& ... args ) const&&// x, y, z, w...
		//{
		//	return (T&&)this->m_pData[ m_Shape.To1D( args... ) ];
		//}


		//================= Element access operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
			return this->m_Data + m_Shape.To1D( indexND );
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
			return this->m_Data + m_Shape.To1D( indexND );
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		//template < typename T_INDEX >
		//std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T >
		//operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		//{
		//	return (T&&)this->m_pData[ m_Shape.To1D( indexND ) ];
		//}



		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			uint64 dims[N];

			for( int i=0; i<Size; ++i )
			{
				m_Shape.ToND( i, dims );

				tcout << _T("  ");
				for( int j=N-1; j>=0; --j )	tcout << _T("[") << dims[j] << _T("]");

				tcout << _T(": ") << this->m_Data[i] << tendl;
			}

			tcout << tendl;
		}


		// Disable subscript operators
		//const T& operator[]( std::size_t n ) const& = delete;
		//T& operator[]( std::size_t n ) & = delete;
		//T operator[]( std::size_t n ) const&& = delete;


		const NDShape<N>& Shape() const { return m_Shape; }


	private:

		const NDShape<N> m_Shape = NDShape<N>(Args...);


		//using Memory<T>::operator[];
		//using Memory<T>::begin;
		//using Memory<T>::end;

	};





}// end of namespace


#endif // !ND_STATIC_ARRAY_PROTO_H