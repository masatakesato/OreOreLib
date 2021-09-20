#ifndef ND_STATIC_ARRAY_PROTO_H
#define ND_STATIC_ARRAY_PROTO_H

#include	<oreore/common/Utility.h>
#include	<oreore/common/TString.h>
#include	<oreore/meta/PeripheralTraits.h>

#include	<oreore/container/NDShape.h>
#include	<oreore/container/StaticArray.h>


//TODO: Disable Subscript operator


namespace OreOreLib
{


	template< typename T, unsigned ... Args >
	class NDStaticArray_proto : public StaticArray<T, mult_<Args...>::value >
	{
		static constexpr size_t N = sizeof...(Args);
		static constexpr size_t Size = mult_<Args...>::value;

	public:

		// Default constructor
		NDStaticArray_proto()
			: StaticArray<T, Size>()
			//, m_Shape( Args... )
		{
			
		}


		// Constructor with external buffer
		NDStaticArray_proto( int len, T* pdata )
			: StaticArray<T, Size>( len, pdata )
			, m_Shape( Args... )
		{

		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
		//NDStaticArray_proto( Args const & ... args )
		//	: m_Data{ args... }
		//{
		//	
		//}


		// Constructor
		NDStaticArray_proto( std::initializer_list<T> ilist )
			: StaticArray<T, Size>( ilist )
			, m_Shape( Args... )
		{

		}


		// Constructor
		NDStaticArray_proto( const Memory<T> &obj )
			: StaticArray<T, Size>( obj )
			, m_Shape( Args... )
		{

		}


		// Destructor
		~NDStaticArray_proto()
		{
			this->m_pData = nullptr;
		}


		// Copy constructor
		NDStaticArray_proto( const NDStaticArray_proto& obj )
		{
			MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Move constructor.
		NDStaticArray_proto( NDStaticArray_proto&& obj )
		{
			MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Copy Assignment opertor =
		inline NDStaticArray_proto& operator=( const NDStaticArray_proto& obj )
		{
			if( this != &obj )
			{
				MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}
			return *this;
		}

		inline NDStaticArray_proto& operator=( const Memory<T>& obj )
		{
			if( this != &obj )
			{
				MemCopy( this->m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}

			return *this;
		}


		// Move assignment opertor.
		NDStaticArray_proto& operator=( NDStaticArray_proto&& obj )
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
			return this->m_pData[ m_Shape.To1D( args... ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
			m_Shape.To1D( args... );
			return this->m_pData[ /*m_Shape.To1D( args... )*/0 ];
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( args... ) ];
		}


		//================= Element access operators(initializer list) ===================//

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


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( indexND ) ];
		}



		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			for( int i=0; i<Size; ++i )
			{
				tcout << _T("  ");
				for( int dim=(int)m_Shape.NumDims()-1; dim>=0; --dim )
					tcout << _T("[") << m_Shape.ToND(i, dim) << _T("]");

				tcout << _T(": ") << this->m_Data[i] << tendl;
			}

			tcout << tendl;
		}


		// Disable subscript operators
		const T& operator[]( std::size_t n ) const& = delete;
		T& operator[]( std::size_t n ) & = delete;
		T operator[]( std::size_t n ) const&& = delete;



	private:

		const NDShape<N> m_Shape = NDShape<N>(Args...);


		using Memory<T>::operator[];

	};





}// end of namespace


#endif // !ND_STATIC_ARRAY_PROTO_H