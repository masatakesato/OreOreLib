#ifndef ND_STATIC_ARRAY_PROTO_H
#define ND_STATIC_ARRAY_PROTO_H

#include	<oreore/common/Utility.h>
#include	<oreore/common/TString.h>
#include	<oreore/meta/PeripheralTraits.h>

#include	<oreore/container/NDShape.h>
#include	<oreore/container/StaticArray.h>



namespace OreOreLib
{


	template< typename T, unsigned ... Args >
	class NDStaticArray_proto : public StaticArray<T, mult_<Args...>::value >
	{
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
			MemCopy( this->begin(), obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Move constructor.
		NDStaticArray_proto( NDStaticArray_proto&& obj )
		{
			MemCopy( this->begin(), obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		//// Copy Assignment opertor =
		//inline NDStaticArray_proto& operator=( const NDStaticArray_proto& obj )
		//{
		//	if( this != &obj )
		//	{
		//		MemCopy( begin(), obj.begin(), Min( this->m_Length, obj.Length() ) );
		//	}
		//	return *this;
		//}

		//inline NDStaticArray_proto& operator=( const Memory<T>& obj )
		//{
		//	if( this != &obj )
		//	{
		//		MemCopy( begin(), obj.begin(), Min( this->m_Length, obj.Length() ) );
		//	}

		//	return *this;
		//}


		// Move assignment opertor.
		//NDStaticArray_proto& operator=( NDStaticArray_proto&& obj )
		//{
		//	if( this != &obj )
		//	{
		//		MemCopy( begin(), obj.begin(), Min( this->m_Length, obj.Length() ) );
		//	}

		//	return *this;
		//}


		//// Subscription operator for read only.( called if StaticMemory is const )
		//inline const T& operator[]( std::size_t n ) const&
		//{
		//	return m_Data[n];
		//}


		//// Subscription operator for read-write.( called if StaticMemory is non-const )
		//inline T& operator[]( std::size_t n ) &
		//{
		//	return m_Data[n];
		//}


		//// Subscription operator. ( called by following cases: "T& a = StaticMemory<T,10>[n]", "auto&& a = Memory<T,20>[n]" )
		//inline T operator[]( std::size_t n ) const&&
		//{
		//	return std::move(m_Data[n]);// return object
		//}


		//inline void Clear()
		//{
		//	memset( m_Data, 0, sizeof(T) * Size );
		//}


		//int Length() const
		//{
		//	return Size;
		//}


		//inline void Swap( int i, int j )
		//{
		//	assert( i>=0 && i<this->length && j>=0 && j<this->length );

		//	if( i==j ) return;

		//	T tmp = m_Data[i];
		//	m_Data[i] = m_Data[j];
		//	m_Data[j] = tmp;
		//}


		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			static uint64 indexND[ sizeof...(Args) ];

			for( int i=0; i<Size; ++i )
			{
				m_Shape.ToND( i, indexND );
				for( int j=Dim-1; j>=0; --j )
					tcout << _T("[") << indexND[j] << _T("]");

				tcout << _T(": ") << *(this->begin() + i) << tendl;
			}

			tcout << tendl;
		}


		// https://stackoverflow.com/questions/31581880/overloading-cbegin-cend
		// begin / end overload for "range-based for loop"
		//inline T* begin()
		//{
		//	return m_Data;
		//}

		//inline const T* begin() const
		//{
		//	return m_Data;
		//}

		//inline T* end()
		//{
		//	return begin() + Size;
		//}

		//inline const T* end() const
		//{
		//	return begin() + Size;
		//}


		// Delete unnecessary parent methods
		void Init( int len, uint8* pdata=nullptr ) = delete;
		template < typename ... Args >	void Init( Args const & ... args ) = delete;
		void Release() = delete;
		//void Clear() = delete;
		bool Resize( int newlen ) = delete;
		bool Extend( int numelms ) = delete;
		bool Shrink( int numelms ) = delete;


	private:

		static constexpr size_t Dim = sizeof...(Args);
		static constexpr size_t Size = mult_<Args...>::value;
		const NDShape<Dim> m_Shape = NDShape<Dim>(Args...);

	};




}// end of namespace


#endif // !ND_STATIC_ARRAY_PROTO_H