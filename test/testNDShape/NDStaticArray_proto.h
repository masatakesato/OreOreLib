#ifndef ND_STATIC_ARRAY_PROTO_H
#define ND_STATIC_ARRAY_PROTO_H

#include	<oreore/common/Utility.h>
#include	<oreore/common/TString.h>
#include	<oreore/meta/PeripheralTraits.h>

#include	<oreore/container/NDShape.h>
#include	<oreore/container/StaticArray.h>



namespace OreOreLib
{


	template< typename T, typename ... Args >
	class StaticNDArray : public StaticArray<T, Size>
	{
	public:

		// Default constructor
		template < std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		StaticNDArray( Args ... args )
		{
			m_Shape( args... );
			StaticArray<T>::Init();

			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;

			memset( m_Data, 0, sizeof(T) * Size );
		}


		// Constructor with external buffer
		StaticNDArray( int len, T* pdata )
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;

			memset( m_Data, 0, sizeof(T) * Size );
			MemCopy( m_Data, pdata, len );
		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
		//StaticNDArray( Args const & ... args )
		//	: m_Data{ args... }
		//{
		//	
		//}


		// Constructor
		StaticNDArray( std::initializer_list<T> ilist )
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;

			auto p = m_Data;
			for( const auto& val : ilist )
			{
				if( p==end() )	break;
				*(p++) = val;
			}
		}


		// Constructor
		StaticNDArray( const Memory<T> &obj )
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;

			MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Destructor
		~StaticNDArray()
		{
			this->m_pData = nullptr;
		}


		// Copy constructor
		StaticNDArray( const StaticNDArray& obj )
		{
			MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Move constructor.
		StaticNDArray( StaticNDArray&& obj )
		{
			MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Copy Assignment opertor =
		inline StaticNDArray& operator=( const StaticNDArray& obj )
		{
			if( this != &obj )
			{
				MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}
			return *this;
		}

		inline StaticNDArray& operator=( const Memory<T>& obj )
		{
			if( this != &obj )
			{
				MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}

			return *this;
		}


		// Move assignment opertor.
		StaticNDArray& operator=( StaticNDArray&& obj )
		{
			if( this != &obj )
			{
				MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}

			return *this;
		}


		// Subscription operator for read only.( called if StaticMemory is const )
		inline const T& operator[]( std::size_t n ) const&
		{
			return m_Data[n];
		}


		// Subscription operator for read-write.( called if StaticMemory is non-const )
		inline T& operator[]( std::size_t n ) &
		{
			return m_Data[n];
		}


		// Subscription operator. ( called by following cases: "T& a = StaticMemory<T,10>[n]", "auto&& a = Memory<T,20>[n]" )
		inline T operator[]( std::size_t n ) const&&
		{
			return std::move(m_Data[n]);// return object
		}


		inline void Clear()
		{
			memset( m_Data, 0, sizeof(T) * Size );
		}


		int Length() const
		{
			return Size;
		}


		inline void Swap( int i, int j )
		{
			assert( i>=0 && i<this->length && j>=0 && j<this->length );

			if( i==j ) return;

			T tmp = m_Data[i];
			m_Data[i] = m_Data[j];
			m_Data[j] = tmp;
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			for( int i=0; i<Size; ++i )
				tcout << _T("  [") << i << _T("]: ") << m_Data[i] << tendl;

			tcout << tendl;
		}


		// https://stackoverflow.com/questions/31581880/overloading-cbegin-cend
		// begin / end overload for "range-based for loop"
		inline T* begin()
		{
			return m_Data;
		}

		inline const T* begin() const
		{
			return m_Data;
		}

		inline T* end()
		{
			return begin() + Size;
		}

		inline const T* end() const
		{
			return begin() + Size;
		}


		// Delete unnecessary parent methods
		void Init( int len, uint8* pdata=nullptr ) = delete;
		template < typename ... Args >	void Init( Args const & ... args ) = delete;
		void Release() = delete;
		//void Clear() = delete;
		bool Resize( int newlen ) = delete;
		bool Extend( int numelms ) = delete;
		bool Shrink( int numelms ) = delete;


	private:

		NDShape<sizeof...(Args)>	m_Shape;


	};




}// end of namespace


#endif // !ND_STATIC_ARRAY_PROTO_H