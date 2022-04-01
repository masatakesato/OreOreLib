#ifndef STATIC_ARRAY_H
#define STATIC_ARRAY_H

#include	<math.h>
#include	<limits>

#include	"../common/Utility.h"
#include	"../common/TString.h"
//#include	"../memory/Memory.h"

#include	"ArrayBase.h"



namespace OreOreLib
{




	template< typename T, sizeType Size, typename InexType >
	class ArrayBase< T, Size, InexType, std::enable_if_t< Size!=detail::DynamicSize > > : public MemoryBase<T, InexType>
	{
	public:

		// Default constructor
		ArrayBase()
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;
			this->m_Capacity	= Size;

			memset( m_Data, 0, sizeof(T) * Size );
		}


		// Constructor with external buffer
		ArrayBase( InexType len, T* pdata )
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;
			this->m_Capacity	= Size;

			memset( m_Data, 0, sizeof(T) * Size );
			MemCopy( m_Data, pdata, len );
		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
		//ArrayBase( Args const & ... args )
		//	: m_Data{ args... }
		//{
		//	
		//}


		// Constructor
		ArrayBase( std::initializer_list<T> ilist )
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;
			this->m_Capacity	= Size;

			auto p = m_Data;
			for( const auto& val : ilist )
			{
				if( p==end() )	break;
				*(p++) = val;
			}
		}


		// Constructor
		ArrayBase( const MemoryBase<T, InexType> &obj )
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;
			this->m_Capacity	= Size;

			MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Destructor
		~ArrayBase()
		{
			this->m_pData = nullptr;
		}


		// Copy constructor
		ArrayBase( const ArrayBase& obj )
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;
			this->m_Capacity	= Size;

			MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Move constructor.
		ArrayBase( ArrayBase&& obj )
		{
			this->m_pData		= m_Data;
			this->m_Length		= Size;
			this->m_AllocSize	= sizeof(T) * Size;
			this->m_Capacity	= Size;

			MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
		}


		// Copy Assignment opertor =
		inline ArrayBase& operator=( const ArrayBase& obj )
		{
			if( this != &obj )
			{
				MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}
			return *this;
		}

		inline ArrayBase& operator=( const MemoryBase<T, InexType>& obj )
		{
			if( this != &obj )
			{
				MemCopy( m_Data, obj.begin(), Min( this->m_Length, obj.Length() ) );
			}

			return *this;
		}


		// Move assignment opertor.
		ArrayBase& operator=( ArrayBase&& obj )
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


		// Subscription operator. ( called by following cases: "T& a = StaticMemory<T,10>[n]", "auto&& a = MemoryBase<T,20>[n]" )
		inline T operator[]( std::size_t n ) const&&
		{
			return std::move(m_Data[n]);// return object
		}


		inline void Clear()
		{
			memset( m_Data, 0, sizeof(T) * Size );
		}


		InexType Length() const
		{
			return Size;
		}


		inline void Swap( InexType i, InexType j )
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

			for( InexType i=0; i<Size; ++i )
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


	protected:

		// Static array
		T	m_Data[Size];


	private:

		// Delete unnecessary parent methods
		void Init( InexType ) = delete;
		void Init( InexType, const T& ) = delete;
		template < typename ... Args >	void Init( Args const & ... args ) = delete;
		void Release() = delete;
		//void Clear() = delete;
		bool Resize( InexType ) = delete;
		bool Resize( InexType, const T& ) = delete;
		bool Reserve( InexType ) = delete;
		bool Extend( InexType ) = delete;
		bool Shrink( InexType ) = delete;
		InexType InsertBefore( InexType ) = delete;
		InexType InsertBefore( InexType, const T& ) = delete;
		InexType InsertBefore( InexType, T&& ) = delete;
		InexType InsertAfter( InexType ) = delete;
		InexType InsertAfter( InexType, const T& ) = delete;
		InexType InsertAfter( InexType, T&& ) = delete;

		// Hide parent methods
		using MemoryBase<T,InexType>::Init;
		using MemoryBase<T,InexType>::Release;
		//using MemoryBase<T,InexType>::Clear;
		using MemoryBase<T,InexType>::Reserve;
		using MemoryBase<T,InexType>::Resize;
		using MemoryBase<T,InexType>::Extend;
		using MemoryBase<T,InexType>::Shrink;
		using MemoryBase<T,InexType>::InsertBefore;
		using MemoryBase<T,InexType>::InsertAfter;

	};


}// end of namespace


#endif // !STATIC_ARRAY_H