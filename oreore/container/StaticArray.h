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




	template< typename T, sizeType Size >
	class ArrayBase< T, Size,std::enable_if_t< Size!=detail::DynamicSize > > : public Memory<T>
	{
		using SizeType = typename Memory<T>::SizeType;

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
		ArrayBase( SizeType len, T* pdata )
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
		ArrayBase( const Memory<T> &obj )
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

		inline ArrayBase& operator=( const Memory<T>& obj )
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


		// Subscription operator. ( called by following cases: "T& a = StaticMemory<T,10>[n]", "auto&& a = Memory<T,20>[n]" )
		inline T operator[]( std::size_t n ) const&&
		{
			return std::move(m_Data[n]);// return object
		}


		inline void Clear()
		{
			memset( m_Data, 0, sizeof(T) * Size );
		}


		SizeType Length() const
		{
			return Size;
		}


		inline void Swap( SizeType i, SizeType j )
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

			for( SizeType i=0; i<Size; ++i )
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
		void Init( SizeType ) = delete;
		void Init( SizeType, const T& ) = delete;
		template < typename ... Args >	void Init( Args const & ... args ) = delete;
		void Release() = delete;
		//void Clear() = delete;
		bool Resize( SizeType ) = delete;
		bool Resize( SizeType, const T& ) = delete;
		bool Reserve( SizeType ) = delete;
		bool Extend( SizeType ) = delete;
		bool Shrink( SizeType ) = delete;
		SizeType InsertBefore( SizeType ) = delete;
		SizeType InsertBefore( SizeType, const T& ) = delete;
		SizeType InsertBefore( SizeType, T&& ) = delete;
		SizeType InsertAfter( SizeType ) = delete;
		SizeType InsertAfter( SizeType, const T& ) = delete;
		SizeType InsertAfter( SizeType, T&& ) = delete;

		// Hide parent methods
		using Memory<T>::Init;
		using Memory<T>::Release;
		//using Memory<T>::Clear;
		using Memory<T>::Reserve;
		using Memory<T>::Resize;
		using Memory<T>::Extend;
		using Memory<T>::Shrink;
		using Memory<T>::InsertBefore;
		using Memory<T>::InsertAfter;

	};


}// end of namespace


#endif // !STATIC_ARRAY_H