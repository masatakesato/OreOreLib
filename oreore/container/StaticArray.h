#ifndef STATIC_ARRAY_H
#define STATIC_ARRAY_H

#include	<math.h>
#include	<limits>

#include	"../common/Utility.h"
#include	"../common/TString.h"
#include	"../memory/Memory.h"



namespace OreOreLib
{


	template< typename T, size_t Size >
	class StaticArray : public Memory<T>
	{
	public:

		// Default constructor
		StaticArray()
		{
			memset( m_Data, 0, sizeof(T) * Size );
		}


		// Constructor with external buffer
		StaticArray( int len, T* pdata )
		{
			memset( m_Data, 0, sizeof(T) * Size );
			memcpy_s( m_Data, sizeof(T) * Size, pdata, sizeof(T) * len );
		}


		// Constructor
		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
		StaticArray( Args const & ... args )
			: m_Data{ args... }
		{
			
		}


		// Copy constructor
		StaticArray( const StaticArray& obj )
		{
			memcpy_s( m_Data, sizeof(T) * Size, obj.m_Data, sizeof(T) * Size );
		}


		// Move constructor
		StaticArray( StaticArray&& obj )
		{
			memcpy_s( m_Data, sizeof(T) * Size, obj.m_Data, sizeof(T) * Size );
		}


		// Copy Assignment opertor =
		inline StaticArray& operator=( const StaticArray& obj )
		{
			memcpy_s( m_Data, sizeof(T) * Size, obj.m_Data, sizeof(T) * Size );
			return *this;
		}


		// Move assignment opertor =
		inline StaticArray& operator=( StaticArray&& obj )
		{
			memcpy_s( m_Data, sizeof(T) * Size, obj.m_Data, sizeof(T) * Size );
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


		inline void CopyFrom( const StaticArray& src )
		{
			size_t copy_size = sizeof(T) * Size;
			memcpy_s( m_Data, copy_size, src.m_Data, copy_size );
		}


		inline void CopyTo( StaticArray& dst ) const
		{
			size_t copy_size = sizeof(T) * Size;
			memcpy_s( dst.m_Data, copy_size, m_Data, copy_size );
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
		void CopyFrom( const Memory<T>& src ) = delete;
		void CopyTo( Memory<T>& dst ) = delete;



	private:

		// Static array
		T	m_Data[Size];


		// Hide parent methods
		using Memory<T>::Init;
		using Memory<T>::Release;
		//using Memory<T>::Clear;
		using Memory<T>::Resize;
		using Memory<T>::Extend;
		using Memory<T>::Shrink;
		using Memory<T>::CopyFrom;
		using Memory<T>::CopyTo;

	};




/*
	template< typename T, size_t Size >
	class StaticArray : public Memory<T>
	{
	public:

		// default constructor
		StaticArray()
			: Memory<T>( Size, m_Data )
		{

		}

		// constructor
		//StaticArray( int len ): Memory<T>(len) { this->Clear(); }

		// constructor with external buffer
		StaticArray( int len, T* pdata )
			: Memory<T>( Size, m_Data )
		{
			memcpy_s( m_Data, sizeof(T) * Size, pdata, len );
		}


		// copy constructor
		StaticArray( const StaticArray& obj )
			: Memory<T>( Size, m_Data )
		{
			memcpy_s( m_Data, sizeof(T) * Size, obj.m_Data, sizeof(T) * Size );
		}


		// move constructor
		StaticArray( StaticArray&& obj )
			: Memory<T>( Size, m_Data )
		{
			memcpy_s( m_Data, sizeof(T) * Size, obj.m_Data, sizeof(T) * Size );
		}

		// copy Assignment opertor =
		inline StaticArray& operator=( const StaticArray& obj )
		{
			memcpy_s( m_Data, sizeof(T) * Size, obj.m_Data, sizeof(T) * Size );
			return *this;
		}

		// move assignment opertor =
		inline StaticArray& operator=( StaticArray&& obj )
		{
			memcpy_s( m_Data, sizeof(T) * Size, obj.m_Data, sizeof(T) * Size );
			return *this;
		}


		//inline int AddToFront()
		//{
		//	return InsertBefore( 0 );
		//}

		//inline int AddToTail()
		//{
		//	return InsertBefore( this->length );
		//}

		//inline int InsertBefore( int elm )
		//{
		//	if( this->Extend( 1 )==false )
		//		return -1;
		//	ShiftElementsRight( elm );
		//	this->pData[elm] = T();
		//	return elm;
		//}

		//inline int InsertAfter( int elm )
		//{
		//	return InsertBefore( elm+1 );
		//}

		
		//inline int AddToFront( const T& src )
		//{
		//	return InsertBefore( 0, src );
		//}

		//inline int AddToTail( const T& src )
		//{
		//	return InsertBefore( this->length, src );
		//}

		//inline int InsertBefore( int elm, const T& src )
		//{
		//	if( this->Extend( 1 )==false )
		//		return -1;
		//	ShiftElementsRight( elm );
		//	this->pData[elm] = src;
		//	return elm;
		//}

		//inline int InsertAfter( int elm, const T& src )
		//{
		//	return InsertBefore( elm+1, src );
		//}


		//inline void FastRemove( int elm )// 削除対象の要素を配列最後尾要素で上書きする & メモリ確保サイズ自体は変更せずlenghデクリメントする
		//{
		//	assert( elm>=0 && elm<this->length );

		//	this->pData[elm].~T();
		//	if( this->length > 0 )
		//	{
		//		if( elm != this->length - 1 )
		//			memcpy( &this->pData[elm], &this->pData[this->length-1], this->data_size );
		//		--this->length;
		//	}

		//	if( this->length==0 )
		//		this->Release();
		//}
		
		//inline void Remove( int elm )
		//{
		//	assert( elm>=0 && elm<this->length );

		//	this->pData[elm].~T();
		//	ShiftElementsLeft( elm );
		//	--this->length;

		//	if( this->length==0 )
		//		this->Release();
		//}

		inline void Swap( int i, int j )
		{
			assert( i>=0 && i<this->length && j>=0 && j<this->length );

			if( i==j ) return;

			T tmp = m_Data[i];
			m_Data[i] = m_Data[j];
			m_Data[j] = tmp;
		}


		void Display()
		{
			tcout << _T("//========== Static Array ===========//\n" );
			for( int i=0; i<Size; ++i )
				tcout << _T("[") << i << _T("]: ") << m_Data[i] << tendl;

		}


		//void Release()
		//{

		//}


		// Delete unnecessary parent methods
		bool Extend( int numelms ) = delete;
		void Init( int len, T* pdata ) = delete;
		void Release() = delete;
		bool Resize( int newlen ) = delete;//using Memory<T>::Resize;
		bool Shrink( int numelms ) = delete;



	private:

		// Static array
		T	m_Data[Size];

		inline void ShiftElementsRight( int elm, int num=1 )
		{
			int numtomove = this->length - elm - num;
			if( numtomove > 0 )
				memmove( &this->pData[elm+num], &this->pData[elm], numtomove * this->data_size );
		}

		inline void ShiftElementsLeft( int elm, int num=1 )
		{
			int numtomove = this->length - elm - num;
			if( numtomove > 0 )
				memmove( &this->pData[elm], &this->pData[elm+num], numtomove * this->data_size );
		}


		// Hide parent methods
		//using Memory<T>::Extend;
		//using Memory<T>::Init;
		//using Memory<T>::Release;
		//using Memory<T>::Resize;
		//using Memory<T>::Shrink;

	};
*/








}// end of namespace


#endif // !STATIC_ARRAY_H