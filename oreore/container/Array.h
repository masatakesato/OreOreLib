#ifndef ARRAY_H
#define ARRAY_H

#include	<math.h>
#include	<limits>

#include	"../common/TString.h"
//#include	"../memory/Memory.h"
#include	"../mathlib/Random.h"

#include	"ArrayBase.h"




namespace OreOreLib
{

	//######################################################################//
	//																		//
	//						Array class implementation						//
	//																		//
	//######################################################################//


	template< typename T >
	class ArrayBase< T, detail::DynamicSize > : public Memory<T>
	{
	public:

		// Default constructor
		ArrayBase(): Memory<T>() {}

		// Constructor
		ArrayBase( int len ) : Memory<T>(len) {}

		// Constructor
//		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
//		ArrayBase( Args const & ... args ) : Memory<T>( args ...) {}

		// Constructor with initializer list
		ArrayBase( std::initializer_list<T> ilist ) : Memory<T>( ilist ) {}

		// Constructor with external buffer
		ArrayBase( int len, T* pdata ): Memory<T>( len, pdata ) {}

		// Constructor using Memory
		ArrayBase( const Memory<T>& obj ) : Memory<T>( obj ) {}

		// Constructor using iterator
		template < class Iter >
		ArrayBase( Iter first, Iter last ) : Memory<T>( first, last ) {}

		// Copy constructor
		ArrayBase( const ArrayBase& obj ) : Memory<T>( obj ) {}

		// Move constructor
		ArrayBase( ArrayBase&& obj ) : Memory<T>( obj ) {}


		// Copy Assignment opertor =
		inline ArrayBase& operator=( const ArrayBase& obj )
		{
			Memory<T>::operator=( obj );
			return *this;
		}


		inline ArrayBase& operator=( const Memory<T>& obj )
		{
			Memory<T>::operator=( obj );
			return *this;
		}


		// Move assignment opertor =
		inline ArrayBase& operator=( ArrayBase&& obj )
		{
			Memory<T>::operator=( (ArrayBase&&)obj );
			return *this;
		}


		inline int AddToFront()
		{
			return InsertBefore( 0 );
		}

		inline int AddToTail()
		{
			return InsertBefore( this->m_Length );
		}

		inline int InsertBefore( int elm )
		{
			if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			this->m_pData[elm] = T();
			return elm;
		}

		inline int InsertAfter( int elm )
		{
			return InsertBefore( elm + 1 );
		}

		
		inline int AddToFront( const T& src )
		{
			return InsertBefore( 0, src );
		}

		inline int AddToFront( T&& src )
		{
			return InsertBefore( 0, src );
		}

		inline int AddToTail( const T& src )
		{
			return InsertBefore( this->m_Length, src );
		}

		inline int AddToTail( T&& src )
		{
			return InsertBefore( this->m_Length, src );
		}


		inline int InsertBefore( int elm, const T& src )
		{
			if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			this->m_pData[elm] = src;
			return elm;
		}


		inline int InsertBefore( int elm, T&& src )
		{
			if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			this->m_pData[elm] = src;
			return elm;
		}



		inline int InsertAfter( int elm, const T& src )
		{
			return InsertBefore( elm+1, src );
		}

		inline int InsertAfter( int elm, T&& src )
		{
			return InsertBefore( elm+1, src );
		}


		inline void FastRemove( int elm )// 削除対象の要素を配列最後尾要素で上書きする & メモリ確保サイズ自体は変更せずlenghデクリメントする
		{
			assert( elm>=0 && elm<this->m_Length );

			this->m_pData[elm].~T();
			if( this->m_Length > 0 )
			{
				if( elm != this->m_Length - 1 )
					this->m_pData[ elm ] = this->m_pData[ this->m_Length-1 ];
					//memcpy( &this->m_pData[elm], &this->m_pData[this->m_Length-1], this->m_ElementSize );

				--this->m_Length;
			}

			if( this->m_Length==0 )
				this->Release();
		}

		//inline void FastRemove( const T& item )// 削除対象の要素を配列最後尾要素で上書きする & メモリ確保サイズ自体は変更せずlenghデクリメントする
		//{
		//	auto index = Find( this, item );

		//	if( index==-1 )
		//		return;

		//	FastRemove( index );
		//}

		
		inline void Remove( int elm )
		{
			assert( elm>=0 && elm<this->m_Length );

			this->m_pData[elm].~T();
			ShiftElementsLeft( elm );
			--this->m_Length;

			if( this->m_Length==0 )
				this->Release();
		}

		//inline void Remove( const T& item )
		//{
		//	auto index = Find( this, item );

		//	if( index==-1 )
		//		return;

		//	Remove( index );
		//}


		inline void Swap( int i, int j )
		{
			assert( i>=0 && i<this->m_Length && j>=0 && j<this->m_Length );

			if( i==j ) return;

			T tmp = this->m_pData[i];
			this->m_pData[i] = this->m_pData[j];
			this->m_pData[j] = tmp;
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T("[ ") << this->m_Length << _T(" ]:\n" );

			for( int i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}



	private:

		inline void ShiftElementsRight( int elm, int num=1 )
		{
			int numtomove = this->m_Length - elm - num;
			if( numtomove > 0 )
				MemMove( &this->m_pData[elm+num], &this->m_pData[elm], numtomove );
		}


		inline void ShiftElementsLeft( int elm, int num=1 )
		{
			int numtomove = this->m_Length - elm - num;
			if( numtomove > 0 )
				MemMove( &this->m_pData[elm], &this->m_pData[elm+num], numtomove );
		}

	};




/*

	template< typename T >
	class Array : public Memory<T>
	{
	public:

		// Default constructor
		Array(): Memory<T>() {}

		// Constructor
		Array( int len ) : Memory<T>(len) {}

		// Constructor
//		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
//		Array( Args const & ... args ) : Memory<T>( args ...) {}

		// Constructor with initializer list
		Array( std::initializer_list<T> ilist ) : Memory<T>( ilist ) {}

		// Constructor with external buffer
		Array( int len, T* pdata ): Memory<T>( len, pdata ) {}

		// Copy constructor
		Array( const Array& obj ) : Memory<T>( obj ) {}

		// Move constructor
		Array( Array&& obj ) : Memory<T>( obj ) {}

		// Copy Assignment opertor =
		inline Array& operator=( const Array& obj )
		{
			if( this != &obj )
			{
				Memory<T>::operator=( obj );
			}

			return *this;
		}

		// Move assignment opertor =
		inline Array& operator=( Array&& obj )
		{
			if( this != &obj )
			{
				Memory<T>::operator=( (Array&&)obj );
			}

			return *this;
		}


		inline int AddToFront()
		{
			return InsertBefore( 0 );
		}

		inline int AddToTail()
		{
			return InsertBefore( this->m_Length );
		}

		inline int InsertBefore( int elm )
		{
			if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			this->m_pData[elm] = T();
			return elm;
		}

		inline int InsertAfter( int elm )
		{
			return InsertBefore( elm + 1 );
		}

		
		inline int AddToFront( const T& src )
		{
			return InsertBefore( 0, src );
		}

		inline int AddToFront( T&& src )
		{
			return InsertBefore( 0, src );
		}

		inline int AddToTail( const T& src )
		{
			return InsertBefore( this->m_Length, src );
		}

		inline int AddToTail( T&& src )
		{
			return InsertBefore( this->m_Length, src );
		}


		inline int InsertBefore( int elm, const T& src )
		{
			if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			this->m_pData[elm] = src;
			return elm;
		}


		inline int InsertBefore( int elm, T&& src )
		{
			if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			this->m_pData[elm] = src;
			return elm;
		}



		inline int InsertAfter( int elm, const T& src )
		{
			return InsertBefore( elm+1, src );
		}

		inline int InsertAfter( int elm, T&& src )
		{
			return InsertBefore( elm+1, src );
		}


		inline void FastRemove( int elm )// 削除対象の要素を配列最後尾要素で上書きする & メモリ確保サイズ自体は変更せずlenghデクリメントする
		{
			assert( elm>=0 && elm<this->m_Length );

			this->m_pData[elm].~T();
			if( this->m_Length > 0 )
			{
				if( elm != this->m_Length - 1 )
					this->m_pData[ elm ] = this->m_pData[ this->m_Length-1 ];
					//memcpy( &this->m_pData[elm], &this->m_pData[this->m_Length-1], this->m_ElementSize );

				--this->m_Length;
			}

			if( this->m_Length==0 )
				this->Release();
		}

		//inline void FastRemove( const T& item )// 削除対象の要素を配列最後尾要素で上書きする & メモリ確保サイズ自体は変更せずlenghデクリメントする
		//{
		//	auto index = Find( this, item );

		//	if( index==-1 )
		//		return;

		//	FastRemove( index );
		//}

		
		inline void Remove( int elm )
		{
			assert( elm>=0 && elm<this->m_Length );

			this->m_pData[elm].~T();
			ShiftElementsLeft( elm );
			--this->m_Length;

			if( this->m_Length==0 )
				this->Release();
		}

		//inline void Remove( const T& item )
		//{
		//	auto index = Find( this, item );

		//	if( index==-1 )
		//		return;

		//	Remove( index );
		//}


		inline void Swap( int i, int j )
		{
			assert( i>=0 && i<this->m_Length && j>=0 && j<this->m_Length );

			if( i==j ) return;

			T tmp = this->m_pData[i];
			this->m_pData[i] = this->m_pData[j];
			this->m_pData[j] = tmp;
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T("[ ") << this->m_Length << _T(" ]:\n" );

			for( int i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}



	private:

		inline void ShiftElementsRight( int elm, int num=1 )
		{
			int numtomove = this->m_Length - elm - num;
			if( numtomove > 0 )
				MemMove( &this->m_pData[elm+num], &this->m_pData[elm], numtomove );
				//memmove( &this->m_pData[elm+num], &this->m_pData[elm], numtomove * this->c_ElementSize );
		}


		inline void ShiftElementsLeft( int elm, int num=1 )
		{
			int numtomove = this->m_Length - elm - num;
			if( numtomove > 0 )
				MemMove( &this->m_pData[elm], &this->m_pData[elm+num], numtomove );
				//memmove( &this->m_pData[elm], &this->m_pData[elm+num], numtomove * this->c_ElementSize );
		}

	};

*/




	//######################################################################//
	//																		//
	//							Helper functions							//
	//																		//
	//######################################################################//

	template < typename T >
	inline void Shuffle( Array<T>& arr )
	{
		for( int i=0; i<arr.Length(); ++i )
		{
			int j = int( genrand_real2() * arr.Length() );
			T temp	= arr[i];
			arr[i]	= arr[j];
			arr[j]	= temp;
		}// end of i loop

	}



}// end of namespace


#endif /* ARRAY_H */