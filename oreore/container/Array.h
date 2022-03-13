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
		using SizeType = typename Memory<T>::SizeType;

	public:

		// Default constructor
		ArrayBase(): Memory<T>() {}

		// Constructor
		ArrayBase( SizeType len ) : Memory<T>(len) {}

		// Constructor
//		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
//		ArrayBase( Args const & ... args ) : Memory<T>( args ...) {}

		// Constructor with initializer list
		ArrayBase( std::initializer_list<T> ilist ) : Memory<T>( ilist ) {}

		// Constructor with default value
		ArrayBase( SizeType len, const T& fill ): Memory<T>( len, fill ) {}

		// Constructor using Memory
		ArrayBase( const Memory<T>& obj ) : Memory<T>( obj ) {}

		// Constructor using iterator
		template < class Iter >
		ArrayBase( Iter first, Iter last ) : Memory<T>( first, last ) {}

		// Copy constructor
		ArrayBase( const ArrayBase& obj ) : Memory<T>( (const Memory<T>&)obj ) {}

		// Move constructor
		ArrayBase( ArrayBase&& obj ) : Memory<T>( (Memory<T>&&)obj ) {}


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


		inline SizeType AddToFront()
		{
			return InsertBefore( 0 );
		}

		inline SizeType AddToTail()
		{
			return InsertBefore( this->m_Length );
		}

		inline SizeType InsertBefore( SizeType elm )
		{
			if( this->Resize( this->m_Length + 1 )==false )//if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			return elm;
		}

		inline SizeType InsertAfter( SizeType elm )
		{
			return InsertBefore( elm + 1 );
		}

		
		inline SizeType AddToFront( const T& src )
		{
			return InsertBefore( 0, src );
		}

		inline SizeType AddToFront( T&& src )
		{
			return InsertBefore( 0, (T&&)src );
		}

		inline SizeType AddToTail( const T& src )
		{
			return InsertBefore( this->m_Length, src );
		}

		inline SizeType AddToTail( T&& src )
		{
			return InsertBefore( this->m_Length, (T&&)src );
		}


		inline SizeType InsertBefore( SizeType elm, const T& src )
		{
			if( this->Resize( this->m_Length + 1 )==false )//if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			T* val = new ( &this->m_pData[elm] ) T(src);//this->m_pData[elm] = src;
			return elm;
		}


		inline SizeType InsertBefore( SizeType elm, T&& src )
		{
			if( this->Resize( this->m_Length + 1)==false )//if( this->Extend( 1 )==false )
				return -1;
			ShiftElementsRight( elm );
			T* val = new ( &this->m_pData[elm] ) T( (T&&)src );//this->m_pData[elm] = src;
			return elm;
		}



		inline SizeType InsertAfter( SizeType elm, const T& src )
		{
			return InsertBefore( elm+1, src );
		}

		inline SizeType InsertAfter( SizeType elm, T&& src )
		{
			return InsertBefore( elm+1, (T&&)src );
		}


		inline void FastRemove( SizeType elm )// 削除対象の要素を配列最後尾要素で上書きする & メモリ確保サイズ自体は変更せずlenghデクリメントする
		{
			ASSERT( elm<this->m_Length );

			this->m_pData[elm].~T();
			if( this->m_Length > 0 )
			{
				if( elm != this->m_Length - 1 )
					this->m_pData[ elm ] = this->m_pData[ this->m_Length-1 ];
					//memcpy( &this->m_pData[elm], &this->m_pData[this->m_Length-1], this->m_ElementSize );

				--this->m_Length;
				this->m_AllocSize -= this->c_ElementSize;
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

		
		inline void Remove( SizeType elm )
		{
			ASSERT( elm<this->m_Length );

			this->m_pData[elm].~T();
			ShiftElementsLeft( elm );

			--this->m_Length;
			this->m_AllocSize -= this->c_ElementSize;

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


		inline void Swap( SizeType i, SizeType j )
		{
			ASSERT( i<this->m_Length && j<this->m_Length );

			if( i==j ) return;

			T tmp = this->m_pData[i];
			this->m_pData[i] = this->m_pData[j];
			this->m_pData[j] = tmp;
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T("[ ") << this->m_Length << _T(" ]:\n" );

			for( SizeType i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}



	private:

		inline void ShiftElementsRight( SizeType elm, SizeType num=1 )
		{
			if( this->m_Length <= ( elm + num ) || num == 0 )
				return;

			//if( this->m_Length < elm + num )
			//SizeType numtomove = this->m_Length - ( elm + num );
			//if( numtomove > 0 )
//				MemMove( &this->m_pData[elm+num], &this->m_pData[elm], /*numtomove*/this->m_Length - ( elm + num ) );


			T* pDst = this->m_pData + this->m_Length - 1;
			T* pSrc = pDst - num;

			while( pSrc >= this->m_pData + elm )
			{
				pDst->~T();
				new ( pDst ) T( (T&&)( *pSrc ) );

				--pDst;
				--pSrc;
			}

			// destruct empty elements
			for( SizeType i=0; i<num; ++i )
				(this->m_pData + i)->~T();
		}


TODO: MemMove使えない. 要素毎にデストラクタ呼び出しながら移動する

		inline void ShiftElementsLeft( SizeType elm, SizeType num=1 )
		{
			if( this->m_Length <= ( elm + num ) || num == 0 )
				return;

			//SizeType numtomove = this->m_Length - ( elm + num );
			//if( numtomove > 0 )
				MemMove( &this->m_pData[elm], &this->m_pData[elm+num], /*numtomove*/this->m_Length - ( elm + num ) );
		}

	};




	//######################################################################//
	//																		//
	//							Helper functions							//
	//																		//
	//######################################################################//

	template < typename T >
	inline void Shuffle( Array<T>& arr )
	{
		for( sizeType i=0; i<arr.Length(); ++i )
		{
			sizeType j = sizeType( genrand_real2() * arr.Length() );
			T temp	= arr[i];
			arr[i]	= arr[j];
			arr[j]	= temp;
		}// end of i loop

	}



}// end of namespace


#endif /* ARRAY_H */