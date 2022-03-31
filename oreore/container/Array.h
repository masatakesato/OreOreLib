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


	template< typename T, typename InexType >
	class ArrayBase< T, detail::DynamicSize, InexType > : public Memory<T, InexType>
	{
	public:

		// Default constructor
		ArrayBase(): Memory<T, InexType>() {}

		// Constructor
		ArrayBase( InexType len ) : Memory<T, InexType>(len) {}

		// Constructor
//		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
//		ArrayBase( Args const & ... args ) : Memory<T, InexType>( args ...) {}

		// Constructor with initializer list
		ArrayBase( std::initializer_list<T> ilist ) : Memory<T, InexType>( ilist ) {}

		// Constructor with default value
		ArrayBase( InexType len, const T& fill ): Memory<T, InexType>( len, fill ) {}

		// Constructor using Memory
		ArrayBase( const Memory<T, InexType>& obj ) : Memory<T, InexType>( obj ) {}

		// Constructor using iterator
		template < class Iter >
		ArrayBase( Iter first, Iter last ) : Memory<T, InexType>( first, last ) {}

		// Copy constructor
		ArrayBase( const ArrayBase& obj ) : Memory<T, InexType>( (const Memory<T, InexType>&)obj ) {}

		// Move constructor
		ArrayBase( ArrayBase&& obj ) : Memory<T, InexType>( (Memory<T, InexType>&&)obj ) {}


		// Copy Assignment opertor =
		inline ArrayBase& operator=( const ArrayBase& obj )
		{
			Memory<T, InexType>::operator=( obj );
			return *this;
		}


		inline ArrayBase& operator=( const Memory<T, InexType>& obj )
		{
			Memory<T, InexType>::operator=( obj );
			return *this;
		}


		// Move assignment opertor =
		inline ArrayBase& operator=( ArrayBase&& obj )
		{
			Memory<T, InexType>::operator=( (ArrayBase&&)obj );
			return *this;
		}


		inline InexType AddToFront()
		{
			return this->InsertBefore( 0 );
		}


		inline InexType AddToFront( const T& src )
		{
			return this->InsertBefore( 0, src );
		}


		inline InexType AddToFront( T&& src )
		{
			return this->InsertBefore( 0, (T&&)src );
		}


		inline InexType AddToTail()
		{
			return this->InsertBefore( this->m_Length );
		}


		inline InexType AddToTail( const T& src )
		{
			return this->InsertBefore( this->m_Length, src );
		}


		inline InexType AddToTail( T&& src )
		{
			return this->InsertBefore( this->m_Length, (T&&)src );
		}


		inline void FastRemove( InexType elm )// 削除対象の要素を配列最後尾要素で上書きする & メモリ確保サイズ自体は変更せずlenghデクリメントする
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

		
		inline void Remove( InexType elm )
		{
			ASSERT( elm < this->m_Length );

			if( this->m_Length > 1 )
				ShiftElementsLeft( elm + 1 );
			else
				this->m_pData[elm].~T();

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


		inline void Swap( InexType i, InexType j )
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

			for( InexType i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}



	private:

		inline void ShiftElementsRight( InexType elm, InexType num=1 )
		{
			if( this->m_Length <= ( elm + num ) || num == 0 )
				return;

			T* pDst = this->m_pData + this->m_Length - 1;
			T* pSrc = pDst - num;

			while( pSrc >= this->m_pData + elm )
			{
				pDst->~T();// destruct dst data first 
				new ( pDst ) T( (T&&)( *pSrc ) );// then move src data

				--pDst;
				--pSrc;
			}

			// destruct empty elements
			for( InexType i=0; i<num; ++i )
				(this->m_pData + elm + i)->~T();
		}


		inline void ShiftElementsLeft( InexType elm, InexType num=1 )
		{
			if( elm < num || num == 0 )
				return;

			T* pSrc = this->m_pData + Max( num, elm );
			T* pDst = pSrc - num;

			while( pSrc < this->end() )
			{
				pDst->~T();// destruct dst data first 
				new ( pDst ) T( (T&&)( *pSrc ) );// then move src data

				++pDst;
				++pSrc;
			}

			// destruct empty elements
			while( pDst != this->end() )
				(pDst++)->~T();
		}


	};




	//######################################################################//
	//																		//
	//							Helper functions							//
	//																		//
	//######################################################################//

	template < typename T, typename InexType >
	inline void Shuffle( ArrayImpl<T, InexType>& arr )
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