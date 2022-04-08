#ifndef ARRAY_H
#define ARRAY_H

#include	<math.h>
#include	<limits>

#include	"../common/TString.h"
#include	"../mathlib/Random.h"

#include	"ArrayBase.h"




namespace OreOreLib
{

	//######################################################################//
	//																		//
	//						Array class implementation						//
	//																		//
	//######################################################################//


	template< typename T, typename IndexType >
	class ArrayBase< T, detail::DynamicSize, IndexType > : public MemoryBase<T, IndexType>
	{
	public:

		// Default constructor
		ArrayBase(): MemoryBase<T, IndexType>() {}

		// Constructor
		ArrayBase( IndexType len ) : MemoryBase<T, IndexType>(len) {}

		// Constructor
//		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
//		ArrayBase( Args const & ... args ) : MemoryBase<T, IndexType>( args ...) {}

		// Constructor with initializer list
		ArrayBase( std::initializer_list<T> ilist ) : MemoryBase<T, IndexType>( ilist ) {}

		// Constructor with default value
		ArrayBase( IndexType len, const T& fill ): MemoryBase<T, IndexType>( len, fill ) {}

		// Constructor using MemoryBase
		ArrayBase( const MemoryBase<T, IndexType>& obj ) : MemoryBase<T, IndexType>( obj ) {}

		// Constructor using iterator
		template < class Iter >
		ArrayBase( Iter first, Iter last ) : MemoryBase<T, IndexType>( first, last ) {}

		// Copy constructor
		ArrayBase( const ArrayBase& obj ) : MemoryBase<T, IndexType>( (const MemoryBase<T, IndexType>&)obj ) {}

		// Move constructor
		ArrayBase( ArrayBase&& obj ) : MemoryBase<T, IndexType>( (MemoryBase<T, IndexType>&&)obj ) {}


		// Copy Assignment opertor =
		inline ArrayBase& operator=( const ArrayBase& obj )
		{
			MemoryBase<T, IndexType>::operator=( obj );
			return *this;
		}


		inline ArrayBase& operator=( const MemoryBase<T, IndexType>& obj )
		{
			MemoryBase<T, IndexType>::operator=( obj );
			return *this;
		}


		// Move assignment opertor =
		inline ArrayBase& operator=( ArrayBase&& obj )
		{
			MemoryBase<T, IndexType>::operator=( (ArrayBase&&)obj );
			return *this;
		}





		// TODO: 追加されたメモリ領域は初期化する????
		inline bool Resize( IndexType newlen )
		{
			// Reallocate memory
			auto oldlen = this->m_Length;

			if( !this->Reallocate( newlen ) )
				return false;

			// Initialize allocated elements using placement new default constructor
			for( auto i=oldlen; i<newlen; ++i )	new ( this->m_pData + i ) T();

			return true;
		}


		inline bool Resize( IndexType newlen, const T& fill )
		{
			// Reallocate memory
			auto oldlen = this->m_Length;

			if( !this->Reallocate( newlen ) )
				return false;

			// Initialize allocated elements using placement new copy constructor
			for( auto i=oldlen; i<newlen; ++i )	new ( this->m_pData + i ) T( fill );

			return true;
		}


		inline bool Extend( IndexType numelms )
		{
			if( numelms==0 || numelms==~0u )	return false;
			return Resize( this->m_Length + numelms );
		}


		inline bool Extend( IndexType numelms, const T& fill )
		{
			if( numelms==0 || numelms==~0u )	return false;
			return Resize( this->m_Length + numelms, fill );
		}


		inline bool Shrink( IndexType numelms )
		{
			if( this->m_Length > numelms )
				return this->Reallocate( this->m_Length - numelms );

			return false;
		}








		inline IndexType AddToFront()
		{
			return this->InsertBefore( 0 );
		}


		inline IndexType AddToFront( const T& src )
		{
			return this->InsertBefore( 0, src );
		}


		inline IndexType AddToFront( T&& src )
		{
			return this->InsertBefore( 0, (T&&)src );
		}


		inline IndexType AddToTail()
		{
			return this->InsertBefore( this->m_Length );
		}


		inline IndexType AddToTail( const T& src )
		{
			return this->InsertBefore( this->m_Length, src );
		}


		inline IndexType AddToTail( T&& src )
		{
			return this->InsertBefore( this->m_Length, (T&&)src );
		}


		inline void FastRemove( IndexType elm )// 削除対象の要素を配列最後尾要素で上書きする & メモリ確保サイズ自体は変更せずlenghデクリメントする
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

		
		inline void Remove( IndexType elm )
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


		inline void Swap( IndexType i, IndexType j )
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

			for( IndexType i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}



	private:

		inline void ShiftElementsRight( IndexType elm, IndexType num=1 )
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
			for( IndexType i=0; i<num; ++i )
				(this->m_pData + elm + i)->~T();
		}


		inline void ShiftElementsLeft( IndexType elm, IndexType num=1 )
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

	template < typename T, typename IndexType >
	inline void Shuffle( ArrayImpl<T, IndexType>& arr )
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