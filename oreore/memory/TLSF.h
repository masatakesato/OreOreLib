#ifndef TLSF_MEMORY_ALLOCATOR_H
#define	TLSF_MEMORY_ALLOCATOR_H

#include	<assert.h>

#include	"../common/BitOperations.h"

#include	"BoundaryTagBlock.h"



namespace OreOreLib
{

	class TLSF
	{
	public:
		
		TLSF();
		TLSF( uint32 allocSize/*, uint32 splitFactor*/ );
		~TLSF();

		void Init( uint32 allocSize/*, uint32 splitFactor*/ );
		void Release();
		void Clear();
		void Resize( uint32 allocSize );


		//template< typename T >
		//T* Allocate( uint32 size=sizeof( T ) )
		//{
		//	return (T*)Allocate( size );
		//}


		//template< typename T >
		//void Free( T* data )
		//{
		//	data->~T();// Call destructor
		//	Free( (uint8*)data );
		//}

		uint8* Allocate( uint32 size );
		void Free( uint8* data );

		uint32 Compact();
		void Info();


	private:
		
		// parameters for 2nd level splitting 
		const uint32		m_SplitFactor;
		uint32				m_NumSLI;

		// memory pool
		/*uint64*/uint32	m_AllocSize;
		/*uint64*/uint32	m_DataSize;
		uint8*				m_pData;
		
		// free list data 
		uint32				m_FreeListLength;
		BoundaryTagBlock	*m_FreeList;
		bool				*m_FreeSlots;

		uint32				m_FLIFreeSlots;

		uint32				m_SLIFreeSlotsLength;
		uint8/*uint16*//*uint32*/		*m_SLIFreeSlots;
		// uint8: SLI MUST 8 division at maximum = pow(2, N) MUST BELOW 8


		inline int SearchFreeFLI( uint32 fli_base ) const
		{
			// sizeを格納可能なfliをフリーリストから探す.m_FLIFreeSlotsのビット演算で算出
			//uint32 fli_mask = 0xffffffff << fli_base;
			//uint32 fli_freebits = m_FLIFreeSlots & fli_mask;
			uint32 fli_freebits = m_FLIFreeSlots & ( 0xffffffff << fli_base );

			return fli_freebits==0 ? -1 : GetLSB( fli_freebits );//GetLSB( fli_freebits );
		}


		inline int SearchFreeSLI( uint32 fli, uint32 sli_base ) const
		{
			//uint8& rb = m_SLIFreeSlots[fli];
			//uint32 sli_mask = 0xffffffff << sli_base;
			//uint32 sli_freebits = rb & sli_mask;
			uint32 sli_freebits = m_SLIFreeSlots[fli] & ( 0xffffffff << sli_base );

			return sli_freebits==0 ? -1 : GetLSB( sli_freebits );//GetLSB( sli_freebits );
		}

		
		// Get Second Level Index( 32 bit version ). N: 分割数 = 2^N
		inline int GetSLI( const unsigned __int32& val, int fli ) const
		{
			//=================== Step by Step Calculation ===================//
			//// 最上位ビット未満のビット列だけを取り出すマスクを計算する.
			//unsigned __int32 mask = ~( 0xffffffff << fli );

			//// valの最上位ビットをゼロにした値を取得する
			//unsigned __int32 val1 = val & mask;

			//// val1のSLI該当ビット列だけ抽出するための右シフト量を計算する. FLIのアドレス範囲を2^N個に分割する場合、val1の上位Nビットだけを使う
			//const unsigned right_shit = fli - m_SplitFactor;

			//return val1 >> right_shit;

			//=================== All in one expression ======================//
			return ( val & ~( 0xffffffff << fli ) ) >> ( fli - m_SplitFactor );
		}


		// Get Second Level Index( 64 bit version ).
		inline int GetSLI( const unsigned __int64& val, int fli ) const
		{
			return ( val & ~( 0xffffffffffffffff << fli ) ) >> ( fli - m_SplitFactor );
		}


		inline uint32 GetFreeListIndex( uint32 fli, uint32 sli ) const
		{
			//return fli * (uint32)pow( 2, m_SplitFactor ) + sli;
			return fli * m_NumSLI + sli;
		}


		inline void AddtoFreeList( BoundaryTagBlock* pBB )
		{
			uint32 fli = GetMSB( pBB->TotalSize() );
			uint32 sli = GetSLI( pBB->TotalSize(), fli );
			uint32 freeListID = GetFreeListIndex( fli, sli );
			
			assert( !m_FreeSlots[freeListID] );
			assert( !( m_SLIFreeSlots[fli] & ( 1 << sli ) ) );
			assert( !( m_FLIFreeSlots & ( 1 << fli ) ) );

			// Register BoundaryTagBlock to free list
			m_FreeList[freeListID].Register( pBB );

			// Set flags
			m_FreeSlots[freeListID] = true;
			m_FLIFreeSlots |= ( 1 << fli );
			m_SLIFreeSlots[fli] |= ( 1 << sli );
		}


		inline void RemovefromFreeList( BoundaryTagBlock* pBB )
		{
			uint32 fli = GetMSB( pBB->TotalSize() );
			uint32 sli = GetSLI( pBB->TotalSize(), fli );
			uint32 freeListID = GetFreeListIndex( fli, sli );

			assert( m_FreeList[freeListID].HasNext() );
			assert( m_FreeSlots[freeListID] );
			assert( m_SLIFreeSlots[fli] & ( 1 << sli ) );
			assert( m_FLIFreeSlots & ( 1 << fli ) );

			pBB->Remove();
			//pBB->SetFree( false );

			if( m_FreeList[freeListID].HasNext() == false )
			{
				// Set Freeslot[ (fli, sli)] to false
				m_FreeSlots[freeListID]	= false;
				// Set SLIFreeSlot to false
				m_SLIFreeSlots[fli] &= ~( 1 << sli );
				// Set FLIFleeSlot[fli] to false only if all sli slots are empty.
				if( m_SLIFreeSlots[fli] == 0 )
					m_FLIFreeSlots &= ~( 1 << fli );
			}// end of if

	
		}


		inline void ClearFreeList()
		{
			for( uint32 i=0; i<m_FreeListLength; ++i )
				while( m_FreeList[i].HasNext() ) m_FreeList[i].RemoveNext();//m_FreeList[i].next->Remove();

			//memset( m_FreeSlots, 0, sizeof( bool ) * freeListLength );
			std::fill_n( m_FreeSlots, m_FreeListLength, false );

			m_FLIFreeSlots	= 0;
			//memset( m_SLIFreeSlots, 0, sizeof( uint8 ) * m_SLIFreeSlotsLength );
			std::fill_n( m_SLIFreeSlots, m_SLIFreeSlotsLength, 0 );
		}


	};



}// end of namespace


#endif // !TLSF_MEMORY_ALLOCATOR_H
