#ifndef	MEMORY_ALLOCATOR_H
#define	MEMORY_ALLOCATOR_H

#include	<assert.h>


#include	"BoundaryTagBlock.h"


namespace OreOreLib
{

	class MemoryAllocator
	{
	public:
		
		MemoryAllocator();
		MemoryAllocator( uint32 allocSize );
		~MemoryAllocator();

		void Init( uint32 allocSize );
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

		void* Allocate( size_t size );
		void Free( uint8* data );


		void* AlignedAllocate( size_t size, size_t alignment=ByteSize::DefaultAlignment );

		uint32 Compact();
	
		void Info();



	private:

		// memory pool
		/*uint64*/uint32		m_AllocSize;
		/*uint64*/uint32		m_DataSize;
		uint8*				m_pData;
		
		// free list data 
		uint32				m_FreeListLength;
		BoundaryTagBlock	m_FreeList;


		

		inline BoundaryTagBlock* GetFreeBoundaryTag( uint32 requiredSize ) const
		{
			BoundaryTagBlock *pBB = m_FreeList.Next();
			while( true )
			{
				if( pBB->DataSize() >= requiredSize )
					return pBB;

				if( !pBB->HasNext() )
					break;

				pBB = pBB->Next();
			}

			return nullptr;
		}


		inline void AddtoFreeList( BoundaryTagBlock* pBB )
		{			
			m_FreeList.Register( pBB );
		}


		inline void RemovefromFreeList( BoundaryTagBlock* pBB )
		{
			assert( m_FreeList.HasNext() );
			pBB->Remove();
		}


		inline void ClearFreeList()
		{
			while( m_FreeList.HasNext() )
				m_FreeList.RemoveNext();
		}


	};



}// end of namespace


#endif // !MEMORY_ALLOCATOR_H
