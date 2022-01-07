#include	"PoolAllocator.h"


#include	"../common/BitOperations.h"
#include	"../os/OSAllocator.h"

#ifdef _WIN64
#include	<windows.h>
#endif



namespace OreOreLib
{


	//######################################################################################//
	//																						//
	//									Page Implementation									//
	//																						//
	//######################################################################################//


	void Page::ConnectAfter( Page* pnode )
	{
		if( !IsAlone() )	return;
			
		// update connection of this node
		prev = pnode;
		next = pnode->next;

		// update connection of pnode
		pnode->next = this;

		// update connection of next node
		if( next )	next->prev = this;
	}


	void Page::ConnectBefore( Page* pnode )
	{
		if( !IsAlone() )	return;
			
		// update connection of this node
		prev = pnode->prev;
		next = pnode;

		// update connection of pnode
		pnode->prev = this;

		// update connection of next node
		if( prev )	prev->next = this;
	}


	void Page::Disconnect()
	{
		if( prev )	prev->next = next;
		if( next )	next->prev = prev;
		next = nullptr;
		prev = nullptr;
	}


	void Page::DisconnectPrev()
	{
		if( prev )	prev->next = nullptr;
		prev = nullptr;
	}


	void Page::DisconnectNext()
	{
		if( prev )	prev->next = nullptr;
		prev = nullptr;
	}


	bool Page::IsAlone() const
	{
		return next && prev ? false : true;
	}




	//######################################################################################//
	//																						//
	//									PageTag Implementation								//
	//																						//
	//######################################################################################//


	void PageTag::Init( /*uint16 ptagsize,*/ uint16 numfreeblocks, size_t bitflagsize )
	{
		//PageTagSize		= ptagsize;
		NumFreeBlocks	= numfreeblocks;

		//========= Initialize Free block bit flag ======//
		// zero clear
		memset( FreeBits, 0x00, bitflagsize );

		// set 1 to free block bits
		for( int i=0; i<NumFreeBlocks; ++i )
			SetBit( i, FreeBits );
	}




	//######################################################################################//
	//																						//
	//								RegionTag Implementation								//
	//																						//
	//######################################################################################//


	void RegionTag::Init( size_t rtagsize, size_t regionsize, size_t pagesize, PoolAllocator* pallocator )
	{
		next			= nullptr;
		RegionTagSize	= rtagsize;
		RegionSize		= regionsize;
		PageSize		= pagesize;
		pAllocator		= pallocator;

		//NumActivePages	= ( regionsize - rtagsize ) / pagesize;
		//NumFreeOSPages	= pRTag->NumActivePages;
	}



	void RegionTag::ConnectAfter( RegionTag* ptag )
	{
		next = ptag->next;
		ptag->next = this;
	}



	void RegionTag::DisconnectNext()
	{
		if( !next )	return;

		RegionTag* newnext = next->next;
		next->next = nullptr;
		next = newnext;
	}






	//######################################################################################//
	//																						//
	//							PoolAllocator Implementation								//
	//																						//
	//######################################################################################//


	const size_t REGION_RTAG_RATIO = 8;// region_size / region_tag_size


	// Default constructor
	PoolAllocator::PoolAllocator()
		: m_BlockSize( 0 )
		, m_CommitBatchSize( 0 )

		, m_AllocSize( 0 )
		//, m_PageDataSize( 0 )
		, m_BitFlagSize( 0 )
		, m_PageTagSize( 0 )
		, m_NumActiveBlocks( 0 )
		, m_PoolSize( 0 )
		
		, m_OSAllocSize( 0 )
		, m_OSAllocationGranularity( 0 )
		, m_FirstPageSize( 0 )
		, m_FirstPageFreeBlocks( 0 )
		, m_RegionTagOffset( 0 )

		, m_Nil{ &m_Nil, &m_Nil, 0x00 }
		, m_CleanFront( &m_Nil )
		, m_DirtyFront( &m_Nil )
		, m_UsedupFront( &m_Nil )

		, m_FeedNil{ nullptr, 0, 0, 0, nullptr }
		, m_pFeedFront( nullptr )
	{
	}



	// Constructor
	PoolAllocator::PoolAllocator( uint32 allocSize, uint32 blockSize, uint32 commitBatchSize )
		: m_BlockSize( blockSize )
		, m_CommitBatchSize( commitBatchSize )

		, m_Nil{ &m_Nil, &m_Nil, 0x00 }
		, m_CleanFront( &m_Nil )
		, m_DirtyFront( &m_Nil )
		, m_UsedupFront( &m_Nil )

		, m_FeedNil{ nullptr, 0, 0, 0, nullptr }
		, m_pFeedFront( nullptr )
	{
		ASSERT( allocSize > blockSize && blockSize > 0 );

		InitPageBlockParams( allocSize, blockSize );
		InitFeedParams( allocSize, blockSize );

		//BatchAllocatePages( m_CommitBatchSize );
	}



	// Copy constructor
	PoolAllocator::PoolAllocator( const PoolAllocator& obj )
		: m_BlockSize( obj.m_BlockSize )
		, m_CommitBatchSize( obj.m_CommitBatchSize )

		, m_AllocSize( obj.m_AllocSize )
		//, m_PageDataSize( obj.m_PageDataSize )
		, m_BitFlagSize( obj.m_BitFlagSize )
		, m_PageTagSize( obj.m_PageTagSize )
		, m_NumActiveBlocks( obj.m_NumActiveBlocks )
		, m_PoolSize( obj.m_PoolSize )

		, m_OSAllocSize( obj.m_OSAllocSize )
		, m_OSAllocationGranularity( obj.m_OSAllocationGranularity )
		, m_FirstPageSize( obj.m_FirstPageSize )
		, m_FirstPageFreeBlocks( obj.m_FirstPageFreeBlocks )
		, m_RegionTagOffset( obj.m_RegionTagOffset )

		, m_Nil{ &m_Nil, &m_Nil, 0x00 }
		, m_CleanFront( &m_Nil )
		, m_DirtyFront( &m_Nil )
		, m_UsedupFront( &m_Nil )

		, m_FeedNil{ nullptr, 0, 0, 0, nullptr }
		, m_pFeedFront( nullptr )
	{
		tcout << _T( "Copy constructor...\n" );


		//BatchAllocatePages( m_CommitBatchSize );
	}



	// Move constructor
	PoolAllocator::PoolAllocator( PoolAllocator&& obj )
		: m_BlockSize( obj.m_BlockSize )
		, m_CommitBatchSize( obj.m_CommitBatchSize )

		, m_AllocSize( obj.m_AllocSize )
		//, m_PageDataSize( obj.m_PageDataSize )
		, m_BitFlagSize( obj.m_BitFlagSize )
		, m_PageTagSize( obj.m_PageTagSize )
		, m_NumActiveBlocks( obj.m_NumActiveBlocks )
		, m_PoolSize( obj.m_PoolSize )

		, m_OSAllocSize( obj.m_OSAllocSize )
		, m_OSAllocationGranularity( obj.m_OSAllocationGranularity )
		, m_FirstPageSize( obj.m_FirstPageSize )
		, m_FirstPageFreeBlocks( obj.m_FirstPageFreeBlocks )
		, m_RegionTagOffset( obj.m_RegionTagOffset )

		, m_Nil{ &m_Nil, &m_Nil, 0x00 }
		, m_CleanFront( &m_Nil )
		, m_DirtyFront( &m_Nil )
		, m_UsedupFront( &m_Nil )

		, m_FeedNil{ nullptr, 0, 0, 0, nullptr }
		, m_pFeedFront( obj.m_pFeedFront )
	{
		tcout << _T( "Move constructor...\n" );

		// Move pages from obj to *this
		if( obj.m_Nil.next != &obj.m_Nil )
		{
			m_Nil.next = obj.m_Nil.next;
			m_Nil.next->prev = &m_Nil;
				
			m_Nil.prev = obj.m_Nil.prev;
			m_Nil.prev->next = &m_Nil;
		}

		if( !obj.IsEmpty( obj.m_CleanFront ) )
			m_CleanFront = obj.m_CleanFront;

		if( !obj.IsEmpty( obj.m_DirtyFront ) )
			m_DirtyFront = obj.m_DirtyFront;

		if( !obj.IsEmpty( obj.m_UsedupFront ) )
			m_UsedupFront = obj.m_UsedupFront;

		// Detach page references from obj
		obj.m_Nil.next = obj.m_Nil.prev = obj.m_CleanFront = obj.m_DirtyFront = obj.m_UsedupFront = &obj.m_Nil;


		// Move Feeds from obj to *this
		if( obj.m_FeedNil.next != &obj.m_FeedNil )
			m_FeedNil.next = obj.m_FeedNil.next;

		// Detach Feed references from obj
		obj.m_FeedNil.next = nullptr;
		obj.m_pFeedFront = nullptr;
	}



	// Destructor
	PoolAllocator::~PoolAllocator()
	{
		ClearPages();
	}
	


	// Copy assignment operator
	PoolAllocator& PoolAllocator::operator=( const PoolAllocator& obj )
	{
		tcout << _T( "Copy assignment operator...\n" );

		if( this != &obj )
		{
			// Free currently allocated memory
			ClearPages();

			// Copy data to *this
			m_BlockSize					= obj.m_BlockSize;
			m_CommitBatchSize			= obj.m_CommitBatchSize;
			m_AllocSize					= obj.m_AllocSize;
			//m_PageDataSize					= obj.m_PageDataSize;
			m_BitFlagSize				= obj.m_BitFlagSize;
			m_PageTagSize				= obj.m_PageTagSize;
			m_NumActiveBlocks			= obj.m_NumActiveBlocks;
			m_PoolSize					= obj.m_PoolSize;
			m_OSAllocSize				= obj.m_OSAllocSize;
			m_OSAllocationGranularity	= obj.m_OSAllocationGranularity;
			m_FirstPageSize				= obj.m_FirstPageSize;
			m_FirstPageFreeBlocks		= obj.m_FirstPageFreeBlocks;
			m_RegionTagOffset			= obj.m_RegionTagOffset;


			//BatchAllocatePages( m_CommitBatchSize );
		}

		return *this;
	}



	// Move assignment operator
	PoolAllocator& PoolAllocator::operator=( PoolAllocator&& obj )
	{
		tcout << _T( "Move assignment operator...\n" );

		if( this != &obj )
		{
			// Free currently allocated memory
			ClearPages();

			// Copy data to *this
			m_BlockSize					= obj.m_BlockSize;
			m_CommitBatchSize			= obj.m_CommitBatchSize;
			m_AllocSize					= obj.m_AllocSize;
			//m_PageDataSize					= obj.m_PageDataSize;
			m_BitFlagSize				= obj.m_BitFlagSize;
			m_PageTagSize				= obj.m_PageTagSize;
			m_NumActiveBlocks			= obj.m_NumActiveBlocks;
			m_PoolSize					= obj.m_PoolSize;
			m_OSAllocSize				= obj.m_OSAllocSize;
			m_OSAllocationGranularity	= obj.m_OSAllocationGranularity;
			m_FirstPageSize				= obj.m_FirstPageSize;
			m_FirstPageFreeBlocks		= obj.m_FirstPageFreeBlocks;
			m_RegionTagOffset			= obj.m_RegionTagOffset;
			m_pFeedFront				= obj.m_pFeedFront;


			// Move Pages from obj to *this
			if( obj.m_Nil.next != &obj.m_Nil )
			{
				m_Nil.next = obj.m_Nil.next;
				m_Nil.next->prev = &m_Nil;
				
				m_Nil.prev = obj.m_Nil.prev;
				m_Nil.prev->next = &m_Nil;
			}

			if( !obj.IsEmpty( obj.m_CleanFront ) )
				m_CleanFront = obj.m_CleanFront;

			if( !obj.IsEmpty( obj.m_DirtyFront ) )
				m_DirtyFront = obj.m_DirtyFront;

			if( !obj.IsEmpty( obj.m_UsedupFront ) )
				m_UsedupFront = obj.m_UsedupFront;

			// Detach Page references from obj
			obj.m_Nil.next = obj.m_Nil.prev = obj.m_CleanFront = obj.m_DirtyFront = obj.m_UsedupFront = &obj.m_Nil;


			// Move Feeds from obj to *this
			if( obj.m_FeedNil.next != &obj.m_FeedNil )
				m_FeedNil.next = obj.m_FeedNil.next;

			// Detach Feed references from obj
			obj.m_FeedNil.next = nullptr;
			obj.m_pFeedFront = nullptr;
		}

		return *this;
	}



	void PoolAllocator::Init( size_t allocSize, size_t blockSize, uint32 commitBatchSize )
	{
		ASSERT( allocSize > blockSize && blockSize > 0 );

		// Free currently allocated memory
		ClearPages();

		m_BlockSize			= blockSize ;
		m_CommitBatchSize	= commitBatchSize;
		
		InitPageBlockParams( allocSize, blockSize );
		InitFeedParams( allocSize, blockSize );

		//BatchAllocatePages( m_CommitBatchSize );
	}



	void* PoolAllocator::Allocate( size_t alignment )
	{
		ASSERT( m_AllocSize > 0 && m_BlockSize > 0 && alignment < m_BlockSize );

		//==================== メモリブロック取得前の準備 ====================//
		// DirtyListが空の場合
		if( IsEmpty( m_DirtyFront ) )
		{
			// Extend clean pages if empty
			if( IsEmpty( m_CleanFront ) )
			{
				tcout << _T( "m_CleanFront is empty. Allocating new Clean page....\n" );
				BatchAllocatePages( m_CommitBatchSize );

				//ASSERT( IsEmpty( m_CleanFront )==false );
				if( IsEmpty( m_CleanFront ) )
					return nullptr;
			}

			tcout << _T( "m_DirtyFront is empty. Acquiring a clean page from m_CleanFront...\n" );
			m_DirtyFront = m_CleanFront;
			m_CleanFront = m_CleanFront->next;
		}

		//====================== メモリブロックの確保 ======================//
		Page* page	= m_DirtyFront;
		void* ptr	= AllocateBlock( page );// メモリブロックを確保する(Pageの空きブロック数がデクリメントされる)

		//ASSERT( ptr !=nullptr );
		if( !ptr )
			return nullptr;

		if( GetPageTag( page )->NumFreeBlocks == 0 )// pageがUsedupになった場合
		{
			// Update m_DirtyFront
			if( m_DirtyFront->next != m_CleanFront )
				m_DirtyFront = m_DirtyFront->next;
			else
				m_DirtyFront = &m_Nil;

			// Update m_UsedupFront if necessary
			if( IsEmpty(m_UsedupFront) )
				m_UsedupFront = page;
		}

		return alignment==0 ? ptr : (void*)RoundUp( size_t(ptr), alignment );
	}

	//void* PoolAllocator::Allocate()
	//{
	//	ASSERT( m_AllocSize > 0 && m_BlockSize > 0 );

	//	//==================== メモリブロック取得前の準備 ====================//
	//	// DirtyListが空の場合
	//	if( IsEmpty( m_DirtyFront ) )
	//	{
	//		// Extend cleap pages if empty
	//		if( IsEmpty( m_CleanFront ) )
	//		{
	//			tcout << _T( "m_CleanFront is empty. Allocating new Clean page....\n" );
	//			BatchAllocatePages( m_CommitBatchSize );

	//			//ASSERT( IsEmpty( m_CleanFront )==false );
	//			if( IsEmpty( m_CleanFront ) )
	//				return nullptr;
	//		}

	//		tcout << _T( "m_DirtyFront is empty. Acquiring a clean page from m_CleanFront...\n" );
	//		m_DirtyFront = m_CleanFront;
	//		m_CleanFront = m_CleanFront->next;
	//	}

	//	//====================== メモリブロックの確保 ======================//
	//	Page* page	= m_DirtyFront;
	//	void* ptr	= AllocateBlock( page );// メモリブロックを確保する(Pageの空きブロック数がデクリメントされる)

	//	//ASSERT( ptr !=nullptr );
	//	if( !ptr )
	//		return nullptr;

	//	if( GetPageTag( page )->NumFreeBlocks == 0 )// pageがUsedupになった場合
	//	{
	//		// Update m_DirtyFront
	//		if( m_DirtyFront->next != m_CleanFront )
	//			m_DirtyFront = m_DirtyFront->next;
	//		else
	//			m_DirtyFront = &m_Nil;

	//		// Update m_UsedupFront if necessary
	//		if( IsEmpty(m_UsedupFront) )
	//			m_UsedupFront = page;
	//	}

	//	return ptr;
	//}



	bool PoolAllocator::Free( void*& ptr, Page* page )
	{
		ASSERT( m_AllocSize > 0 && m_BlockSize > 0 );

		if( !ptr )	return false;

		tcout << _T( "PoolAllocator::Free()..." ) << ptr << tendl;
		
		// Try to get Page from ptr if page is nullptr
		if( !page )
			page = GetPage( ptr );
		
		// return false if page not found. 
		if( !page )
			return false;

		// Clean: 0, Dirty: 1, Usedup: 2
		PageStates stateBefore = GetPageState( page );
		FreeBlock( page, ptr );
		ptr = nullptr;
		PageStates stateAfter = GetPageState( page );


		if( stateBefore==Usedup && stateAfter==Dirty )
		{
			tcout << _T( "  Usedup -> Dirty...\n" );

			Page* pPivot = IsEmpty( m_DirtyFront ) ? m_CleanFront : m_DirtyFront;

			if( page->next != pPivot )
			{
				if( page == m_UsedupFront ) m_UsedupFront = page->next;// page is the first usedup element 
				page->Disconnect();
				page->ConnectBefore( pPivot );
			}
			else if( page == m_UsedupFront )// page is the only remaining usedup, and pPivot's neighbor.
			{
				m_UsedupFront = &m_Nil;
			}

			m_DirtyFront = page;
		}

		else if( stateBefore==Dirty && stateAfter==Clean )
		{
			tcout << _T( "  Dirty -> Clean...\n" );	

			if( page->next != m_CleanFront )
			{
				if( page == m_DirtyFront ) m_DirtyFront = page->next;
				page->Disconnect();
				page->ConnectBefore( m_CleanFront );
			}
			else if( page == m_DirtyFront )// page is the only remaining Dirty, and m_CleanFront's neighbor.
			{
				m_DirtyFront = &m_Nil;
			}

			m_CleanFront = page;
		}

		else if( stateBefore==Usedup && stateAfter==Clean )// Occurs only when Page contains single block.
		{
			tcout << _T( "  Usedup -> Clean...\n" );

			if( page->next != m_CleanFront )
			{
				if( page == m_UsedupFront )	m_UsedupFront = page->next;
				page->Disconnect();
				page->ConnectBefore( m_CleanFront );
			}
			else if( page == m_UsedupFront )// page is the only remaining Usedup, and m_CleanFront's neighbor.
			{
				m_UsedupFront = &m_Nil;
			}

			m_CleanFront	= page;
		}

		else if( stateBefore==Dirty && stateAfter==Dirty )// Do nothing.
		{
			tcout << _T( "  Dirty -> Dirty...\n" );
		}

		tcout << tendl;

		return true;
	}



	bool PoolAllocator::SafeFree( void*& ptr )
	{
		ASSERT( m_AllocSize > 0 && m_BlockSize > 0 );
		
		tcout << _T( "PoolAllocator::SafeFree()..." ) << ptr << tendl;

		// Find page from ptr address
		Page* page = SafeGetPage( ptr );

		if( !page )
		{
			tcout << _T( "  Aborting: Could not find ptr...\n" );
			return false;
		}

		// Clean: 0, Dirty: 1, Usedup: 2
		PageStates stateBefore = GetPageState( page );
		FreeBlock( page, ptr );
		ptr = nullptr;
		PageStates stateAfter = GetPageState( page );


		if( stateBefore==Usedup && stateAfter==Dirty )
		{
			tcout << _T( "  Usedup -> Dirty...\n" );

			Page* pPivot = IsEmpty( m_DirtyFront ) ? m_CleanFront : m_DirtyFront;

			if( page->next != pPivot )
			{
				if( page == m_UsedupFront ) m_UsedupFront = page->next;// page is the first usedup element 
				page->Disconnect();
				page->ConnectBefore( pPivot );
			}
			else if( page == m_UsedupFront )// page is the only remaining usedup, and pPivot's neighbor.
			{
				m_UsedupFront = &m_Nil;
			}

			m_DirtyFront = page;
		}

		else if( stateBefore==Dirty && stateAfter==Clean )
		{
			tcout << _T( "  Dirty -> Clean...\n" );	

			if( page->next != m_CleanFront )
			{
				if( page == m_DirtyFront ) m_DirtyFront = page->next;
				page->Disconnect();
				page->ConnectBefore( m_CleanFront );
			}
			else if( page == m_DirtyFront )// page is the only remaining Dirty, and m_CleanFront's neighbor.
			{
				m_DirtyFront = &m_Nil;
			}

			m_CleanFront = page;
		}

		else if( stateBefore==Usedup && stateAfter==Clean )// Occurs only when Page contains single block.
		{
			tcout << _T( "  Usedup -> Clean...\n" );

			if( page->next != m_CleanFront )
			{
				if( page == m_UsedupFront )	m_UsedupFront = page->next;
				page->Disconnect();
				page->ConnectBefore( m_CleanFront );
			}
			else if( page == m_UsedupFront )// page is the only remaining Usedup, and m_CleanFront's neighbor.
			{
				m_UsedupFront = &m_Nil;
			}

			m_CleanFront	= page;
		}

		else if( stateBefore==Dirty && stateAfter==Dirty )// Do nothing.
		{
			tcout << _T( "  Dirty -> Dirty...\n" );
		}

		tcout << tendl;

		return true;
	}



	void PoolAllocator::Display() const
	{
		tcout << _T( "//========== " ) << typeid( *this ).name() << _T( " ==========//\n" );

		tcout << _T( " Allocated Size: " ) << m_AllocSize << _T( "[bytes] ( linkedlist pointer: " << Page::HeaderSize << "[bytes], data: " << m_AllocSize - Page::HeaderSize/*m_PageDataSize*/ << "[bytes] )\n" );
		tcout << _T( " Active size:    " ) << ( m_PoolSize + m_PageTagSize ) << _T( "[bytes] ( pool: " ) << m_PoolSize << _T( "[bytes], tag: " ) << m_PageTagSize << _T( "[bytes] )\n" );
		tcout << _T( " Usage:          " ) << float32( m_PoolSize + m_PageTagSize ) / (float32)m_AllocSize * 100 << _T( "[%] ( " ) << m_AllocSize-m_PoolSize-m_PageTagSize << _T( " [bytes] wasted. ) \n" );

		tcout << _T( " Block Size:     " ) << m_BlockSize << _T( "[bytes]\n" );
		tcout << _T( " Active Blocks:  " ) << m_NumActiveBlocks << tendl;

		tcout << _T( " OSAllocSize:     " ) << m_OSAllocSize << _T( "[bytes]\n" );
		tcout << _T( " OSAllocationGranularity:  " ) << m_OSAllocationGranularity << _T( "[bytes]\n" );
		tcout << _T( " Commit Batch Size:  " ) << m_CommitBatchSize << tendl;

		tcout << tendl;

		tcout << _T( " Feeds:\n" );
		for( RegionTag* rtag=m_FeedNil.next; rtag!=nullptr; rtag=rtag->next )
		{
			OSAllocator::DisplayMemoryInfo( rtag );
		}



		tcout << _T( " UsedUp Pages...\n" );
		for( Page* page=m_UsedupFront; page!=m_DirtyFront && page!=m_CleanFront && page!=&m_Nil; page=page->next )
		{
			tcout << _T( "  [" ) << (unsigned*)page << _T( "]: " ) << GetPageTag( page )->NumFreeBlocks << _T( "/" ) << m_NumActiveBlocks << tendl;
		}

		tcout << _T( " Dirty Pages...\n" );
		for( Page* page=m_DirtyFront; page!=m_CleanFront && page!=&m_Nil; page=page->next )
		{
			tcout << _T( "  [" ) << (unsigned*)page << _T( "]: " ) << GetPageTag( page )->NumFreeBlocks << _T( "/" ) << m_NumActiveBlocks << tendl;
		}

		tcout << _T( " Clean Pages...\n" );
		for( Page* page=m_CleanFront; page!=&m_Nil; page=page->next )
		{
			tcout << _T( "  [" ) << (unsigned*)page << _T( "]: " ) << GetPageTag( page )->NumFreeBlocks << _T( "/" ) << m_NumActiveBlocks << tendl;
		}

		tcout << tendl;
	}



	uint8* PoolAllocator::GetPoolBase( const void* ptr ) const
	{
		return (uint8*)GetPage( ptr ) + Page::HeaderSize + m_PageTagSize;
	}




	void PoolAllocator::BatchAllocatePages( uint32 batchsize )
	{
		uint8* reserved = nullptr;
		Page* newPage = nullptr;

		for( uint32 i=0; i<batchsize; ++i )
		{
			// Reserve virtual address if empty
			if( !m_pFeedFront )
			{
				#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT
					// Reserve alignable virtual address space
					m_pFeedFront = OSAllocator::ReserveUncommited( m_OSAllocationGranularity + RegionTag::Alignment );
				#else
					m_pFeedFront = OSAllocator::ReserveUncommited( m_OSAllocationGranularity );
				#endif
				//tcout << _T( "Reserving new os memory[" ) << m_OSAllocationGranularity << _T( "]...\n" );
			}

			// Find memory space from reserved region
			#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT
				// アラインメントした先頭アドレスを使って領域を探す
				reserved = (uint8*)OSAllocator::FindRegion( (void*)RoundUp( (size_t)m_pFeedFront, RegionTag::Alignment ), OSAllocator::Reserved, m_OSAllocSize, m_OSAllocationGranularity );
			#else
				reserved = (uint8*)OSAllocator::FindRegion( m_pFeedFront, OSAllocator::Reserved, m_OSAllocSize, m_OSAllocationGranularity );
			#endif

			// Abort committing if reserved cannot be found from m_pFeed
			if( !reserved )
			{
				//tcerr << _T( "Failed to find region...\n" );
				m_pFeedFront = nullptr;
				break;
			}

			// Setup RegionTag if "reserved" is the start address m_pFeed.
			#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT
			if( reserved==(void*)RoundUp( (size_t)m_pFeedFront, RegionTag::Alignment ) )
			#else
			if( reserved==m_pFeedFront )
			#endif
			{
				OSAllocator::Commit( reserved, m_FirstPageSize/*commitSize*/ );

				// Initialize RegionTag
				RegionTag* pRTag = (RegionTag*)reserved;
				pRTag->Init( m_RegionTagOffset, m_OSAllocationGranularity, m_OSAllocSize, this );//((RegionTag*)reserved)->Init( m_RegionTagOffset, m_OSAllocationGranularity, m_OSAllocSize, this );
				pRTag->ConnectAfter( &m_FeedNil );

				#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT
					pRTag->AllocationBase = m_pFeedFront;// 仮想メモリ空間の先頭アドレスを保持する
				#endif

				newPage = (Page*)( reserved + m_RegionTagOffset );

				// Initialize PageTag
				PageTag* pPTag = GetPageTag( newPage );
				pPTag->Init( /*m_PageTagSize,*/ m_FirstPageFreeBlocks, m_BitFlagSize );
			}
			else
			{
				newPage = (Page*)OSAllocator::Commit( reserved, m_OSAllocSize );
				//tcout << "  Page: "<< (unsigned *)reserved << tendl;

				// Initialize PageTag
				PageTag* pPTag = GetPageTag( newPage );
				pPTag->Init( /*m_PageTagSize,*/ m_NumActiveBlocks, m_BitFlagSize );
			}


			// Deploy as "Clean" page.
			newPage->ConnectBefore( m_CleanFront );
			m_CleanFront = newPage;


			// Detach m_pFeed if usedup
			if( (size_t)reserved + m_OSAllocSize >= (size_t)m_pFeedFront + m_OSAllocationGranularity )
				m_pFeedFront	= nullptr;

		}// end of i loop

	}



	bool PoolAllocator::FreePage2( Page*& page )
	{
		if( !page )
			return false;

		page->Disconnect();
		OSAllocator::Decommit( page, m_OSAllocSize );

		return true;
	}



	void PoolAllocator::ClearPages()
	{
		m_Nil.next = m_Nil.prev = m_CleanFront = m_DirtyFront = m_UsedupFront = &m_Nil;

		while( m_FeedNil.next )
		{
			RegionTag* next = m_FeedNil.next;
			m_FeedNil.DisconnectNext();
			
			#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT
				OSAllocator::Release( next->AllocationBase );
			#else
				OSAllocator::Release( next );
			#endif
		}

		m_FeedNil.next = nullptr;
		m_pFeedFront = nullptr;
	}



	void PoolAllocator::Cleanup()
	{
		tcout << _T( "PoolAllocator::Cleanup()...\n" );

		RegionTag* feed = m_FeedNil.next;
		RegionTag* prev = &m_FeedNil;

		while( feed )
		{
			if( IsInUse( feed ) == false )
			{
				tcout << _T( "  Unused Feed found: " ) << (unsigned *)feed << tendl;

				prev->DisconnectNext();
				
				// Remove clean pages commited from feed
				tcout << _T( "    Removing commited clean pages...\n" );

				Page* page = m_CleanFront;
				while( page != &m_Nil )
				{
					Page* nextpage = page->next;
					void* base = OSAllocator::GetAllocationBase( page );					
					if( base == feed )
					{
						tcout << _T( "      " ) << (unsigned*)page << tendl;
						if( page == m_CleanFront )	m_CleanFront = nextpage;
						page->Disconnect();
					}
					page = nextpage;
				}

				if( feed == m_pFeedFront )
				{
					tcout << _T( "    Invalidating FeedFront...\n" );
					m_pFeedFront=nullptr;
				}

				OSAllocator::Release( feed );
				
				feed = prev->next;
			}
			else
			{
				prev = feed;
				feed = feed->next;
			}
		}
		
	}



	bool PoolAllocator::IsInUse( const void* mem ) const
	{		
		for( Page* p=m_Nil.next; p!=m_CleanFront; p=p->next )
		{
			if( mem==OSAllocator::GetAllocationBase( p ) )
				return true;
		}

		return false;
	}



	bool PoolAllocator::IsEmpty( const Page* p ) const
	{
		return p==&m_Nil;
	}



	PoolAllocator::PageStates PoolAllocator::GetPageState( Page* p ) const
	{
		uint16 blocks = GetPageTag(p)->NumFreeBlocks;
		return PageStates( ( blocks<m_NumActiveBlocks) << uint8(blocks==0) );
		// 0: Clean, 1: Dirty, 2: Usedup
	}



	void* PoolAllocator::AllocateBlock( Page* page )
	{
		PageTag* pPTag = GetPageTag( page );

		int blockIndex = testLSB( pPTag->FreeBits, (int32)m_BitFlagSize );

		if( blockIndex >= 0 )
		{
			// Turn off free block flag
			UnsetBit( blockIndex, pPTag->FreeBits );

			// Decrement number of free blocks
			--pPTag->NumFreeBlocks;

			//return GetPool(page) + blockIndex * m_BlockSize;
			return GetPool( page, blockIndex );
		}

		return nullptr;
	}



	bool PoolAllocator::FreeBlock( Page* page, void* pdata )
	{
		uint8* pool = GetPool( page );

		if( (uint8*)pdata < pool || (uint8*)pdata >= pool + m_PoolSize )
			return false;

		PageTag* pPTag = GetPageTag( page );

		//tcout << "Page Address = " << std::hex<< ( (uint8*)ptr & ~( sizeof(m_Data)-1 ) ) << std::dec << tendl;

		size_t offset = (uint8*)pdata - pool;// offset bytes from page start.
		//tcout << "Address offset = " << offset << "[bytes]." << tendl;

		uint32 blockIndex = uint32( offset / m_BlockSize );
		//tcout << "Memory Block Index = " << blockIndex << tendl;

		// Turn on free flag
		SetBit( blockIndex, pPTag->FreeBits );

		// Increment number of free blocks
		++pPTag->NumFreeBlocks;

		return true;
	}



	Page* PoolAllocator::GetPage( const void* ptr ) const
	{
		#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT
			size_t base	= size_t(ptr) & RegionTag::AlignmentMask;//RoundUp( (size_t)OSAllocator::GetAllocationBase( ptr ), RegionTag::Alignment );
			tcout << base % RegionTag::Alignment << tendl;
		#else
			size_t base		= (size_t)OSAllocator::GetAllocationBase( ptr );
		#endif


		RegionTag* pRTag	= (RegionTag*)base;
		size_t offset		= Round( (size_t)ptr - base, pRTag->PageSize )
							+ Round( pRTag->RegionTagSize, OSAllocator::PageSize() );// shift if RegionTag only page exists.

		if( offset >= pRTag->RegionSize )
			return nullptr;
			
		if( offset == 0 )
		{
			tcout << "GetPage from FIRST PAGE. Shifting offset by " << pRTag->RegionTagSize << "\n";
			offset += pRTag->RegionTagSize;
		}

		return (Page*)( base + offset );
	}



	Page* PoolAllocator::SafeGetPage( const void* ptr ) const
	{
		for( Page* page=m_Nil.next; page!=&m_Nil; page=page->next )
		{
			uint8* pool = GetPool( page );
			if( (uint8*)ptr >= pool && (uint8*)ptr < pool+m_PoolSize )
				return page;
		}

		return nullptr;
	}



	PageTag* PoolAllocator::GetPageTag( const Page* page ) const
	{
		return (PageTag*)( (uint8*)page + Page::HeaderSize );
	}



	uint8* PoolAllocator::GetPool( const Page* page, uint32 blockIndex ) const
	{
		return (uint8*)page + Page::HeaderSize + m_PageTagSize + blockIndex * m_BlockSize;
	}



	void PoolAllocator::InitPageBlockParams( size_t allocSize, size_t blockSize )
	{
		size_t pageSizeLimit	= allocSize - Page::HeaderSize;

		m_BitFlagSize		= DivUp( pageSizeLimit / blockSize, BitSize::uInt8 );
		m_PageTagSize		= RoundUp( sizeof(PageTag::NumFreeBlocks) + m_BitFlagSize, ByteSize::DefaultAlignment );

		m_NumActiveBlocks	= int32( ( pageSizeLimit - m_PageTagSize ) / blockSize );
		m_PoolSize			= m_NumActiveBlocks * blockSize;

		size_t activeSize = m_PoolSize + m_PageTagSize;
		size_t wastedSize = allocSize - activeSize;

		//tcout << "wastedSize = m_AllocSize - m_NumActiveBlocks * m_BlockSize - m_PageTagSize = " << wastedSize << tendl;

		m_AllocSize = wastedSize > OSAllocator::PageSize() ?  RoundUp( (size_t)activeSize, OSAllocator::PageSize() ) : allocSize;
		//m_PageDataSize =  m_AllocSize - Page::HeaderSize;

		ASSERT( m_NumActiveBlocks > 0 );
	}



	void PoolAllocator::InitFeedParams( size_t allocSize, size_t blockSize )
	{
		m_OSAllocSize			= RoundUp( (size_t)allocSize, OSAllocator::PageSize() );// 4KBページサイズでAllocSizeを切り上げる
		m_FirstPageFreeBlocks	= m_NumActiveBlocks;

		uint16 numRTagBlocks		= (uint16)DivUp( sizeof(RegionTag), (size_t)blockSize );// Number of blocks required to store RegionTag
		size_t blockAlignedRTagSize	= (size_t)numRTagBlocks * (size_t)blockSize;// blocksize aligned RegionTag size

		m_FirstPageSize		= m_OSAllocSize;
		m_RegionTagOffset	= blockAlignedRTagSize;

		if( allocSize + blockAlignedRTagSize > m_OSAllocSize )// if m_OSAllocSize is short
		{
			tcout << "m_OSAllocSize is short: ";

			// Isolate RegionTag from Page if RegionTag size exceeds 80% of allocSize. // ex1. AllocSize=4096, BlockSize=4076	// ex2. AllocSize=8192, BlockSize=8150
			if( allocSize / blockAlignedRTagSize < REGION_RTAG_RATIO )
			{
				tcout << "Isolating RegionTag from Page ...\n";
				m_FirstPageSize		+= OSAllocator::PageSize();
				m_RegionTagOffset	= OSAllocator::PageSize();
			}
			else// Embed RegionTag space inside Page ... // ex. AllocSize=4096, BlockSize=16
			{
				tcout << "Embedding RegionTag in Page ...\n";
				m_FirstPageFreeBlocks -= numRTagBlocks;
			}
		}
		//else
		//{
		//	// ex. AllocSize=4098, BlockSize=16
		//	// ex. AllocSize=4100, BlockSize=4080
		//}

		m_OSAllocationGranularity = RoundUp( m_FirstPageSize, OSAllocator::AllocationGranularity() );// 64KBアドレス空間でm_OSAllocSizeを切り上げる
	}



	RegionTag* GetRegionTag( const void* ptr )
	{
		return (RegionTag*)OSAllocator::GetAllocationBase( ptr );
	}



}// end of namespace





//######################################################################################//
//																						//
//								Deprecated implementation								//
//																						//
//######################################################################################//

	

// Deprecated. 2021.06.12
//void PoolAllocator::ClearPages()
//{
//	Page* p1, *p2;
//
//	p1 = m_Nil.next;
//
//	while( p1 != &m_Nil )
//	{
//		auto base_p1 = OSAllocator::GetAllocationBase( p1 );
//
//		p2 = p1->next;
//		while( p2 != &m_Nil )
//		{
//			auto base_p2 = OSAllocator::GetAllocationBase( p2 );
//			p2 = p2->next;
//			if( base_p1 == base_p2 )
//			{
//				p2->prev->Disconnect();
//				//Page* prev = p2->prev;
//				//prev->Disconnect();
//				//OSAllocator::Decommit( prev, m_OSAllocSize );
//			}
//		}
//
//		p1 = p1->next;
//			
//		//tcout << OSAllocator::Decommit( p1->prev, m_OSAllocSize ) << tendl;
//
//		//tcout << "Releasing OSMemory...\n";
//		//OSAllocator::DisplayMemoryInfo( m_pFeed );
//
//		OSAllocator::Release( base_p1 );
//	}
//
//	//tcout << "m_pFeed at the end of ClearPages2...\n";
//	//OSAllocator::DisplayMemoryInfo( m_pFeed );
//	m_Nil.next = m_Nil.prev = m_CleanFront = m_DirtyFront = m_UsedupFront = &m_Nil;
//
//	m_FeedNil.next = nullptr;
//	m_pFeedFront = nullptr;
//
//}



// Deprecated. 2021.06.06
//Page* PoolAllocator::AllocatePage( size_t allocSize )
//{
//	// Allocate new Page
//	uint8* mem = new uint8[ allocSize ];//(uint8*) malloc( allocSize );
//	Page* newPage = /*(Page*)mem;*/new (mem) Page;
//	
//	// Initialize PageTag
//	PageTag* pPTag = GetPageTag( newPage );
//	pPTag->Init( /*m_PageTagSize,*/ m_NumActiveBlocks, m_BitFlagSize );

//	return newPage;
//}


// Deprecated. 2021.06.06
//void PoolAllocator::BatchAllocatePages( int32 batchsize )
//{
//	for( int32 i=0; i<batchsize; ++i )
//	{
//		//============= Allocate new Page ===============//
//		uint8* mem = new uint8[ m_AllocSize ];//(uint8*) malloc( m_AllocSize );
//		Page* newPage = /*(Page*)mem;*/new (mem) Page;


//		//============= Initialize PageTag ==============//
//		PageTag* pPTag = GetPageTag( newPage );
//		pPTag->Init( /*m_PageTagSize,*/ m_NumActiveBlocks, m_BitFlagSize );


//		//=========== Deploy as "Clean" page ===========//
//		newPage->ConnectBefore( m_CleanFront );
//		m_CleanFront = newPage;

//	}// end of i loop

//}



// Deprecated. 2021.06.06
//bool PoolAllocator::FreePage( Page*& page )
//{
//	if( !page )
//		return false;

//	page->Disconnect();
//	SafeDeleteArray( (uint8*&)page );//delete [] (uint8*&)page;

//	return true;
//}



// Deprecated. 2021.06.06
//void PoolAllocator::ClearPages()
//{
//	Page* page = m_Nil.next;
//	while( page != &m_Nil )
//	{
//		Page* next = page->next;
//		SafeDeleteArray( (uint8*&)page );
//		page = next;
//	}

//	m_Nil.next = m_Nil.prev = m_CleanFront = m_DirtyFront = m_UsedupFront = &m_Nil;
//}





/*
// Reference Implementation.
	template< uint32 BLOCK_SIZE, uint32 NUM_BLOCKS >
	class MemoryBlock
	{
	public:

		// Default constructor
		MemoryBlock()
		{
			Init();
		}


		// Copy constructor
		MemoryBlock( const MemoryBlock& obj )
		{

		}


		// Move constructor
		MemoryBlock( MemoryBlock&& obj )
		{


		}


		// Destructor
		~MemoryBlock()
		{

		}


		// Copy assignment operator
		MemoryBlock& operator=( const MemoryBlock& obj )
		{
			if( this != &obj )
			{

			}

			return *this;
		}


		// Move assignment operator
		MemoryBlock& operator=( MemoryBlock&& obj )
		{
			if( this != &obj )
			{

			}

			return *this;
		}


		void Init()
		{
			//std::fill( &m_FreeBlockBitMask[0], &m_FreeBlockBitMask[ BITMASK_LENGTH-1 ], 0xffffffff );

			int length = NUM_BLOCKS;

			for( int i=0; i<BITMASK_LENGTH; ++i )
			{
				if( length >= MASK_DIGITS )
					m_FreeBlockBitMask[i] = 0xffffffff;
				else
					m_FreeBlockBitMask[i] = ~(0xffffffff << Max( length, 0) );

				length -= MASK_DIGITS;

				//tcout << std::bitset<32>(m_FreeBlockBitMask[i]) << tendl;
			}

		}


		void Clear()
		{
			std::fill( &m_FreeBlockBitMask[0], &m_FreeBlockBitMask[ BITMASK_LENGTH ], 0x0 );

		}


		template< typename T >
		T* Allocate()
		{
			//tcout << GetLSB( m_FreeBlockBitMask[0] ) << tendl;

			for( int i=0; i<BITMASK_LENGTH; ++i )
			{
				int b = GetLSB( m_FreeBlockBitMask[i] );

				if( b>=0 )
				{
					// turn off free flag
					m_FreeBlockBitMask[i] &= ~(1 << b);
					
					// convert to memory block index
					b += i * MASK_DIGITS;

					//tcout << std::bitset<32>(m_FreeBlockBitMask[i]) << tendl;

					return (T*)(m_Data + b * BLOCK_SIZE);
				}

			}

			return nullptr;
		}


		template< typename T >
		bool Free( T* &ptr )
		{
			if( !HasBlock(ptr) )
				return false;

			tcout << _T( "MemoryBlock::Free()..." ) << ptr << tendl;

			//tcout << "Page Address = " << std::hex<< ( (uintptr)ptr & ~( sizeof(m_Data)-1 ) ) << std::dec << tendl;

			auto offset = (uint8*)ptr - &m_Data[0];// offset bytes from page start.
			//tcout << "Address offset = " << offset << "[bytes]." << tendl;
		
			auto blockIndex = offset / BLOCK_SIZE;
			//tcout << "Memory Block Index = " << blockIndex << tendl;

			auto bitMaskIndex = blockIndex / MASK_DIGITS;
			//tcout << "BitMask Index = " << bitMaskIndex << tendl;

			auto bitShift = blockIndex % MASK_DIGITS;
			//tcout << "bitShift = " << bitShift << tendl;

			// Enable free flag
			m_FreeBlockBitMask[bitMaskIndex] |= (1 << bitShift );

			//tcout << std::bitset<32>(m_FreeBlockBitMask[bitMaskIndex]) << tendl;

			ptr	= nullptr;

			return true;
		}


		uint32 AllocSize() const
		{
			return BLOCK_SIZE * NUM_BLOCKS;
		}


		bool HasBlock( void* ptr ) const
		{
			return ( (uint8*)ptr >= m_Data ) && ( (uint8*)ptr < m_Data + DATA_SIZE  );
		}


		bool IsFull() const
		{
			uint32 result = 0;
			for( uint32 bitmask : m_FreeBlockBitMask )
			{
				//tcout << std::bitset<32>(bitmask) << tendl;
				result |= bitmask;
			}

			tcout << _T( "IsFull()..." ) << (result==0) << tendl;

			return result == 0;
		}


		bool IsEmpty() const
		{
			uint32 result = 0;
			for( uint32 bitmask : m_FreeBlockBitMask )
				result |= bitmask;

			return result == 0xffffffff;
		}



	private:

		enum
		{
			ALLOC_SIZE		= BLOCK_SIZE * NUM_BLOCKS,
			MASK_DIGITS		= sizeof(uint32) * 8,	// std::numeric_limits<uint32>::digits
			
			
			TAG_SIZE		= DivUp( NUM_BLOCKS, (uint32)MASK_DIGITS ),

			DATA_SIZE		= ( ALLOC_SIZE - TAG_SIZE ) / BLOCK_SIZE * BLOCK_SIZE, //ALLOC_SIZE,

			BITMASK_LENGTH	= DivUp( NUM_BLOCKS, (uint32)MASK_DIGITS )

		};

		uint8	m_Data[ ALLOC_SIZE ];
		uint32	m_FreeBlockBitMask[ BITMASK_LENGTH ];

	};

*/