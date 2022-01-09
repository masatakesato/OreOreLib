#ifndef POOL_ALLOCATOR_H
#define	POOL_ALLOCATOR_H


// Qtのプールアロケータ実装例
// https://www.qt.io/blog/a-fast-and-thread-safe-pool-allocator-for-qt-part-1

// 高度に最適化された...
// https://www.slideshare.net/DADA246/ss-13347085


// https://stackoverflow.com/questions/48095828/unsigned-char-variable-is-not-incremented


#include	"../common/Utility.h"
#include	"../mathlib/MathLib.h"



namespace OreOreLib
{
	class PoolAllocator;


	//##########################################################################################//
	//																							//
	//								structs for memory management								//
	//																							//
	//##########################################################################################//

	// VirtualMemory information. Aligned to DefaultAlignment ( 8 bytes for x64. 4 bytes for x86 ).
	struct RegionTag
	{
		RegionTag* next = nullptr;
		size_t	RegionTagSize;// size of this struct rounded up to blocksize
		size_t	RegionSize;// reserved size rounded up to dwAllocationGranularity
		size_t	PageSize;// reserved size rounded up to dwPageSize
		PoolAllocator*	pAllocator;

#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT

		void* AllocationBase = nullptr;// Allocation base address. Required to Free allocated virtual address space.
		static const size_t /*REGION_TAG_ALIGNMENT*/Alignment = 4194304;// 4MiB. RegionTag alignment for address bitmask access.
		// RegionTagのアラインメント. 任意ポインタの下位22ビットをゼロにすればRegionTagに到達できる
		static const size_t AlignmentMask = 0xFFFFFFFFFFC00000;

#endif // ENABLE_VIRTUAL_ADDRESS_ALIGNMENT

		//size_t	NumActivePages;// number of available pages
		//size_t	NumFreeOSPages;// number of free-to-use pages

		void Init( size_t rtagsize, size_t regionsize, size_t pagesize, PoolAllocator* pallocator );
		void ConnectAfter( RegionTag* ptag );
		void DisconnectNext();

	};



	struct Page
	{
		Page* next = nullptr;
		Page* prev = nullptr;
		uint8 data[1];


		void ConnectAfter( Page* pnode );
		void ConnectBefore( Page* pnode );
		void Disconnect();
		void DisconnectPrev();
		void DisconnectNext();
		bool IsAlone() const;


		static const size_t HeaderSize = sizeof(Page*) * 2;// next/prev byte length
	};



	// struct for accessing tag area of Page. Must be aligned to DefaultAlignment ( 8 bytes for x64. 4 bytes for x86 ).
	struct PageTag
	{
		//uint16	PageTagSize;
		uint16	NumFreeBlocks;
		uint8	FreeBits[1];// Dynamic bitarray. Bit flags of block status. 1: free, 0: used.

		void Init( /*uint16 ptagsize,*/ uint16 numfreeblocks, size_t bitflagsize );
	};




	//##########################################################################################//
	//																							//
	//										Pool Allocator										//
	//																							//
	//##########################################################################################//

	class PoolAllocator
	{
	public:

		static const uint32 COMMIT_BATCH_SIZE = 4;

		PoolAllocator();// Default constructor
		PoolAllocator( uint32 allocSize, uint32 blockSize, uint32 commitBatchSize=COMMIT_BATCH_SIZE );// Constructor
		PoolAllocator( const PoolAllocator& obj );// Copy constructor
		PoolAllocator( PoolAllocator&& obj );// Move constructor
		~PoolAllocator();// Destructor

		PoolAllocator& operator=( const PoolAllocator& obj );// Copy assignment operator
		PoolAllocator& operator=( PoolAllocator&& obj );// Move assignment operator

		void Init( size_t allocSize, size_t blockSize, uint32 commitBatchSize=4 );
		void Cleanup();

		void* Allocate( size_t alignment=0 );//void* Allocate();
		bool Free( void*& ptr, Page* page=nullptr );
		bool SafeFree( void*& ptr );
		void Display() const;

		uint8* GetPoolBase( const void* ptr ) const;// ポインタが所属するプール先頭アドレスを取得する



	private:

		
		//////////////////////////////////////////////////////////// Feed structure //////////////////////////////////////////////////////////////////
		//																																			//
		// ( *for RegionTag access with address masking )																							//
		//	|               |                                       |                         |                              |             |   |	//
		//	|<- alignment ->|=============== RegionTag =============|=== Page ===|** unused **|====== Page =====|** unused **|==...   ...**|---|	//
		//	|				|                                                                                                                  |	//
		//	|				| <-- m_AlignedRegionTagSize [bytes] -->                                                                           |	//
		//	|				|                                                                                                                  |	//
		//	|				| <--------------- m_AlignedFirstPageSize [bytes] ---------------> <- m_AlignedPageSize [bytes]-> <-- ...          |	//
		//	|				|                                                                                                                  |	//
		//	|				| <--------------------------------------- m_AlignedReserveSize [bytes] -----------------------------------------> |	//
		//																																			//
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		////////////////////////////////// Page structure ////////////////////////////////
		//																				//
		//																				//
		//	|===== next =====|===== prev =====|=============== data ===============|	//
		//																				//
		//	 <--- Page::HeaderSize [bytes] --> <----- m_PageDataSize [bytes] ----->		//
		//																				//
		//	 <------------------------ m_PageSize [bytes] ------------------------>		//
		//																				//
		//////////////////////////////////////////////////////////////////////////////////


		///////////////////////////////////////////////// Page::data structure ///////////////////////////////////////////////////
		//																														//
		//	|                   PageTag                       |                  Pool                  |						//
		//																														//
		//	|== NumFreeBlocks ==|========= FreeBits ==========|========================================|*** unused area ****|	//
		//																														//
		//	 <--- 2 [bytes] ---> <-- m_BitFlagSize [bytes] --> <---------- m_PoolSize [bytes] --------> 						//
		//																														//
		//   <------------ m_PageTagSize [bytes] ------------>																	//
		//																														//
		//	 <------------------------------------------ m_PageDataSize [bytes] ------------------------------------------->	//
		//																														//
		//																														//
		//	PageTag: Management data area. Can be accessed via GetPageTag() method.												//
		//	Pool: Acttive memory area. Can be accessed via GetPool() method.													//
		//																														//
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		///////////////////////////////////////// Pool structure /////////////////////////////////////////////
		//																									//
		//	|===========================|===========================| ... |===========================|		//
		//																									//
		//	 <-- m_BlockSize [bytes] --> <-- m_BlockSize [bytes] -->  ...  <-- m_BlockSize [bytes] -->		//
		//																									//
		//	 <-------------- m_PoolSize = m_NumActiveBlocks * m_BlockSize [bytes] ------------------->		//
		//																									//
		//////////////////////////////////////////////////////////////////////////////////////////////////////


		// Page structural paremeters
		size_t	m_BlockSize;
		uint32	m_CommitBatchSize;	// number of pages to commit at once
		size_t	m_PageSize;//m_AllocSize;
		//size_t	m_PageDataSize;			// = m_AllocSize - Page::HeaderSize;
		size_t	m_BitFlagSize;		// = DivUp( m_PageDataSize / m_BlockSize, BitSize::uInt8 );
		size_t	m_PageTagSize;		// = RoundUp( sizeof(PageTag::NumFreeBlocks) + m_BitFlagSize, ByteSize::DefaultAlignment );
		int32	m_NumActiveBlocks;	// = ( m_PageDataSize - m_PageTagSize ) / m_BlockSize;
		size_t	m_PoolSize;			// = m_NumActiveBlocks * m_BlockSize;

		// Feed and relevant parameters.
		RegionTag	m_FeedNil;		// Nill for Virtual Memory list.
		void*	m_pFeedFront;		// Current Virtual Memory reserved from OS.
		size_t	m_AlignedPageSize;	// m_PageSize aligned by OS page size (4096 bytes etc..)
		size_t	m_AlignedReserveSize;// Virtual memory reserve size. Alinged by OS allocation granularity (64kb etc...)
		size_t	m_AlignedFirstPageSize;// Size of FirstPage (containing RegionTag )
		uint16	m_NumFirstPageActiveBlocks;
		size_t	m_AlignedRegionTagSize;// sizeof RegionTag alignmed to m_BlockSize (or OS page size if RegionTag only page allocation)

		// Page linked list
		enum PageStates{ Clean, Dirty, Usedup, NumPageStates };
		Page	m_Nil;
		Page*	m_CleanFront;
		Page*	m_DirtyFront;
		Page*	m_UsedupFront;


		// Page Opearations
		void BatchAllocatePages( uint32 batchsize );
		bool FreePage2( Page*& page );
		void ClearPages();

		// Page State Check
		bool IsInUse( const void* mem ) const;
		bool IsEmpty( const Page* p ) const;
		PageStates GetPageState( Page* p ) const;

		// Block Operations
		void* AllocateBlock( Page* page );
		bool FreeBlock( Page* page, void* pdata );

		// Byte array decoding
		Page* GetPage( const void* ptr ) const;
		Page* SafeGetPage( const void* ptr ) const;
		PageTag* GetPageTag( const Page* page ) const;
		uint8* GetPool( const Page* page, uint32 blockIndex=0 ) const;

		// Initialization
		void InitPageBlockParams( size_t allocSize, size_t blockSize );
		void InitFeedParams( size_t allocSize, size_t blockSize, size_t commitBatchSize );

	
		// Friend functions
		friend RegionTag* GetRegionTag( const void* ptr );
		//friend void ExtractMemoryInfo( const void* ptr, OreOreLib::Page*& page, OreOreLib::PoolAllocator*& alloc );


		friend class MemoryManager;

	};


	


	//##########################################################################################//
	//																							//
	//									Friend functions										//
	//																							//
	//##########################################################################################//

	RegionTag* GetRegionTag( const void* ptr );



	//void ExtractMemoryInfo( const void* ptr, OreOreLib::Page*& page, OreOreLib::PoolAllocator*& alloc )
	//{
	//	size_t base = (size_t)OreOreLib::OSAllocator::GetAllocationBase( ptr );
	//	OreOreLib::RegionTag* pRTag	= (OreOreLib::RegionTag*)base;

	//	// Get Allocator
	//	alloc = pRTag->pAllocator;

	//	// Get Page
	//	size_t offset = Round( (size_t)ptr - base, pRTag->PageSize ) + Round( pRTag->RegionTagSize, OreOreLib::OSAllocator::PageSize() );// shift if RegionTag only page exists.

	//	if( offset < pRTag->RegionSize )
	//	{
	//		if( offset == 0 )//tcout << "GetPage from FIRST PAGE. Shifting offset by " << pRTag->RegionTagSize << "\n";
	//			offset += pRTag->RegionTagSize;
	//		page = (OreOreLib::Page*)( base + offset );
	//	}
	//	else
	//	{
	//		page = nullptr;
	//	}
	//}




}// end of namesapce



#endif // !POOL_ALLOCATOR_H

