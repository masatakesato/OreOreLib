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


	//######################################################################################//
	//																						//
	//							PoolAllocator memory data structure							//
	//																						//
	//######################################################################################//



		/////////////////////////////////////////////////////////// m_pVirtualMemory /////////////////////////////////////////////////////////////////
		//																																			//
		// m_pVirtualMemory  m_pRegionBase                                                                                                          //
		//  v               v                                                                                                                       //
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


		///////////////////////////////////// Page ///////////////////////////////////////
		//																				//
		//																				//
		//	|== Page::next ==|== Page::prev ==|============ Page::data ============|	//
		//																				//
		//	 <--- Page::HeaderSize [bytes] --> <----- m_PageDataSize [bytes] ----->		//
		//																				//
		//	 <------------------------ m_PageSize [bytes] ------------------------>		//
		//																				//
		//////////////////////////////////////////////////////////////////////////////////


		///////////////////////////////////////////////////// Page::data /////////////////////////////////////////////////////////////
		//																															//
		//	|                   PageTag                                |                  Pool                  |                 |	//
		//																															//
		//	|== PageTag::NumFreeBlocks ==|===== PageTag::FreeBits =====|========================================|** unused area **|	//
		//																															//
		//	 <-------- 2 [bytes] -------> <-- m_BitFlagSize [bytes] --> <---------- m_PoolSize [bytes] -------->					//
		//																															//
		//   <----------------- m_PageTagSize [bytes] ---------------->                                                             //
		//																															//
		//	 <--------------------------------------------- m_PageDataSize [bytes] ---------------------------------------------->	//
		//																															//
		//																															//
		//	PageTag: Management data area. Can be accessed via GetPageTag() method.													//
		//	Pool: Acttive memory area. Can be accessed via GetPool() method.														//
		//																															//
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		///////////////////////////////////////////// Pool ///////////////////////////////////////////////////
		//																									//
		//	|===========================|===========================| ... |===========================|		//
		//																									//
		//	 <-- m_BlockSize [bytes] --> <-- m_BlockSize [bytes] -->  ...  <-- m_BlockSize [bytes] -->		//
		//																									//
		//	 <-------------- m_PoolSize = m_NumActiveBlocks * m_BlockSize [bytes] ------------------->		//
		//																									//
		//////////////////////////////////////////////////////////////////////////////////////////////////////




	//######################################################################################//
	//																						//
	//									Forward declaration									//
	//																						//
	//######################################################################################//
	
	struct RegionTag;
	struct Page;
	struct PageTag;
	class PoolAllocator;



	//######################################################################################//
	//																						//
	//					RegionTag (struct for virtual memory mamagement )					//
	//																						//
	//######################################################################################//
	
	struct RegionTag
	{
		RegionTag* next = nullptr;
		size_t	RegionTagSize;// size of this struct rounded up to blocksize
		size_t	RegionSize;// reserved size rounded up to virtual memory reserve granularity
		size_t	FirstPageSize;// first page size rounded up to OS page size
		size_t	PageSize;// page size rounded up to OS page size
		PoolAllocator*	pAllocator;

#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT

		void* AllocationBase = nullptr;// Allocation base address. Required to Free allocated virtual address space.
		static const size_t Alignment = 4194304;// 4MiB. RegionTag alignment for address bitmask access.
		static const size_t AlignmentMask = 0xFFFFFFFFFFC00000;// Mask for RegionTag address calculation. (lower 22 bits are zero).

#endif // ENABLE_VIRTUAL_ADDRESS_ALIGNMENT

		//size_t	NumActivePages;// number of available pages
		//size_t	NumFreeOSPages;// number of free-to-use pages


		void Init(
			size_t regionTagSize, size_t regionSize, size_t firstPageSize, size_t pageSize, PoolAllocator* allocator
			#ifdef ENABLE_VIRTUAL_ADDRESS_ALIGNMENT
			, void* base
			#endif
		);
		void ConnectAfter( RegionTag* ptag );
		void DisconnectNext();

	};




	//######################################################################################//
	//																						//
	//											Page 										//
	//																						//
	//######################################################################################//

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




	//######################################################################################//
	//																						//
	//										PageTag											//
	//																						//
	//######################################################################################//

	// struct for accessing tag area of Page.
	struct PageTag
	{
		//uint16	PageTagSize;
		uint16	NumFreeBlocks;
		uint8	FreeBits[1];// Dynamic bitarray. Bit flags of block states. 1: free, 0: used.

		void Init( /*uint16 ptagsize,*/ uint16 numfreeblocks, size_t bitflagsize );
	};




	//##########################################################################################//
	//																							//
	//										Pool Allocator										//
	//																							//
	//##########################################################################################//

	class PoolAllocator
	{
		static const uint32 DefaultCommitBatchSize = 4;
		static const uint32 DefaultPageCApacity = 4;

	public:

		PoolAllocator();// Default constructor
		PoolAllocator( uint32 allocSize, uint32 blockSize, uint32 commitBatchSize=DefaultCommitBatchSize, uint32 pageCapacity=DefaultPageCApacity );// Constructor
		PoolAllocator( const PoolAllocator& obj );// Copy constructor
		PoolAllocator( PoolAllocator&& obj );// Move constructor
		~PoolAllocator();// Destructor

		PoolAllocator& operator=( const PoolAllocator& obj );// Copy assignment operator
		PoolAllocator& operator=( PoolAllocator&& obj );// Move assignment operator

		void Init( size_t allocSize, size_t blockSize, uint32 commitBatchSize=DefaultCommitBatchSize, uint32 pageCapacity=DefaultPageCApacity );
		void Cleanup();// Release unused memory

		void* Allocate( size_t alignment=0 );//void* Allocate();
		bool Free( void*& ptr, Page* page=nullptr );
		bool SafeFree( void*& ptr );
		void Display() const;

		uint8* GetPoolBase( const void* ptr ) const;// ポインタが所属するプール先頭アドレスを取得する
		uint8* GetBlockBase( const void* ptr ) const;// ポインタが所属するブロック先頭アドレスを取得する


	private:

		//============== Private variables =================//

		// Page structural paremeters
		size_t		m_BlockSize;		// size of single data
		uint32		m_CommitBatchSize;	// number of pages to commit at once
		size_t		m_PageSize;			// page size
		//size_t	m_PageDataSize;		// = m_PageSize - Page::HeaderSize;
		size_t		m_BitFlagSize;
		size_t		m_PageTagSize;
		int32		m_NumActiveBlocks;
		size_t		m_PoolSize;

		// Feed and relevant parameters.
		size_t		m_AlignedPageSize;		// m_PageSize aligned by OS page size (4096 bytes etc..)
		size_t		m_AlignedReserveSize;	// Virtual memory reserve size. Alinged by OS allocation granularity (64kb etc...)
		size_t		m_AlignedFirstPageSize;	// Size of FirstPage (containing RegionTag )
		uint16		m_NumFirstPageActiveBlocks;
		size_t		m_AlignedRegionTagSize;	// sizeof RegionTag alignmed to m_BlockSize (or alignmed to OS page size if RegionTag only page allocation)

		// Page linked list
		enum PageStates{ Clean, Dirty, Usedup, NumPageStates };
		Page*		m_CleanFront;
		Page*		m_DirtyFront;
		Page*		m_UsedupFront;

		// Virtual memory
		RegionTag	m_RegionNil;				// Nill for Virtual Memory list.
		void*		m_pVirtualMemory;			// Current Virtual Memory reserved from OS.
		uint32		m_PageCapacity;				// Maximum number of pages m_pVirtualMemory can hold.
		uint8*		m_pCurrentCommitBase;		// Temporary variable to store base address for page commit.
		uint32		m_CurrentCommitedPageCount;// Temporary variable. Number of commited pagtes from m_pRegionBase.


		//============== Private methods =================//

		// Page Opearations
		void AllocatePages( uint32 numPages );
		bool FreePage( Page*& page );
		void ClearPages();

		// Page State Check
		bool IsInUse( const void* mem ) const;
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
		void InitFeedParams( size_t blockSize, size_t pageCapacity );

	
		//============== Friend functions ==============//

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

