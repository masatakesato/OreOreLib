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


	struct Page
	{
		Page *next = nullptr;
		Page *prev = nullptr;
		uint8 data[1];


		void ConnectAfter( Page* pnode );
		void ConnectBefore( Page* pnode );
		void Disconnect();
		void DisconnectPrev();
		void DisconnectNext();
		bool IsAlone() const;

	};




	// struct for accessing tag area of Page. Must be aligned to 8 bytes ( DEEAULT_ALIGNMENT ).
	struct PageTag
	{
		//uint16	PageTagSize;
		uint16	NumFreeBlocks;
		uint8	FreeBits[1];// Dynamic bitarray. Bit flags of block status. 1: free, 0: used.

		void Init( /*uint16 ptagsize,*/ uint16 numfreeblocks, size_t bitflagsize );
	};




	// VirtualMemory information. Aligned to 8 bytes( DEFUALT_ALIGNMENT ).
	struct RegionTag
	{
		RegionTag* next = nullptr;
		size_t	RegionTagSize;// size of this struct rounded up to blocksize
		size_t	RegionSize;// reserved size rounded up to dwAllocationGranularity
		size_t	PageSize;// reserved size rounded up to dwPageSize
		PoolAllocator*	pAllocator;
		//size_t	NumActivePages;// number of available pages
		//size_t	NumFreeOSPages;// number of free-to-use pages

		void Init( size_t rtagsize, size_t regionsize, size_t pagesize, PoolAllocator* pallocator );
		void ConnectAfter( RegionTag* ptag );
		void DisconnectNext();
	};




	// Pool Allocator
	class PoolAllocator
	{
	public:

		PoolAllocator();// Default constructor
		PoolAllocator( uint32 allocSize, uint32 blockSize, int32 commitBatchSize=c_COMMIT_BATCH_SIZE );// Constructor
		PoolAllocator( const PoolAllocator& obj );// Copy constructor
		PoolAllocator( PoolAllocator&& obj );// Move constructor
		~PoolAllocator();// Destructor

		PoolAllocator& operator=( const PoolAllocator& obj );// Copy assignment operator
		PoolAllocator& operator=( PoolAllocator&& obj );// Move assignment operator

		void Init( uint32 allocSize, uint32 blockSize, int32 commitBatchSize=4 );
		void Cleanup();

		void* Allocate( size_t alignment=0 );//void* Allocate();
		bool Free( void*& ptr, Page* page=nullptr );
		bool SafeFree( void*& ptr );
		void Display() const;



	private:

		
		//////////////////////////////////////////////// Feed structure //////////////////////////////////////////////////
		//																												//
		//	|                                |                         |                             |           |   |	//
		//	|========== RegionTag ===========|=== Page ===|** unused **|===== Page =====|** unused **|==...	...**|---|	//
		//																												//
		//   <-- m_RegionTagOffset [bytes]-->																			//
		//																												//
		//   <--------------- m_FirstPageSize [bytes] ----------------> <-- m_OSAllocSize [bytes]---> <-- ...			//
		//																												//
		//	 <----------------------------------- m_OSAllocationGranularity [bytes] -------------------------------->	//
		//																												//
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		////////////////////////////////// Page structure ////////////////////////////////
		//																				//
		//																				//
		//	|===== next =====|===== prev =====|=============== data ===============|	//
		//																				//
		//	 <----- c_DataOffset [bytes] ----> <------- m_PageSize [bytes] ------->		//
		//																				//
		//	 <----------------------- m_AllocSize [bytes] ------------------------>		//
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
		//   <-------------- m_TagSize [bytes] -------------->																	//
		//																														//
		//	 <-------------------------------------------- m_PageSize [bytes] --------------------------------------------->	//
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
		int32	m_CommitBatchSize;	// number of pages to commit at once
		size_t	m_AllocSize;
		size_t	m_PageSize;			// = m_AllocSize - c_DataOffset;
		size_t	m_BitFlagSize;		// = DivUp( m_PageSize / m_BlockSize, BitSize::uInt8 );
		size_t	m_PageTagSize;		// = RoundUp( sizeof(PageTag::NumFreeBlocks) + m_BitFlagSize, ByteSize::DefaultAlignment );
		int32	m_NumActiveBlocks;	// = ( m_PageSize - m_PageTagSize ) / m_BlockSize;
		size_t	m_PoolSize;			// = m_NumActiveBlocks * m_BlockSize;

		// Feed and relevant parameters.
		RegionTag	m_FeedNil;		// Nill for Virtual Memory list.
		void*	m_pFeedFront;		// Current Virtual Memory reserved from OS.
		size_t	m_OSAllocSize;
		size_t	m_OSAllocationGranularity;
		size_t	m_FirstPageSize;
		uint16	m_FirstPageFreeBlocks;
		size_t	m_RegionTagOffset;

		// Page linked list
		enum PageStates{ Clean, Dirty, Usedup, NumPageStates };
		Page	m_Nil;
		Page*	m_CleanFront;
		Page*	m_DirtyFront;
		Page*	m_UsedupFront;

		static const int32 c_DataOffset = sizeof(Page::next) + sizeof(Page::prev);// 16 bytes
		static const uint32 c_COMMIT_BATCH_SIZE = 4;


		// Page Opearations
		void BatchAllocatePages( int32 batchsize );
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
		void InitFeedParams( size_t allocSize, size_t blockSize );

	
		// Friend functions
		friend RegionTag* GetRegionTag( const void* ptr );
		//friend void ExtractMemoryInfo( const void* ptr, OreOreLib::Page*& page, OreOreLib::PoolAllocator*& alloc );

		// Deprecated implementation
		//Page* AllocatePage( size_t allocSize );// Deprecated. 2021.06.06


		friend class MemoryManager;

	};


	RegionTag* GetRegionTag( const void* ptr );
	//bool Free__( void*& ptr );


}// end of namesapce



#endif // !POOL_ALLOCATOR_H

