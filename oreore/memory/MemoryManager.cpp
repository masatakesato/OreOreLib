#include	"MemoryManager.h"


#include	"../common/Utility.h"
#include	"../common/TString.h"
#include	"../os/OSAllocator.h"




namespace OreOreLib
{
	// UE4 のメモリブロックサイズ
	// https://qiita.com/EGJ-Yutaro_Sawada/items/4983d0ebfa945611d324

	// A Highly Optimized Portable Memory Manager
	// https://www.slideshare.net/DADA246/ss-13347085
                   

	//const uint32 MemoryManager::c_BlockSizs[ c_NumBlockSizes ] = 
	//{
	//	// Small size block ( 4[KB])
	//	8,		16,		24,		32,		// +8
	//	40,		48,		56,		64,		// +8
	//	80,		96,		112,	128,	// +16
	//	160,	192,	224,	256,	// +32

	//	// Medium size block ( 16[KB] )
	//	288,	320,	384,	448,	512,	// +64
	//	576,	640,	704,	768,	832,	896,	960,	1024,	// +64
	//	1152,	


	//	2048,

	//	4096,

	//};



	// https://github.com/esneider/malloc/blob/master/malloc.c


	//#define POOL_SIZE 89
	//const size_t MemoryManager::c_BlockSizs[ POOL_SIZE ] = {

	//		 8,    16,    24,    32,    40,    48,    56,    64,    72,    80,
	//		88,    96,   104,   112,   120,   128,   136,   144,   152,   160,
	//	   168,   176,   184,   192,   200,   208,   216,   224,   232,   240,
	//	   248,   256,   264,   272,   280,   288,   296,   304,   312,   320,
	//	   328,   336,   344,   352,   360,   368,   376,   384,   392,   400,
	//	   408,   416,   424,   432,   440,   448,   456,   464,   472,   480,
	//	   488,   496,   504,   512,   576,   640,   768,  1024,  2048,  4096,
	//		 0x2000/*8192*/,     0x4000/*16384*/,     0x8000/*32768*/,    0x10000/*65536*/,    0x20000,    0x40000,
	//		0x80000,   0x100000,   0x200000,   0x400000,   0x800000,  0x1000000,
	//	  0x2000000,  0x4000000,  0x8000000, 0x10000000, 0x20000000, 0x40000000,
	//	 0x80000000
	//};


	// binning based on mimalloc
	const size_t MemoryManager::c_BlockSizes[] =
	{
		8,		// 1000
		16,		// 10000
		24,		// 11000
		32,		// 100000

		40,		// 101000
		48,		// 110000
		56,		// 111000
		64,		// 1000000

		80,		// 1010000
		96,		// 1100000
		112,	// 1110000
		128,	// 10000000

		160,	// 10100000
		192,	// 11000000
		224,	// 11100000
		256,	// 100000000

		320,	// 101000000
		384,	// 110000000
		448,	// 111000000
		512,	// 1000000000

		640,	// 1010000000
		768,	// 1100000000
		896,	// 1110000000
		1024,	// 10000000000

		1280,	// 10100000000
		1536,	// 11000000000
		1792,	// 11100000000
		2048,	// 100000000000

		2560,	// 101000000000
		3072,	// 110000000000
		3584,	// 111000000000
		4096,	// 1000000000000

		5120,	// 1010000000000
		6144,	// 1100000000000
		7168,	// 1110000000000
		8192,	// 10000000000000

		10240,	// 10100000000000
		12288,	// 11000000000000
		14336,	// 11100000000000
		16384,	// 100000000000000

		20480,	// 101000000000000
		24576,	// 110000000000000
		28672,	// 111000000000000
		32768,	// 1000000000000000
	};



	const size_t MemoryManager::c_AllocSizes[] =
	{
		4096,
		8192,
		16384,
		32768,
		49152,
		65536,
		98304,
		131072,
	};




	// Default constructor
	MemoryManager::MemoryManager()
	{
		Init();
	}



	// Constructor
	//MemoryManager::MemoryManager( int blockSize, int pageSize )
	//{

	//}



	// Copy constructor
	MemoryManager::MemoryManager( const MemoryManager& obj )
	{

	}



	// Move constructor
	MemoryManager::MemoryManager( MemoryManager&& obj )
	{

	}


	
	// Destructor
	MemoryManager::~MemoryManager()
	{


	}



	// Copy assignment operator
	MemoryManager& MemoryManager::operator=( const MemoryManager& obj )
	{
		if( this != &obj )
		{

		}

		return *this;
	}



	// Move assignment operator
	MemoryManager& MemoryManager::operator=( MemoryManager&& obj )
	{
		if( this != &obj )
		{

		}

		return *this;
	}



	// https://www.gamedev.net/articles/programming/general-and-gameplay-programming/c-custom-memory-allocation-r3010/

	void* MemoryManager::Allocate( size_t size, size_t alignment )
	{
		tcout << _T( "MemoryManager::Allocate( size_t size, size_t alignment )...\n" );

		alignment = alignment!=ByteSize::DefaultAlignment ? alignment : 0;
		size += alignment;

		if( size < c_NumSizes )// PoolAllocator can handle "size" allocation
		{
			PoolAllocator* pAllocator = m_pSizeToPoolTable[ size ];
		
			//pAllocator->Display();
			void* mem = pAllocator->Allocate();
			//pAllocator->Display();
			//tcout << "  Allocated address: " << (uint8*) mem << tendl;

			return alignment==0
				? mem
				: (void*)RoundUp( size_t(mem), alignment );
		}
		else// PoolAllocator is unavailable for large memory allocation 
		{
			size_t allocSize = sizeof(RegionTag) + size;
			RegionTag* mem = (RegionTag*)OSAllocator::ReserveAndCommit( allocSize );
			mem->Init( sizeof(RegionTag), size, 1, nullptr );// Set RegionTag.pAllocator to nullptr
			//OSAllocator::DisplayMemoryInfo( mem );
			//tcout << "  Allocated address: " << (uint8*)mem + sizeof(RegionTag) << tendl;

			return alignment==0
				? (uint8*)mem + sizeof(RegionTag)
				: (void*)RoundUp( size_t( (uint8*)mem + sizeof(RegionTag) ), alignment );
		}

	}




	void* MemoryManager::Callocate( size_t n, size_t size, size_t alignment )
	{

		size_t total_size	= n * size;

		void* mem = Allocate( total_size, alignment );

		if( mem )
			memset( mem, 0, total_size );

		return mem;
	}



	void* MemoryManager::Reallocate( void*& mem, size_t size, size_t alignment )
	{

		if( mem==nullptr )
			return Allocate( size, alignment );

		size_t base				= (size_t)OSAllocator::GetAllocationBase( mem );
		RegionTag* pRTag		= (RegionTag*)base;
		PoolAllocator* pAlloc	= pRTag->pAllocator; 
		size_t oldsize			= pAlloc ? pAlloc->m_BlockSize : pRTag->RegionSize;
		bool bIsAligned			= size_t(mem) % alignment == 0;

		if( size <= oldsize && bIsAligned )// Do nothing if specified size is smaller than oldsize and alignment is satisfied.
		{
			return mem;
		}
		else// Reallocate memory and copy old data
		{
			void* new_mem = Allocate( size, alignment );
			memcpy( new_mem, mem, oldsize );
			Free( mem );
			
			//OSAllocator::DisplayMemoryInfo( new_mem );
			return new_mem;
		}

	}



	bool MemoryManager::Free( void*& mem )
	{

		size_t base			= (size_t)OSAllocator::GetAllocationBase( mem );
		RegionTag* pRTag	= (RegionTag*)base;
		size_t offset		= Round( (size_t)mem - base, pRTag->PageSize )
							+ Round( pRTag->RegionTagSize, OSAllocator::PageSize() );// shift if RegionTag-only page exists.
		
		if( !pRTag->pAllocator )// Allocated memory without PoolAllocator
		{
			mem = nullptr;
			return OSAllocator::Release( (void*)base );
		}
		else if( offset < pRTag->RegionSize )// Allocated memory using PoolAllocator
		{
			if( offset == 0 )//tcout << "GetPage from FIRST PAGE. Shifting offset by " << pRTag->RegionTagSize << "\n";
				offset += pRTag->RegionTagSize;

			bool result = pRTag->pAllocator->Free( (void*&)mem, (Page*)( base + offset ) );

			pRTag->pAllocator->Display();

			return result;//pRTag->pAllocator->Free( (void*&)mem, (Page*)( base + offset ) );
		}
		else// MemoryManager cannot trace Allocation
		{
			return false;
		}

	}



	void MemoryManager::Cleanup()
	{

		for( int32 i=0; i<c_NumPoolTables; ++i )
			m_PoolTable[i].Cleanup();

	}



	void MemoryManager::Display() const
	{

		for( int32 i=0; i<c_NumPoolTables; ++i )
			m_PoolTable[i].Display();

	}



	void MemoryManager::Init()
	{

		for( int32 i=0, j=0; i<c_NumPoolTables; ++i )
		{
			size_t blocksize = c_BlockSizes[ i ];

			if( blocksize * 8 > c_AllocSizes[j] ||  blocksize * 4 > c_AllocSizes[j] )	j = Min( ++j, c_NumAllocSizes-1 );
			
			auto allocsize = c_AllocSizes[j];// OS::PageSizeの整数倍? 

			//tcout << "blocksize: "<< blocksize << ", pagesize: " << allocsize << tendl;

			m_PoolTable[i].Init( allocsize, blocksize );

		}// end of i loop

		
		for( int32 pg_idx=0, sz_idx=0; pg_idx<MemoryManager::c_NumPoolTables; ++pg_idx )
		{
			while( sz_idx < 32768 && sz_idx <= c_BlockSizes[pg_idx] )
			{
				m_pSizeToPoolTable[ sz_idx++ ] = &m_PoolTable[ pg_idx ];//pg_idx;
			}
		}

	}


}// end of namespace




//######################################################################################//
//																						//
//								Deprecated implementation								//
//																						//
//######################################################################################//



// Deprecated. 2021.06.15
//void* MemoryManager::Allocate( size_t size )
//{

//	if( size < c_NumSizes )
//	{
//		PoolAllocator* pAllocator = m_pSizeToPoolTable[ size ];
//	
//		//pAllocator->Display();
//		void* mem = pAllocator->Allocate();
//		pAllocator->Display();

//		return mem;
//	}
//	else
//	{
//		size_t allocSize = sizeof(RegionTag) + size;
//		RegionTag* mem = (RegionTag*)OSAllocator::ReserveAndCommit( allocSize );

//		//OSAllocator::DisplayMemoryInfo( mem );

//		mem->Init( sizeof(RegionTag), size, 1, nullptr );

//		return (uint8*)mem + sizeof(RegionTag);
//	}

//}


