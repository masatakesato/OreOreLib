#include	<crtdbg.h>

#ifdef _WIN64
#include	<windows.h>
#endif


#include	<oreore/common/TString.h>
#include	<oreore/common/Utility.h>
#include	<oreore/mathlib/MathLib.h>
#include	<oreore/os/OSAllocator.h>
using namespace OreOreLib;


//https://www.tokovalue.jp/function/VirtualAlloc.htm

//	  https://www.codeproject.com/Questions/102942/why-VirtualQuery-return-0-sometimes



void DisplayMemInfo( const MEMORY_BASIC_INFORMATION& meminfo )
{
	tcout << "  AllocationBase: " << meminfo.AllocationBase << tendl;
	tcout << "  BaseAddress:    " << meminfo.BaseAddress << tendl;
	tcout << "  RegionSize: " << meminfo.RegionSize << tendl;
	tcout << "  State: " << std::hex << meminfo.State << std::dec << tendl;
	tcout << tendl;


	// State:
	//	10000:	MEM_FREE
	//	1000:	MEM_COMMIT
	//	2000:	MEM_RESERVE
}




void* SearchPage( void* ptr, size_t alignment )
{
	tcout << "//======== SearchPage =========//\n";
	// Get alligned page size
	size_t alignedPageSize = RoundUp( alignment, OSAllocator::PageSize() );

	// Align ptr position using aligned page size.
	uintptr alignedptr = Round( (size_t)ptr, (size_t)alignedPageSize );

	tcout << "Query Address: " << (uintptr*)ptr << tendl;
	tcout << "BaseAddress:   " << (uintptr*)alignedptr << tendl;

	return &alignedptr;
}



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	tcout << GetLargePageMinimum() << tendl;
	// https://stackoverflow.com/questions/3351940/detecting-the-memory-page-size

	tcout << "//=========================== System Info ======================================//\n";

	MEMORY_BASIC_INFORMATION meminfo;

	//GetSystemInfo( & sysInfo );
	tcout << "Page size: " << OSAllocator::PageSize() << "[bytes]\n";
	tcout << "Allocation Granularity: " << OSAllocator::AllocationGranularity() << "[bytes]\n";


	size_t PAGE_SIZE = OSAllocator::PageSize();

	tcout << tendl;



	//{
	//	tcout << "//======================= Allocate independent pages( page1 and page2 )==========================================//\n";

	//	uint8* page1 = ( uint8* )OSAllocator::ReserveUncommited( sizeof(uint8) );//(uint8*)VirtualAlloc( nullptr, sizeof( uint8 ), MEM_COMMIT, PAGE_READWRITE );
	//	uint8* page2 = (uint8*)OSAllocator::ReserveUncommited( sizeof( uint8 )*PAGE_SIZE );//VirtualAlloc( nullptr, sizeof( uint8 )*PAGE_SIZE, MEM_COMMIT, PAGE_READWRITE );

	//	tcout << "//====== page1 ======//\n";
	//	VirtualQuery( page1, &meminfo, sizeof( MEMORY_BASIC_INFORMATION ) );
	//	DisplayMemInfo( meminfo );

	//	tcout << "//====== page2 ======//\n";
	//	VirtualQuery( page2, &meminfo, sizeof( MEMORY_BASIC_INFORMATION ) );
	//	DisplayMemInfo( meminfo );


	//	tcout << "//====== Access page2 info from ptr =====//\n";
	//	tcout << "uint8& ptr = page2[3825];\n";
	//	uint8& ptr = page2[3825];
	//	VirtualQuery( &ptr, &meminfo, sizeof( MEMORY_BASIC_INFORMATION ) );
	//	DisplayMemInfo( meminfo );

	//	// Free page1 and page2
	//	OSAllocator::Release( page1 );//VirtualFree( page1, 0, MEM_RELEASE );
	//	OSAllocator::Release( page2 );//  VirtualFree( page2, 0, MEM_RELEASE );

	//}

	//tcout << tendl;

	//{
	//	tcout << "//================================ int *a = new int( 0 ); =================================//\n";

	//	int *a = new int[ 100 ];

	//	VirtualQuery( a, &meminfo, sizeof( MEMORY_BASIC_INFORMATION ) );
	//	DisplayMemInfo( meminfo );

	//	tcout << "a Address: "<< (unsigned*)a << tendl;

	//	delete a;
	//}

	//tcout << tendl;

	{
		size_t m_PageSize = 6144;//4096;//
		size_t m_vmPageSize = RoundUp( m_PageSize, OSAllocator::PageSize() );

		tcout << "//================================ ????????????????????? =================================//\n";
		uint8* reserved = (uint8*)OSAllocator::ReserveUncommited( sizeof( uint8 )*/*PAGE_SIZE*17*/4194304 );

		VirtualQuery( reserved, &meminfo, sizeof( MEMORY_BASIC_INFORMATION ) );
		DisplayMemInfo( meminfo );

		// Align commit address to dwPageSize
		auto addressOffset = 6144*2 / m_vmPageSize * m_vmPageSize;//Floor( size_t(6144*2), m_vmPageSize );// 

		// allocate 6144 bytes ( ceiled to closest dwPageSize )
		//uint8* page = (uint8*)VirtualAlloc( reserved + addressOffset, m_PageSize, MEM_COMMIT, PAGE_READWRITE );//(uint8*)OSAllocator::Commit( reserved + addressOffset, sizeof( uint8 )*PAGE_SIZE );
		uint8* page = (uint8*)OSAllocator::CommitAligned( (reserved + 6144*2), m_PageSize/2, m_PageSize );



		// 8192 bytes(2 pages) allocated. starts from 6144 assdess offset. 
		VirtualQuery( page, &meminfo, sizeof( MEMORY_BASIC_INFORMATION ) );
		tcout << "page info...\n";
		DisplayMemInfo( meminfo );

		SearchPage(page, m_PageSize );


		uint8* ptr = &page[ 8192 +55];//		size_t startOffset = m_PageSize;

		// assuming jump to "NEXT" 8192 page block
		SearchPage(ptr, m_PageSize );
		//                        addressOffset =Floor( 6114*2, vmPageSize )
		//                              |--------page-------|           *&page[8192]
		// |--------------|-------------|--------------|---- ...    ---|
		//              4096          8192           12288            16384

	
		OSAllocator::DecommitAligned( page, m_PageSize/2, m_PageSize );
		VirtualQuery( page, &meminfo, sizeof( MEMORY_BASIC_INFORMATION ) );
		tcout << "page info...\n";
		DisplayMemInfo( meminfo );



//		SearchPage( page2+4099, m_PageSize );
	}
	
	tcout << tendl;

	{
		tcout << "//========================== Reserve page and divide into buffers ============================//\n";

		uint32 ReservedSize = sizeof(uint8) * 98304;//49152;//49000;// 96KB

		tcout << "Reserve page with VirtualAlloc()...\n  ReservedSize: " << ReservedSize << "[bytes]\n";
		uint8* page = (uint8*)OSAllocator::ReserveUncommited( ReservedSize );//(uint8*)VirtualAlloc( nullptr, ReservedSize, MEM_RESERVE, PAGE_READWRITE );//

		VirtualQuery( page, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
		DisplayMemInfo( meminfo );

		uint32 CommitedSize = 0;
		uint32 BufferSize = sizeof(uint8) * 12288;// 12KB
		const int numBuffers = 8;
		uint8* buffers[numBuffers];

		for( int i=0; i<numBuffers; ++i )
		{
			tcout << "Allocating buffer from reserved page\n  BufferSize: " << BufferSize << "[bytes]\n";
			buffers[i] = page + BufferSize*i;
			(uint8*)OSAllocator::Commit( buffers[i], BufferSize ); //(uint8*)VirtualAlloc( page + BufferSize*i, BufferSize, MEM_COMMIT, PAGE_READWRITE );//
			CommitedSize += BufferSize;
		}

tcout << "Is FullyCommited? " << OSAllocator::IsFullyCommited( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is FullyReserved? " << OSAllocator::IsFullyReserved( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is Released? " << OSAllocator::IsReleased( buffers[numBuffers-2] ) << tendl;
		tcout << tendl;

		//tcout << "//========== Check buffer status =========//\n";
		//for( int i=0; i<numBuffers; ++i )
		//{
		//	VirtualQuery( buffers[i], &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );

		//	tcout << "buffers[" << i << "]\n";
		//	DisplayMemInfo( meminfo );
		//}

		tcout << tendl;


		tcout << "//========== Decommit buffers[ numBuffers-2 ] =========//\n";
		OSAllocator::Decommit( buffers[ numBuffers-2 ], BufferSize );//VirtualFree( buffers[ numBuffers-2 ], BufferSize, MEM_DECOMMIT );//
		
		VirtualQuery( buffers[numBuffers-2], &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
		DisplayMemInfo( meminfo );

tcout << "Is FullyCommited? " << OSAllocator::IsFullyCommited( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is FullyReserved? " << OSAllocator::IsFullyReserved( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is Released? " << OSAllocator::IsReleased( buffers[numBuffers-2] ) << tendl;
		tcout << tendl;


		tcout << "//========== Free buffers[ numBuffers-2 ] =========//\n";
		OSAllocator::Release( buffers[ numBuffers-2 ] );//VirtualFree( buffers[ numBuffers-2 ], 0, MEM_RELEASE );
		
		VirtualQuery( buffers[numBuffers-2], &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
		DisplayMemInfo( meminfo );

		//tcout << "IsState( MEM_RESERVE ): " << IsState( page, BufferSize, numBuffers, MEM_RESERVE ) << tendl;
tcout << "Is FullyCommited? " << OSAllocator::IsFullyCommited( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is FullyReserved? " << OSAllocator::IsFullyReserved( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is Released? " << OSAllocator::IsReleased( buffers[numBuffers-2] ) << tendl;
		tcout << tendl;


		tcout << "//========== Decommit page, and then check buffers[ numBuffers-2 ] =========//\n";
		OSAllocator::Decommit( page, ReservedSize );//VirtualFree( page, ReservedSize, MEM_DECOMMIT );


tcout << "Is FullyCommited? " << OSAllocator::IsFullyCommited( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is FullyReserved? " << OSAllocator::IsFullyReserved( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is Released? " << OSAllocator::IsReleased( buffers[numBuffers-2] ) << tendl;
		tcout << tendl;


		tcout << "//========== Free page, and then check buffers[ numBuffers-2 ] =========//\n";
		OSAllocator::Release( page );//VirtualFree( page, 0, MEM_RELEASE );
		
		for( int i=0; i<numBuffers; ++i )
		{
			VirtualQuery( buffers[i], &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );

			tcout << "buffers[" << i << "]\n";
			DisplayMemInfo( meminfo );
		}

tcout << "Is FullyCommited? " << OSAllocator::IsFullyCommited( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is FullyReserved? " << OSAllocator::IsFullyReserved( buffers[numBuffers-2], ReservedSize ) << tendl;
tcout << "Is Released? " << OSAllocator::IsReleased( buffers[numBuffers-2] ) << tendl;
		
		tcout << tendl;



		//uint8* data = &buffers[ numBuffers ][ BufferSize ];// NG. out of range
		//uint8* data = &buffers[ numBuffers ][ BufferSize-1 ];// OK
		//uint8* data = &buffers[ numBuffers-1 ][ BufferSize ];// OK? data refers buffers[ numBuffers ][ 0 ]

	}



	// VirtualAllocでReserveできる最小単位は64[KB]
	// Commit/Decommitできる最小単位は4[KB]

	// ページサイズ(16KB,etc...)に区切ってCommitする


	// ページ単位でコミット/デコミットは重い. アロケーション単位でコミット/デコミットする



	// 最初にVirtualAllocで論理アドレス空間(64KB)を予約しておく.
	// Allocateする際に、
	// IsFullyCommited



	// 64KB単位で確保と解放する
	// ページサイズ(16)毎に


	// 確保した領域を解放できるかどうかチェックする



	return 0;
}