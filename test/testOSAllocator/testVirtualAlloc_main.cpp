#include	<crtdbg.h>

#ifdef _WIN64
#include	<Windows.h>
#pragma comment( lib, "mincore" )
#endif


#include	<oreore/common/Utility.h>
#include	<oreore/mathlib/MathLib.h>




//void DisplayMemInfo( const MEMORY_BASIC_INFORMATION& meminfo )
//{
//	tcout << "  AllocationBase: " << meminfo.AllocationBase << tendl;
//	tcout << "  BaseAddress:    " << meminfo.BaseAddress << tendl;
//	tcout << "  RegionSize: " << meminfo.RegionSize << tendl;
//	tcout << "  State: " << std::hex << meminfo.State << std::dec << tendl;
//	tcout << tendl;
//
//
//	// State:
//	//	10000:	MEM_FREE
//	//	1000:	MEM_COMMIT
//	//	2000:	MEM_RESERVE
//}



//void* SearchPage( void* ptr, size_t alignment )
//{
//	tcout << "//======== SearchPage =========//\n";
//	// Get alligned page size
//	size_t alignedPageSize = RoundUp( alignment, OSAllocator::PageSize() );
//
//	// Align ptr position using aligned page size.
//	size_t alignedptr = Round( (size_t)ptr, (size_t)alignedPageSize );
//
//	tcout << "Query Address: " << (uintptr*)ptr << tendl;
//	tcout << "BaseAddress:   " << (uintptr*)alignedptr << tendl;
//
//	return (void*)alignedptr;
//}



//void* AllocateAlignedBelow2GB( size_t size, size_t alignment )
//{
//    MEM_ADDRESS_REQUIREMENTS addressReqs = {0};
//    MEM_EXTENDED_PARAMETER param = {0};
//
//    addressReqs.Alignment = alignment;
//    addressReqs.HighestEndingAddress = (PVOID)(ULONG_PTR) 0x7fffffff;
//
//    param.Type = MemExtendedParameterAddressRequirements;
//    param.Pointer = &addressReqs;
//
//    return VirtualAlloc2 (
//        nullptr, nullptr,
//        size,
//        MEM_RESERVE | MEM_COMMIT,
//        PAGE_READWRITE,
//        &param, 1);
//}



void* AllocateAligned( size_t size, size_t alignment )
{
    MEM_ADDRESS_REQUIREMENTS addressReqs = {0};
    MEM_EXTENDED_PARAMETER param = {0};

    addressReqs.Alignment = alignment;// dwAllocationGranularityの整数倍だと動く
    addressReqs.HighestEndingAddress = (PVOID)(ULONG_PTR)0x7fffffff;

    param.Type = MemExtendedParameterAddressRequirements;
    param.Pointer = &addressReqs;

    return VirtualAlloc2( nullptr, nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, &param, 1 );
}




int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	SYSTEM_INFO sysInfo;
	GetSystemInfo( &sysInfo );

	//VirtualAlloc()
	auto arr = AllocateAligned( /*sizeof(float) * 256*/1024, /*1024 */65536 );

	
	tcout << arr << tendl;
	tcout << size_t(arr) % size_t( 65536*4) << tendl;
	tcout << GetLastError() << tendl;

	return 0;
}