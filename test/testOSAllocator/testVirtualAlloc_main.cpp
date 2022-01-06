#include	<crtdbg.h>

#ifdef _WIN64
#include	<Windows.h>
#pragma comment( lib, "mincore" )
#endif


#include	<oreore/common/Utility.h>
#include	<oreore/mathlib/MathLib.h>
#include	<oreore/os/OSAllocator.h>
using namespace OreOreLib;




void DisplayMemInfo( const void* mem )
{
	MEMORY_BASIC_INFORMATION meminfo;
	VirtualQuery( mem, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );

	tcout << "//========= MEMORY_BASIC_INFORMATION =========//\n";
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



void* AllocateAligned( size_t size, size_t alignment )
{
	alignment = RoundUp( alignment, OSAllocator::PageSize() );

	auto mem = VirtualAlloc( nullptr, RoundUp( size + alignment, OSAllocator::PageSize() ), MEM_RESERVE, PAGE_READWRITE );

	tcout << "mem: " << mem << ", %alignment: "<< (size_t)mem % alignment << tendl;

    return VirtualAlloc( (PVOID)RoundUp( (size_t)mem, alignment ), size, MEM_COMMIT, PAGE_READWRITE );
}




void* AllocateAligned2( size_t size, size_t alignment )
{
    MEM_ADDRESS_REQUIREMENTS addressReqs = {0};
    MEM_EXTENDED_PARAMETER param = {0};

    addressReqs.Alignment = alignment;// dwAllocationGranularityの整数倍だと動く
    addressReqs.HighestEndingAddress = (PVOID)(ULONG_PTR)0x7fffffff;

    param.Type = MemExtendedParameterAddressRequirements;
    param.Pointer = &addressReqs;

    return VirtualAlloc2( nullptr, nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, &param, 1 );
}



void* CommitAligned_( void* ptr, size_t size, size_t alignment, bool writable, bool executable )
{
	ASSERT( alignment % OSAllocator::PageSize() == 0 && _T("Invalid alignment value. must be multiplier of OSAllocator::PageSize().") );
	// Get alligned page size
	//alignment = RoundUp( alignment, OSAllocator::PageSize() );

	// Align ptr position using aligned page size.
	uintptr alignedptr = RoundUp( (uintptr)ptr, (uintptr)alignment );
	
	tcout << "alignment:" << alignment << tendl;
	tcout << "ptr:       " << (unsigned*)(ptr) << tendl;
	tcout << "alignedptr:" << (unsigned*)(alignedptr) << tendl;

	// Align size
	size_t pageAlignedAllocSize = RoundUp( size, alignment );

	tcout << "size:                " << size << tendl;
	tcout << "pageAlignedAllocSize:" << pageAlignedAllocSize << tendl;

	return (void*)VirtualAlloc( (uintptr*)alignedptr, pageAlignedAllocSize, MEM_COMMIT, PAGE_READWRITE );

}




int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	//{
	//	MEMORY_BASIC_INFORMATION meminfo;
	//	SYSTEM_INFO sysInfo;
	//	GetSystemInfo( &sysInfo );

	//	size_t size = 1024;
	//	size_t alignment = 65536*12;
	//	auto mem = AllocateAligned( /*sizeof(float) * 256*/size, /*1024 */alignment );

	//	tcout << "aln: " << mem << ", %alignment: "<< (size_t)mem % alignment << tendl;

	//	DisplayMemInfo( mem );

	//	//tcout << VirtualFree( mem, size, MEM_DECOMMIT ) << tendl;
	//	//DisplayMemInfo( mem );

	//

	//	VirtualQuery( mem, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );

	//	tcout << VirtualFree( meminfo.BaseAddress, 0, MEM_RELEASE ) << tendl;
	//	DisplayMemInfo( /*meminfo.AllocationBase*/mem );
	//}

	{

		size_t size = 1024;
		size_t alignment = 65536*12;


		auto basemem = OSAllocator::ReserveUncommited( size + alignment );
		tcout << "basemem: " << basemem << ", %alignment: "<< (size_t)basemem % alignment << tendl;

		//DisplayMemInfo( virtualmem );
		auto alignedmem = OSAllocator::CommitAligned( basemem, size, alignment );
		tcout << "alignedmem: " << alignedmem << ", %alignment: "<< (size_t)alignedmem % alignment << tendl;

		((char *)alignedmem)[ 1023 ] = 'g';

		//DisplayMemInfo( alignedmem );
		tcout << VirtualFree( basemem, 0, MEM_RELEASE ) << tendl;
		//tcout << VirtualFree( alignedmem, 0, MEM_RELEASE ) << tendl;// NG

		
	}

	//CommitAligned_( (void *)val1, size, alignment, false, false );


	// val1をalignment1区切りに揃える
//	auto val_aligned = Round( val1, alignment );
//	tcout << "RoundUp( val1, alignment ): " << val_aligned << tendl;


	return 0;
}