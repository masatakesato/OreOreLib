#include	"OSAllocator.h"


#ifdef _WIN64
#include	<Windows.h>
#endif


#include	"../common/Utility.h"
#include	"../common/TString.h"
#include	"../mathlib/MathLib.h"


// https://github.com/WebKit/webkit/blob/main/Source/WTF/wtf/OSAllocator.h



namespace OreOreLib
{
	namespace OSAllocator
	{
		static SystemInfo c_SysInfo;


		const DWORD c_Protect[] =
		{							// writable/executable
			PAGE_READONLY,			// 00
			PAGE_EXECUTE_READ,		// 01
			PAGE_READWRITE,			// 10
			PAGE_EXECUTE_READWRITE,	// 11
		};


		const DWORD c_Status[] =
		{
			MEM_FREE,
			MEM_RESERVE,
			MEM_COMMIT,
		};




		size_t PageSize()
		{
			return c_SysInfo.m_PageSize;
		}



		size_t AllocationGranularity()
		{
			return c_SysInfo.m_AllocationGranularity;
		}



		void* GetBaseAddress( const void* ptr )
		{
			static MEMORY_BASIC_INFORMATION meminfo;
			VirtualQuery( ptr, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
			return  meminfo.BaseAddress;
		}



		void* GetAllocationBase( const void* ptr )
		{
			static MEMORY_BASIC_INFORMATION meminfo;
			VirtualQuery( ptr, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
			return  meminfo.AllocationBase;
		}



		// ReserveUncomitted
		void* ReserveUncommited( size_t size, bool writable, bool executable )
		{
			void* mem = (void*)VirtualAlloc( nullptr, size, MEM_RESERVE, c_Protect[(DWORD)writable<<1 | (DWORD)executable] );
			return mem;
		}



		// Commit
		void* Commit( void* ptr, size_t size, bool writable, bool executable )
		{
			return (void*)VirtualAlloc( ptr, size, MEM_COMMIT, c_Protect[(DWORD)writable<<1 | (DWORD)executable] );
		}



		// CommitAligned
		//void* CommitAligned( void* ptr, size_t size, size_t alignment, bool writable, bool executable )
		//{
		//	// Get alligned page size
		//	size_t alignedPageSize = RoundUp( alignment, OSAllocator::PageSize() );

		//	// Align ptr position using aligned page size.
		//	uintptr alignedptr = Round( (size_t)ptr, (size_t)alignedPageSize );
		//	//tcout << (unsigned*)( ((size_t)ptr / (size_t)alignedPageSize) * (size_t)alignedPageSize ) << tendl;
		//	//tcout << "alignedptr:" << (unsigned*)(alignedptr) << tendl;

		//	// Align size
		//	size_t alignedSize = RoundUp( size, alignedPageSize );


		//	return (void*)VirtualAlloc( (uintptr*)alignedptr, alignedSize, MEM_COMMIT, c_Protect[(DWORD)writable<<1 | (DWORD)executable] );
		//}


		void* CommitAligned( void* ptr, size_t size, size_t alignment, bool writable, bool executable )
		{
			ASSERT( alignment % OSAllocator::PageSize() == 0 && _T("Invalid alignment value. must be multiplier of OSAllocator::PageSize().") );
			
			return VirtualAlloc( (PVOID)RoundUp( (size_t)ptr, alignment ), size, MEM_COMMIT, c_Protect[(DWORD)writable<<1 | (DWORD)executable] );
		}



		// ReserveAndCommit
		void* ReserveAndCommit( size_t size, bool writable, bool executable )
		{
			void* mem = (void*)VirtualAlloc( nullptr, size, MEM_RESERVE | MEM_COMMIT, c_Protect[(DWORD)writable<<1 | (DWORD)executable] );
			return mem;
		}



		// ReserveAndCommitAligned
		void* ReserveAndCommitAligned( size_t size, size_t alignment, bool writable, bool executable )
		{
			ASSERT( alignment % OSAllocator::PageSize() == 0 && _T("Invalid alignment value. must be multiplier of OSAllocator::PageSize().") );

			// Reserve Virtual Address space
			auto mem = VirtualAlloc( nullptr, size + alignment, MEM_RESERVE, c_Protect[(DWORD)writable<<1 | (DWORD)executable] );

			//tcout << "mem: " << mem << ", %alignment: "<< (size_t)mem % alignment << tendl;
			// Commit memory using aligned start address
			return VirtualAlloc( (PVOID)RoundUp( (size_t)mem, alignment ), size, MEM_COMMIT, c_Protect[(DWORD)writable<<1 | (DWORD)executable] );
		}



		// Decommit
		bool Decommit( void* ptr, size_t size )
		{
			return VirtualFree( ptr, size, MEM_DECOMMIT ) ? true : false;
		}



		// DecommitAligned
		bool DecommitAligned( void* ptr, size_t size, size_t alignment )
		{
			// Get alligned page size
			size_t alignedPageSize = RoundUp( alignment, OSAllocator::PageSize() );

			// Align ptr position using aligned page size.
			uintptr alignedptr = Round( (size_t)ptr, (size_t)alignedPageSize );

			// Align size
			size_t alignedSize = RoundUp( size, alignedPageSize );

			return VirtualFree( (uintptr*)alignedptr, alignedSize, MEM_DECOMMIT ) ? true : false;
		}



		// ReleaseDecommited. ptr adress must be retured value from VirtualAlloc
		bool Release( void* ptr )
		{
			return VirtualFree( ptr, 0, MEM_RELEASE ) ? true : false;
		}



		// ReallocateAndCommit
		void* ReallocateAndCommit( void* src, size_t srcSize, size_t newSize, bool writable, bool executable )
		{
			// Allocate new memory
			void* mem = (void*)VirtualAlloc( nullptr, newSize, MEM_RESERVE | MEM_COMMIT, c_Protect[(DWORD)writable<<1 | (DWORD)executable] );

			// copy data
			size_t size = Min(srcSize, newSize);
			memcpy_s( mem, size, src, size );

			// free src
			VirtualFree( src, 0, MEM_RELEASE );

			return mem;
		}



		void* FindRegion( void* ptr, Status status, size_t regionSize, size_t searchRange )
		{
			if( !ptr )	return nullptr;

			uint8* mem = (uint8*)ptr;
			uint8* end = mem + searchRange;

			MEMORY_BASIC_INFORMATION meminfo;
			VirtualQuery( mem, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );

			while( mem < end )
			{
				if( meminfo.State == c_Status[status] && meminfo.RegionSize >= regionSize )
					return mem;

				mem += meminfo.RegionSize;
				VirtualQuery( mem, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
			}

			return nullptr;
		}




		//######################### Check if entire page is Reserved (Can be Released) #############################

		// * MEM_RESERVE と MEM_COMMIT が混在するケース
		// - MEM_RESERVE と MEM_FREE が混在するケース
		// - MEM_FREE と MEM_COMMIT が混在するケース
		// - MEM_RESERVE と MEM_COMMIT と MEM_FREE が混在するケース
		// * 全てMEM_RESERVEのケース...........................................(1)
		// * 全てMEM_COMMITのケース............................................(2)
		// * 全てMEM_FREEのケース...........................................(3)



		// (1) 確保した論理アドレス空間が空でOSに返却できる
		bool IsFullyReserved( void* ptr, size_t regionSize )
		{
			MEMORY_BASIC_INFORMATION meminfo;

			VirtualQuery( ptr, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
			VirtualQuery( meminfo.AllocationBase, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );

			return ( meminfo.State==MEM_RESERVE && meminfo.RegionSize==regionSize );
		}



		// (2) 確保した論理アドレス空間が満杯でOSから新規取得が必要
		bool IsFullyCommited( void* ptr, size_t regionSize )
		{
			MEMORY_BASIC_INFORMATION meminfo;

			VirtualQuery( ptr, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
			VirtualQuery( meminfo.AllocationBase, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );

			return ( meminfo.State==MEM_COMMIT && meminfo.RegionSize==regionSize );
		}



		// (3) 論理アドレス空間自体が未使用、無効なアドレス空間
		bool IsReleased( void* ptr )
		{
			MEMORY_BASIC_INFORMATION meminfo;

			VirtualQuery( ptr, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );
			//VirtualQuery( meminfo.AllocationBase, &meminfo, sizeof(MEMORY_BASIC_INFORMATION) );

			return ( meminfo.State==MEM_FREE && meminfo.AllocationBase==0 );
		}


		
		void DisplayMemoryInfo( const void* ptr )
		{
			MEMORY_BASIC_INFORMATION meminfo;
			VirtualQuery( ptr, &meminfo, sizeof(meminfo) );

			tcout << _T("  AllocationBase: ") << meminfo.AllocationBase << tendl;
			tcout << _T("  BaseAddress:    ") << meminfo.BaseAddress << tendl;
			tcout << _T("  RegionSize: ") << meminfo.RegionSize << tendl;
			tcout << _T("  State: ") << std::hex << meminfo.State << std::dec << tendl;
			tcout << tendl;

			// State:
			//	10000:	MEM_FREE
			//	1000:	MEM_COMMIT
			//	2000:	MEM_RESERVE
		}



		//bool IsState( void* ptr, size_t size, size_t numBuffers, DWORD state )
		//{
		//	MEMORY_BASIC_INFORMATION meminfo;

		//	for( uint32 i=0; i<numBuffers; ++i )
		//	{
		//		VirtualQuery( (uint8*)ptr + size*i, &meminfo, sizeof( MEMORY_BASIC_INFORMATION ) );
	
		//		if( meminfo.State != state )
		//			return false;
		//	}

		//	return true;
		//}



		SystemInfo::SystemInfo()
		{
			SYSTEM_INFO sysInfo;
			GetSystemInfo( &sysInfo );

			m_PageSize	= sysInfo.dwPageSize;
			m_AllocationGranularity	= sysInfo.dwAllocationGranularity;
		}

	}

}// end of namespace
