#include	<chrono>
#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/mathlib/MathLib.h>
#include	<oreore/memory/TLSF.h>
using namespace OreOreLib;


const uint32	g_MemorySize = 300;
const int N = 2;


uint8* byte_array = nullptr;
TLSF	mem;



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	tcout << _T( "//################### testTLSFMemoryAllocator:: Allocation test ####################//\n" );

	TLSF	mem;

	tcout << _T( "//============ Init =============//\n" );
	
	mem.Init( g_MemorySize );
	
	mem.Info();


	tcout << _T( "//============ Allocate 4 16[bytes] data =============//\n" );

	uint8* data1 = mem.Allocate( 16 );
	uint8* data2 = mem.Allocate( 16 );
	uint8* data3 = mem.Allocate( 16 );
	uint8* data4 = mem.Allocate( 16 );

	mem.Info();


	tcout << _T( "//============ Free 2 16[bytes] data =============//\n" );

//	mem.Free( data1 );
	mem.Free( data4 );
//	mem.Free( data2 );
	mem.Free( data3 );

	mem.Info();// ここで断片化の発生を確認できる


	tcout << _T( "//============ Compact =============//\n" );

	mem.Compact();// Execute memory compaction	
	mem.Info();


	tcout << _T( "//============ Clear =============//\n" );

	mem.Clear();

	mem.Info();

	tcout << _T( "//============ Allocate 1 16[bytes] data =============//\n" );

	uint8* data5 = (uint8 *)mem.Allocate( 16 );

	mem.Info();



	mem.Release();
	

	return 0;
}