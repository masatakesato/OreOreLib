#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/memory/DebugNew.h>
#include	<oreore/memory/Memory.h>
using namespace OreOreLib;



struct test
{
	int a;
	float b;
};



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	Memory<int, MemSizeType> mem;

	{
		// Reserve
		mem.Reserve(10);
	
		// Init
		const int defaultval[] = {-1, -2, -3, -4};
		mem.Init( &defaultval[0], &defaultval[4] );//4, (int*)defaultval );
		tcout << mem[2] << tendl;

		// Reinit
		mem.Init( 100 );
		tcout << mem[2] << tendl;

		// Rereserve
		mem.Reserve(101);

		// Release
		mem.Release();
	}

	tcout << tendl;

	{
		// Reserve
		mem.Reserve(10);
	
		// Resize
		mem.Resize( 4 );
		tcout << mem[2] << tendl;

		// Rereeize
		mem.Init( 100 );
		tcout << mem[2] << tendl;

		// Release
		mem.Release();
	}

	tcout << tendl;

	{
		// Reserve
		mem.Reserve(10);
	
		// Extend
		mem.Extend( 4 );
		tcout << mem[2] << tendl;

		// ReExtend
		mem.Extend( 100 );
		tcout << mem[2] << tendl;

		// Release
		mem.Release();
	}

	tcout << tendl;

	{
		// Reserve
		mem.Reserve(10);
	
		// Shrink
		mem.Shrink( 4 );
		tcout << mem[2] << tendl;

		// ReShrink
		mem.Shrink( 100 );
		tcout << mem[2] << tendl;

		// Release
		mem.Release();
	}

	
	return 0;

}
