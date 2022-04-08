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


	Memory<int> mem;

	//while(1)
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

	//while(1)
	{
		// Reserve
		mem.Reserve(10);
	
		// Extend using Reallocate
		mem.Reallocate( mem.Capacity() + 4 );
		tcout << mem[2] << tendl;

		// Rereeize
		mem.Init( 100 );
		tcout << mem[2] << tendl;

		// Release
		mem.Release();
	}

	tcout << tendl;

	//while(1)
	{
		// Reserve
		mem.Reserve(10);
	
		// Extend using Reallocate
		mem.Reallocate( mem.Capacity() + 4 );
		tcout << mem[2] << tendl;

		// ReExtend
		mem.Reallocate( mem.Capacity() + 100 );
		tcout << mem[2] << tendl;

		// Release
		mem.Release();
	}

	tcout << tendl;

	while(1)
	{
		// Reserve
		mem.Reserve(10);
	
		// Shrink using Reallocate
		mem.Reallocate( mem.Capacity() - 4 );
		tcout << mem[2] << tendl;

		// ReShrink with wrong parameter. -> Max(0, mem.Capacity()-100) 
		mem.Reallocate( /*mem.Capacity() -100*/ Max(0, mem.Capacity()-100)  );
		tcout << mem[2] << tendl;

		// Release
		mem.Release();
	}

	
	return 0;

}
