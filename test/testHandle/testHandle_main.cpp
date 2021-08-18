#include	<crtdbg.h>
#include	<chrono>
#include	<iostream>
#include	<unordered_map>


#include	<oreore/MathLib.h>


#include	"HandleTable.h"
using namespace OreOreLib;



uint32 g_TableSize = 10;



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	int a = 10;
	int b = 20;

	int *p = &a;

//	uint64 h = (uint64)(/*handle*/&p);

	Handle<int> handle = &p;


	tcout << **handle << tendl << std::dec;

//	tcout << *(int*)( *(uint64*)(h) ) << tendl;

	p = &b;
	tcout << /*std::hex <<*/ **handle << tendl;

	return 0;
}


/*
int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	tcout << _T( "//################### testTLSFMemoryAllocator:: Performance Comparison ####################//\n" );


	std::chrono::system_clock::time_point	start, end;


	std::unordered_map<int, int>	map;

	start = std::chrono::system_clock::now();

	map.reserve( 5 );
	map[0] = 3;

	for( int i=0; i<10000000; ++i )
	{
		map.at(0) = 3;
		//byte_array = new byte[g_MemorySize];
		//delete[] byte_array;
	}

	end = std::chrono::system_clock::now();

	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
	tcout << _T( " map performance: " ) << elapsed << _T( " [ms].\n" );


	int arr[5];


	start = std::chrono::system_clock::now();

	for( int i=0; i<10000000; ++i )
	{
		arr[0] = 3;
	}

	end = std::chrono::system_clock::now();

	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
	tcout << _T( " array performance: " )  << elapsed << _T( " [ms].\n" );

	return 0;
}
*/