//#include	<chrono>
//#include	<crtdbg.h>
//
//#include	<oreore/common/TString.h>
//#include	<oreore/MathLib.h>
//
//
//#include	"TLSF.h"
//using namespace OreOreLib;
//
//
//const unsigned int	g_MemorySize = 300;
//const int N = 2;
//
//
//byte* byte_array = nullptr;
//TLSFMemoryAllocator	mem;
//
//
//int main()
//{
//	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
//
//
//	tcout << _T( "//################### testTLSF:: Allocation test ####################//\n" );
//
//	TLSFMemoryAllocator	mem;
//
//	tcout << _T( "//============ Init =============//\n" );
//	
//	mem.Init( g_MemorySize );
//	
//	mem.Info();
//
//
//	tcout << _T( "//============ Allocate 4 16[bytes] data =============//\n" );
//
//	byte* data1 = mem.Allocate( 16 );
//	byte* data2 = mem.Allocate( 16 );
//	byte* data3 = mem.Allocate( 16 );
//	byte* data4 = mem.Allocate( 16 );
//
//	mem.Info();
//
//
//	tcout << _T( "//============ Free 2 16[bytes] data =============//\n" );
//
////	mem.Free( data1 );
//	mem.Free( data4 );
////	mem.Free( data2 );
//	mem.Free( data3 );
//
//	mem.Info();// ここで断片化の発生を確認できる
//
//
//	tcout << _T( "//============ Resize =============//\n" );
//
//	mem.Resize( g_MemorySize * 2 );
//
//	mem.Release();
//	
//
//	return 0;
//}

#include <iostream>
#include <vector>




int main()
{

	std::vector<float> vec ={ 1.0f, 2.0f, 3.0f };

	float *data = &vec[2];

	std::cout << *data << std::endl;


	vec.resize( 5 );

	std::cout << *data << std::endl;

	return 0;
}