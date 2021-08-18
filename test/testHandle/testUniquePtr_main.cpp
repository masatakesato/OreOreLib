#include	<crtdbg.h>
#include	<iostream>


#include	<oreore/Vector.h>
#include	<oreore/memory/UniquePtr.h>
using namespace OreOreLib;


class AAA
{
public:
	AAA()
	{
		for( int i=0; i<256; ++i )
			val[i] = i;
	
	}

	int val[256];
};



auto l = [](int* p) { tcout << _T("lambda delete func.\n"); delete p; };


int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	//while(1)
	//{

	//UniquePtr<AAA> fu;

	//fu = new AAA();
	//fu = new AAA();

	//tcout << fu->val[128] << tendl;
	//}
	//return 0;


	{
		tcout << _T("//================== UniquePtr<fArray> test ===============//\n" );

		UniquePtr<fArray/*, DefaultDeleter<fArray>*/ > uptr = new fArray(3);// copy constructor

		uptr.Reset();
		uptr = new fArray(3);// Move assignment operator

		(*uptr)[0] = 55.541541f;
		tcout << (*uptr)[0] << tendl;

		return 0;
	}


	{
		tcout << _T("//================== UniquePtr<int> test ===============//\n" );

		UniquePtr<int/*, DefaultDeleter<int>*/ > uptr( new int(3333) );
		*uptr = -55;
		tcout << *uptr << tendl;
	}

	{
		tcout << _T("//================== UniquePtr<int> with custom deleter test ===============//\n" );

		UniquePtr<int, decltype(l) > uptr( new int(3333), l );
		*uptr = -55;
		tcout << *uptr << tendl;
	}


	{
		tcout << _T("//================== UniquePtr<int[]> test ===============//\n" );

		int* a = new int[8];
		UniquePtr< int[]/*, DefaultDeleter<int[]>*/ > uptr( /*new int[8]*/a );
		uptr[0] = 3369;
		tcout << uptr[0] << tendl;	
	}

//	Array< UniquePtr<int[]/*, DeleteArrayFunc<int>*/ > > uparr;
//	uparr.Init(3);




//	uparr[0] = UniquePtr<int[]/*, DeleteArrayFunc<int>*/ >( new int[4] );
/*
	*(uparr[0]) = 225;

	tcout << *(uparr[0]) << tendl;

	tcout << (*arrptr)[0] << tendl;
*/	
	return 0;
}