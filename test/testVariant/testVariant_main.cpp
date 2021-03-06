#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/memory/SharedPtr.h>
#include	<oreore/memory/WeakPtr.h>
#include	<oreore/container/Variant2.h>
using namespace OreOreLib;


#include	<memory>
#include	<string>
using namespace std;



auto Add( int a, int b )
{
	return a + b;
}



void func( int& val )
{

	val++;
}



auto Add_( const Variant2& a, const Variant2& b )
{
	Variant2 c =a;
//	return a + b;
}





int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	//SharedPtr<int> sp( new int(5) );
	////shared_ptr<int> sp( new int(5) );

	//void* valaa = sp;



	//tcout << valaa << tendl;

	//return 0;


	// unsafe cast example. Causes assertion failure.
	Variant2 vvv;
	vvv = 6;
	tcout << (short)vvv << tendl;
	//return 0;


	{

		tcout << "//====================== Pointer test ===================//\n";
		int* a = new int(999);

		Variant2 refa;//( a );
		//refa = a;// Variant2& operator=( T& obj )
		refa = a;
		
		//func( (int*)refa );

		int* b = new int(9999);
		Variant2 refb = b;
		tcout << ( (int)refa!=(int)refb) << tendl;

		
		tcout << "---------- *a = 4; ---------\n";
		*a = 4;
		tcout << "a = " << *a << tendl;
		tcout << "refa = " << *((int*)refa) << tendl;


		tcout << "---------- *(int*)refa = 777; ---------\n";
		(int&)refa = 777;//*(int*)refa = 777;
		tcout << "a = " << *a << tendl;
		tcout << "refa = " << (int)refa << tendl;


		auto& ppp = (int&)refa;
		ppp = -111;
		tcout << "a = " << *a << tendl;
		tcout << "refa = " << (int)refa << tendl;
		tcout << "refa = " << (int*)refa << tendl;


		SafeDelete( a );

		//return 0;
	}

	tcout << tendl;

	{

		tcout << "//====================== Reference test ===================//\n";
		int a = 999;

		Variant2 refa( a );
		//refa = a;// Variant2& operator=( T& obj )
		refa = &a;
		
		//func( (int*)refa );

		tcout << "---------- a = 4; ---------\n";
		a = 4;

		//int& p = *(int*)refa;

		tcout << "a = " << a << tendl;
		tcout << "refa = " << *(int*)refa << tendl;
		tcout << "refa = " << (int*)refa << tendl;

		//const int* ffff = (int*)refa;


		tcout << "---------- refa = 777; ---------\n";
		*(int*)refa = 777;//*(int*)refa = 777;// (int&)refa = 777;

		tcout << "a = " << a << tendl;
		tcout << "refa = " << *(int*)refa << tendl;

//		return 0;
	}


	tcout << tendl;


	tcout << "//======================  ===================//\n";


	Variant2 var;

	var = tstring( _T("TestString") );

	tcout << ((tstring)var).c_str()<< tendl;

//	while(1)
	{
		var = string("Hello World");
		tcout << ((string)var).c_str() << tendl;

		auto tmp_str = string("Hello World2");
		var = (string&&)tmp_str;
		tcout << ((string)var).c_str() << tendl;


		var = nullptr;

	
		SharedPtr<int> ival = new int(6);//shared_ptr<int> ival( new int(6) );
		var = ival;//(SharedPtr<int>)ival;
		tcout << *(SharedPtr<int>)var << tendl;


		SharedPtr<int> ival2 = new int(-222);
		var = ival2;
		tcout << *(SharedPtr<int>)var << tendl;


		//SharedPtr<int> ival3 = new int(-222);
		var = SharedPtr<int>( new int(-222) );//ival3;
		tcout << *(SharedPtr<int>)var << tendl;

		
		var = 6.5666f;
		tcout << float(var) << tendl;


		var = WeakPtr<int>(ival2  );//dval1;
		tcout << *(WeakPtr<int>)var << tendl;

	}

	//tcout << *WeakPtr<int>(var) << tendl;


	Variant2 var2 = SharedPtr<int>( new int(88888888) );//6.5666f;
	tcout << "var2 = " << *(SharedPtr<int>)var2 << tendl;

	Variant2 var3 = -66666666;
	tcout << "var3 = " << (int)var3 << tendl;

	tcout << "Add( *(SharedPtr<int>)var2, var3 ) = " << Add( *(SharedPtr<int>)var2, var3 ) << tendl;


	 Add_( *(SharedPtr<int>)var2, var3 );

	return 0;
}