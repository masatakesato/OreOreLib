#include	<oreore/common/TString.h>
#include	<oreore/memory/ReferenceWrapper.h>
using namespace OreOreLib;



struct A
{
	int val;

};


int main()
{
	{
		tcout << _T( "//=========== test int reference wrapper ===========//\n" );

		int a = 66, b = -55555;

		tcout << _T( "int a = 66, b = -55555;\n" );
		tcout << tendl;


		tcout << _T( "//=== ReferenceWrapper<int> ref( a ); ===//\n" );

		ReferenceWrapper<int> ref( a );
		tcout << ref << tendl;

		tcout << tendl;


		tcout << _T( "//==== ref.Get() = 999; ===//\n" );

		ref.Get() = 999;
		tcout << ref << tendl;

		tcout << tendl;


		tcout << _T( "//=== ref = b; ===//\n" );
		ref = b;
		tcout << ref << tendl;

		tcout << tendl;

	}
	
	tcout << tendl;

	{
		tcout << _T( "//=========== test struct reference wrapper ===========//\n" );

		A a{66}, b{-55555};

		tcout << _T( "A a{66}, b{-55555};\n" );
		tcout << tendl;


		tcout << _T( "//=== ReferenceWrapper<int> ref( a ); ===//\n" );
		ReferenceWrapper<A> ref( a );

		tcout << ref->val << tendl;

		tcout << tendl;


		tcout << _T( "//==== ref.Get() = { 999 }; ===//\n" );

		ref.Get() = { 999 };
		tcout << ref.Get().val << tendl;

		tcout << tendl;


		tcout << _T( "//=== ref = b; ===//\n" );
		ref = b;
		tcout << ref->val << tendl;

		tcout << tendl;






		//int a = 66, b = -55555;
		//ReferenceWrapper<int> ref( a );

		//tcout << ref << tendl;

	//	ref.Get() = 999;
	//
	//	tcout << ref << tendl;
	//
	//
	////	ref.Ref() = b;
	//	ref = b;
	//
	//	tcout << ref << tendl;

	}


	return 0;
}
