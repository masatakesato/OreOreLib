#include	<oreore/common/TString.h>
#include	<oreore/memory/ReferenceWrapper.h>
using namespace OreOreLib;



int main()
{
	int a=66;
	int b= -55555;
	ReferenceWrapper<int> ref( a );

	tcout << ref << tendl;

	ref.Get() = 999;

	tcout << ref << tendl;


//	ref.Ref() = b;
	ref = b;

	tcout << ref << tendl;

	return 0;
}




//int main()
//{
//	int a=66;
//	int b= -55555;
//
//	std::reference_wrapper<int> ref( a );
//
//	tcout << ref << tendl;
//
//	ref.get() = 999;
//	ref.
//	tcout << ref << tendl;
//
//
////	ref.Ref() = b;
//	ref = b;
//
//	tcout << ref << tendl;
//
//	return 0;
//}