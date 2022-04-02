#include	<oreore/common/TString.h>
#include	<oreore/container/Set.h>
#include	<oreore/container/StaticArray.h>

using namespace OreOreLib;



int main()
{
	Set<int> set;

	while(1)
	{
		for( int i=0; i<24; ++i )
		{
			tcout << i << tendl;
			set.Put( i );
		}


		for( auto& val : set )
		{
			tcout << val << _T(", Exists: ") << set.Exists( val ) << tendl;
		}

		set.Clear();
	}

	return 0;
}



// unsafe set usage.
//#include	<set>
//using namespace std;
//
//#include	<oreore/common/TString.h>
//
//
//int main()
//{
//
//	set<int> aa= {1, 1, 2, 2, 3, 3, 4 };
//
//	for( auto& val : aa )
//	{
//		int& v = (int&)val;
//		v = 5;
//		tcout << val << tendl;
//	}
//
//	return 0;
//}
