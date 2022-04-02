#include	<oreore/common/TString.h>
#include	<oreore/container/HashMap.h>

using namespace OreOreLib;



int main()
{
	HashMap<tstring, int> hashMap;

	while(1)
	{
		for( int i=0; i<24; ++i )
		{
			tcout << i << tendl;
			hashMap.Put( to_tstring(i), i );
		}


		for( auto& val : hashMap )
		{
			tcout << val.first << _T(", Exists: ") << hashMap.Exists( val.first ) << tendl;
		}

		hashMap.Clear();
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
