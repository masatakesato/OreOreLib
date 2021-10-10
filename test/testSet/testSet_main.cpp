#include	<oreore/common/TString.h>
#include	<oreore/container/Set.h>

using namespace OreOreLib;



int main()
{

	const size_t tableSize = 10;

	Set<tstring, tableSize> set2;


	{
		Set<tstring, tableSize> set1;


		set1.Put( _T("Value1") );
		set1.Put( _T("Value2") );

		
		//int value=5;
		//set1.Get( _T("Value1") );


		tcout << "set1.Exists( _T(\"gfdsgds\") )...\n";
		tcout << set1.Exists( _T("gfdsgds") ) << tendl;


		tcout << "//===== for( auto iter = set1.begin(); iter != set1.end(); ++iter ) ========//\n";
		for( auto iter = set1.begin(); iter != set1.end(); ++iter )
		{
			tcout << *iter << tendl;
		}


		tcout << "//===== for( const auto& pair : set1 ) ========//\n";
		for( const auto& data : set1 )
		{
			tcout << data << tendl;
		}


		set2 = std::move(set1);
	}


	for( const auto& data : set2 )
	{
		tcout << data << tendl;
	}


	tcout << tendl;


	{
		const Set< tstring, tableSize > set3 = { _T("aaa"), _T("bbb"), _T("bbb") };

		for( const auto& data : set3 )
		{
			tcout << data << tendl;
		}

	}


	{
		const Set< float, tableSize > set3 = { 0.5f, 0.6f, 0.6f };

		for( auto& data : set3 )
		{
			float& p = (float&)data;
			//*p = 5.54f;
			//tcout << data << tendl;
		}


		for( auto& data : set3 )
		{
//			float& p = data;
			tcout << data << tendl;
		}

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
