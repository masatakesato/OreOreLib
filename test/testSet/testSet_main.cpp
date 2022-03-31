#include	<oreore/common/TString.h>
#include	<oreore/container/Set.h>
#include	<oreore/container/StaticArray.h>

using namespace OreOreLib;



int main()
{

	const size_t tableSize = 10;

	Set<tstring/*, tableSize*/> set2;

	//while(1)
	{
		Set<tstring/*, tableSize*/> set1;


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
		const Set< tstring/*, tableSize*/ > set3 = { _T("aaa"), _T("bbb"), _T("bbb") };

		for( const auto& data : set3 )
		{
			tcout << data << tendl;
		}

	}

	tcout << tendl;

	{
		float vals[] = { 0.5f, 0.6f, 0.6f };
		const Set< float, uint64/*, tableSize*/ > set3( std::begin(vals), std::end(vals) );

		for( auto& data : set3 )
		{
			tcout << data << tendl;
		}
	}

	tcout << tendl;

	{
		const StaticArray<const char*, 3> chararray = { "VK_KHR_swapchain", "VK_KHR_swapchain", "VK_KHR_swapchain" };

		Set< std::string/*, tableSize*/ > set3( chararray.begin(), chararray.end() );

//		set3.Put( "dsfds" );


		for( auto& data : set3 )
		{
			tcout << data.c_str() << tendl;
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
