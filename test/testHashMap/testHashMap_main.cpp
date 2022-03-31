#include	<unordered_map>


#include	<oreore/common/TString.h>
#include	<oreore/container/HashMap.h>

using namespace OreOreLib;



int main()
{


	const size_t tableSize = 10;

	HashMap<tstring, int/*, tableSize*/> hmap2;

	//while(1)
	{
		HashMap<tstring, int/*, tableSize*/> hmap;


		hmap.Put( _T("Value1"), -6666 );
		hmap.Put( _T("Value2"), 6666 );

		
		int value=5;
		hmap.Get( _T("Value1"), value );


		tcout << "hmap.Exists( _T(\"gfdsgds\") )...\n";
		tcout << hmap.Exists( _T("gfdsgds") ) << tendl;

		tcout << "//========== hmap[ _T(\"Value3\") ] = 9965; =========//\n";
		hmap[ _T("Value3") ] = 9965;
		tcout << hmap[ _T("Value3") ] << tendl;

		//hmap[ _T("Value2") ] = 4;
		//tcout << hmap[ _T("Value2") ] << tendl;


		tcout << "//===== for( auto iter = hmap.begin(); iter != hmap.end(); ++iter ) ========//\n";
		for( auto iter = hmap.begin(); iter != hmap.end(); ++iter )
		{
			tcout << iter->first << " " << iter->second << tendl;
		}


		tcout << "//===== for( const auto& pair : hmap ) ========//\n";
		for( const auto& pair : hmap )
		{
			tcout << pair.first << " " << pair.second << tendl;
		}


		hmap2 = std::move(hmap);
	}


	for( const auto& pair : hmap2 )
	{
		tcout << pair.first << " " << pair.second << tendl;
	}


	tcout << tendl;

	
//	Pair<float, float> a = { {0.5f, 0.5f} };

//	std::initializer_list< int > aaa;
	const HashMap< tstring, float, uint8/*, tableSize*/ > hmap3 =
	{
		{ _T("aaa"), 0.1f },	//Pair<tstring, float>{ _T("aaa"), 0.1f },
		{ _T("bbb"), -5.5f }	//Pair<tstring, float>{ _T("bbb"), -5.5f }
	};



	tcout << hmap3.At(_T("aaa")) << tendl;
	tcout << hmap3[ _T("aaa") ] << tendl;
	tcout << hmap3[ _T("bbb") ] << tendl;


	return 0;
}
