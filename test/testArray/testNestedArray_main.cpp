// https://stackoverflow.com/questions/20086206/is-there-a-cleaner-way-to-replicate-an-unordered-map-with-multi-type-values-in-c

// http://libclaw.sourceforge.net/multi_type_map.html


#include <string>
#include <crtdbg.h>
#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/ArrayView.h>




class Class
{
public:

	Class(){}
	Class( const Class& obj )
		: val( obj.val )
	{

	}


	int val = 0;


};



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	//while(1)
	{
		tcout << "//================== float 2d array test ===================//\n";


		//tcout <<  std::is_trivially_copyable_v< OreOreLib::Array<float> > << tendl;

		OreOreLib::Array< OreOreLib::Array<float> > array2d;

		array2d.Init( 4 );

		int val=0;
		array2d[0].AddToTail( float(val++) );
		array2d[0].AddToTail( float(val++) );
		array2d[0].AddToTail( float(val++) );
		array2d[0].AddToTail( float(val++) );

		array2d[1].AddToTail( float(val++) );
		array2d[1].AddToTail( float(val++) );
		array2d[1].AddToTail( float(val++) );
		array2d[1].AddToTail( float(val++) );

		array2d[2].AddToTail( float(val++) );
		array2d[2].AddToTail( float(val++) );
		array2d[2].AddToTail( float(val++) );
		array2d[2].AddToTail( float(val++) );

		array2d[3].AddToTail( float(val++) );
		array2d[3].AddToTail( float(val++) );
		array2d[3].AddToTail( float(val++) );
		array2d[3].AddToTail( float(val++) );


		for( int i=0; i<array2d.Length(); ++i )
		{
			for( int j=0; j<array2d[i].Length(); ++j )
			{
				tcout << array2d[i][j] << ", ";
			}
			tcout << tendl;

		}

		tcout << tendl;
	}


	// memcpy NG case
	//{
	//	// https://stackoverflow.com/questions/5272370/copying-arrays-of-strings-in-c
	//	std::wstring* m_pData = new std::wstring[ 4 ]();
	//	m_pData[0] = _T("1111");
	//	m_pData[1] = _T("2222");
	//	m_pData[2] = _T("3333");
	//	m_pData[3] = _T("4444");

	//	std::wstring* m_pData2 = new std::wstring[ 4 ]();

	//	memcpy( m_pData2, m_pData, 4 * sizeof(std::wstring) );// NG
	//	//std::copy( m_pData, m_pData+4, m_pData2 );// OK

	//	SafeDeleteArray( m_pData2 );
	//	SafeDeleteArray( m_pData );
	//
	//	return 0;
	//}

	//while(1)
	{
		tcout << "//================== string add test ===================//\n";

		OreOreLib::Array< tstring > strarr;
		strarr.AddToTail( _T("1") );
		strarr.AddToTail( _T("2") );
		strarr.AddToFront( _T("0") );

		for( auto& val : strarr )
			tcout << val << tendl;
	}

	//while(1)
	{
		tcout << "//================== string 2d array test ===================//\n";

		OreOreLib::Array< OreOreLib::Array<std::wstring> > array2d;

		array2d.Init( 4 );
	
		int val=0;
		array2d[0].AddToTail( to_tstring(val++) );
		array2d[0].AddToTail( to_tstring(val++) );
		array2d[0].AddToTail( to_tstring(val++) );
		array2d[0].AddToTail( to_tstring(val++) );

		array2d[1].AddToTail( to_tstring(val++) );
		array2d[1].AddToTail( to_tstring(val++) );
		array2d[1].AddToTail( to_tstring(val++) );
		array2d[1].AddToTail( to_tstring(val++) );

		array2d[2].AddToTail( to_tstring(val++) );
		array2d[2].AddToTail( to_tstring(val++) );
		array2d[2].AddToTail( to_tstring(val++) );
		array2d[2].AddToTail( to_tstring(val++) );

		array2d[3].AddToTail( to_tstring(val++) );
		array2d[3].AddToTail( to_tstring(val++) );
		array2d[3].AddToTail( to_tstring(val++) );
		array2d[3].AddToTail( to_tstring(val++) );


		for( int i=0; i<array2d.Length(); ++i )
		{
			auto& arr = array2d[i];
			for( int j=0; j<arr.Length(); ++j )
			{
				tcout << arr[j].c_str() << ", ";
			}
			tcout << tendl;

		}

		tcout << tendl;


	}


	


	return 0;

}
