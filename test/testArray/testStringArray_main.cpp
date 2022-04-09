// https://stackoverflow.com/questions/20086206/is-there-a-cleaner-way-to-replicate-an-unordered-map-with-multi-type-values-in-c

// http://libclaw.sourceforge.net/multi_type_map.html


#include <crtdbg.h>
#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/ArrayView.h>


struct String
{
	tstring str;


	String( ){}
	String( const tstring& s ) : str(s)
	{
	}

	String( const String& obj ) : str(obj.str) 
	{
	}


	String( String&& obj )
		 : str(obj.str)
	{
		//obj.str=_T("");
	}

};



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	//while(1)
	//{
	//	OreOreLib::Array<tstring>	arr1{_T("a"), _T("b"), _T("c")} ;

	//	tstring aaa=_T("sgsgs");
	//	arr1.AddToFront( aaa );
	//	arr1.AddToFront( _T("fff") );
	//	arr1.InsertAfter( 1 );
	//	arr1.Remove(0);
	//}


	while(1)
	{
		OreOreLib::Array<String>	arr1{ String(_T("a")),  String(_T("b")),  String(_T("c")) };

		String aaa{_T("sgsgs")};
		arr1.InsertBefore( 0, aaa );
		arr1.AddToFront( String(_T("fff")) );
		arr1.AddToTail( String(_T("ggg")) );
		//arr1.AddToTail( aaa );
		//arr1.InsertAfter( 1 );

		for( auto& elm : arr1 )
			tcout << elm.str << tendl;

		arr1.Remove(1);

		tcout << tendl;
	}

	return 0;

}
