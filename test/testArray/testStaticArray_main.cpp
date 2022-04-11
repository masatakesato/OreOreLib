// https://stackoverflow.com/questions/20086206/is-there-a-cleaner-way-to-replicate-an-unordered-map-with-multi-type-values-in-c

// http://libclaw.sourceforge.net/multi_type_map.html


#include <crtdbg.h>
#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/StaticArray.h>
#include	<oreore/container/ArrayView.h>



const size_t SIZE = 16;
using fSArray = OreOreLib::StaticArray<float, SIZE>;


int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	fSArray arr_;


	fSArray	arr1 = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f };

	//arr1.Init(0.0f, 0.1f);

	tcout << _T("//================ fill arr1 with -99999 ====================//\n");
	for( int i=0; i<SIZE; ++i )	arr1[i] = -99999;
	arr1.Display();


	tcout << _T("//================ fSArray arr1_1 = arr1; ====================//\n");
	fSArray	arr1_1 = arr1;// copy constructor
	arr1_1.Display();
	

	tcout << _T("//================ fSArray arr1_2(arr1_1); ====================//\n");
	fSArray	arr1_2(arr1_1);// copy constructor
	arr1_2.Display();


	tcout << _T("//================ arr1_1 = fArray(); ====================//\n");
	arr1_1 = fSArray();// move assignment operator
	arr1_1.Display();
	arr1_1.CopyFrom( arr1_1);

	tcout << _T("//================ fSArray arr1_3 = fArray(); ====================//\n");
	fSArray	arr1_3 = fSArray();// constructor
	arr1_3[5] = -4613;
	arr1_3[6] = 1111111.111f;
	arr1_3.Display();


	tcout << _T("//================ vec_farray.push_back( arr1_3 );  ====================//\n");
	std::vector<fSArray> vec_farray;
	vec_farray.push_back( arr1_3 );// copy constructor

	for( int i=0; i<arr1_3.Length<int>(); ++i )
		arr1_3[i] = 33;

	for( int i=0; i<vec_farray[0].Length<int>(); ++i )
		vec_farray[0][i] = 55;

	tcout << _T("//---------- vec_farray[0] ----------//\n");
	vec_farray[0].Display();

	tcout << _T("//---------- arr1_3 ----------//\n");
	arr1_3.Display();


	tcout << _T("//================ vec_farray.push_back( fArray() );  ====================//\n");
	vec_farray.push_back( fSArray() );// move constructor(all vector elements are reallocated using move constructor )

	for( int i=0; i<vec_farray[0].Length<int>(); ++i )
		vec_farray[1][i] = -7777;

	tcout << _T("//---------- vec_farray[1] ----------//\n");
	vec_farray[1].Display();



	float *data = new float[10];
	for( int i=0; i<10; ++i )	data[i] = (float)pow(2, i);

	fSArray arr2( 10, data );
	arr2.Display();

	delete [] data;




	OreOreLib::ArrayView<float> view2( arr2.begin()+3, 5 );

	view2.Display();

	tcout << _T( "//============ Invert signs =============//\n" );
	for( int i=0; i<view2.Length<int>(); ++i )	view2[i] *= -100;
	view2.Display();

	tcout << _T( "//============ view[0] = view[2] =============//\n" );
	view2[0] = view2[2];
	view2.Display();


	view2.Release();



	OreOreLib::ArrayView<float> view3( arr2.begin()+3, 5 );

	view3.Display();

	tcout << _T( "//============ Scale by 0.001 =============//\n" );
	for( int i=0; i<view3.Length<int>(); ++i )	view3[i] *= 0.001f;
	view3.Display();

	tcout << _T( "//============ view3[0] = view3[2] =============//\n" );
	view3[0] = view3[2];
	view3.Display();


	view3.Release();







	arr2.Display();

	return 0;

}
