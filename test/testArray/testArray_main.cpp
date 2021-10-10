// https://stackoverflow.com/questions/20086206/is-there-a-cleaner-way-to-replicate-an-unordered-map-with-multi-type-values-in-c

// http://libclaw.sourceforge.net/multi_type_map.html


#include <crtdbg.h>
#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/StaticArray.h>
#include	<oreore/container/ArrayView.h>



using fArray = OreOreLib::Array<float>;
using fSArray16 = OreOreLib::StaticArray<float, 16>;



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	{
		float aaa[3] = {1.0f, 2.0f, 3.0f};

		fArray faaa;//( std::begin(aaa), std::end(aaa) );
		faaa.Init( 3, aaa );

	}
	//fSArray16 pp;

	//fArray a0(5);

	//a0 = pp;



	fArray	arr1{ 0.5f, 0.1f, 0.3f, 0.6f, 0.8f, 0.9f, 1.1f, -5.5f, 9.6f, 0.0f };
	
	fArray	arr1_1 = arr1;// copy constructor
	
	fArray	arr1_2(arr1);// copy constructor

	arr1_1	= fArray(3);// move assignment operator

	fArray	arr1_3 = fArray(15);// constructor

	std::vector<fArray> vec_farray;

	vec_farray.push_back( arr1_3 );// copy constructor
	vec_farray.push_back( fArray(2) );// move constructor(all vector elements are reallocated using move constructor )



	float *data = new float[10];
	for( int i=0; i<10; ++i )	data[i] = (float)pow(2, i);

	fArray arr2( 10, data );// constructor
	fArray arr2_1 = arr2;// copy constructor
	fArray arr2_2(arr2);// copy constructor

	arr2_1	= fArray(3);// move assignment operator. リファレンス型のarr2_1に実体型を代入すると、実体型に変わる

	arr2_1.Init(5);

	tcout << _T( "arr2_1.Resize(16): " ) << arr2_1.Resize( 16 ) << tendl;
	tcout << _T( "arr2_2.Resize(16): " ) << arr2_2.Resize( 16 ) << tendl;	

	//arr2_1.SetValues( 0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111 );
	arr2_1.SetValues( {0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111} );

	for( int i=0; i<arr2_1.Length(); ++i )
		tcout << arr2_1[i] << tendl;

//	arr2_1.AddToFront();
//	arr2_1.AddToTail();

//	arr2_2.AddToFront();

	delete [] data;



//	arr2_1.InsertAfter( 7 );
//	arr2_1.InsertAfter( 7, -55.6f );


	arr2_1.AddToTail( -9999.66f );

//	while( arr2_1.Length()>0 )
//		arr2_1.FastRemove( 0 );

	while( arr2_1.Length()>0 )
		arr2_1.Remove( 0 );



	OreOreLib::ArrayView<float> view2( arr2.begin()+3, 5 );

	view2.Display();

	tcout << _T( "//============ Invert signs =============//\n" );
	for( int i=0; i<view2.Length(); ++i )	view2[i] *= -100;
	view2.Display();

	tcout << _T( "//============ view[0] = view[2] =============//\n" );
	view2[0] = view2[2];
	view2.Display();

	view2.Release();

	arr2.Display();

	
	return 0;

}
