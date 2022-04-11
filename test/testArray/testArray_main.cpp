// https://stackoverflow.com/questions/20086206/is-there-a-cleaner-way-to-replicate-an-unordered-map-with-multi-type-values-in-c

// http://libclaw.sourceforge.net/multi_type_map.html


#include <crtdbg.h>
#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/StaticArray.h>
#include	<oreore/container/ArrayView.h>

#include	<oreore/algorithm/Algorithm.h>



using fArray = OreOreLib::Array<float>;
using fSArray16 = OreOreLib::StaticArray<float, 16>;


//struct Val
//{
//	Val() : value()
//	{
//	tcout << "Val()\n";
//	}
//	Val( float val )
//		: value( val )
//	{
//		tcout << "Val( float val )\n";
//	}
//
//	~Val()
//	{
//		tcout << "~Val()\n";
//		//value = -999999999.9f;
//	}
//
//	Val( const Val& obj )
//		: value( obj.value )
//	{
//		tcout << "Val( const Val& obj )\n";
//	}
//
//	//Val( Val&& obj )
//	//	: value( obj.value )
//	//{
//	//	 obj.value = nullptr;
//	//}
//
//
//	float value;
//
//};



struct Val
{
	Val() : value( new float(0) )
	{
		tcout << "Val()\n";
	}

	Val( float val )
		: value( new float(val) )
	{
		tcout << "Val( float val )\n";
	}

	~Val()
	{
		tcout << "~Val(): " << tendl;
		SafeDelete( value );
	}

	Val( const Val& obj )
		: value( new float(*obj.value)  )
	{
		tcout << "Val( const Val& obj ) : " << *value << tendl;
	}

	Val( Val&& obj )
		: value( obj.value )
	{
		tcout << "Val( Val&& obj )\n";
		 obj.value = nullptr;
	}


	float* value;

};




int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	//while(1)
	{

//		Val a(0.5f);
//		Val b;
//		b = Val( std::move(a) );// Array::AddToTail( T&& src )はこれと同じ内部処理をやってる. 一次変数Valを作成した後代入演算子使ってるだけ. ムーブできてない
//		TODO: ムーブするにはどうすればいい？
//			1. move assignment operatorをValクラスに実装すれば、b = std::move(a);を記述できるようになる
//			2. Val b = Val( std::move(a) ); もしくは Val b( std::move(a) ); と記述してムーブコンストラクタを呼び出す


		//std::vector<Val> aaa;

		//aaa.push_back( 33333.33f );
		//aaa.push_back( 44444.44f );

		//Val v(999.5f);
		//aaa.push_back( v );

		//aaa.clear();


		OreOreLib::Array<Val> aaa;

		aaa.AddToTail( 33333.33f );		//aaa.AddToTail( Val(0.111f) );

		//////aaa.AddToTail( 999.5f );
		const Val v(999.5f);
		aaa.AddToFront( v );

		aaa.Extend( 2 );
		aaa.Extend( 3, -0.0001f );
		aaa.Resize(5);
		aaa.Resize(8, 10.0f);
		aaa.Resize(1);
		aaa.Release();



		//Val src[] = { 1.1f, 2.2f, 3.3f };
		//Val dst[] = { 0.0f, 0.0f, 0.0f };

		// これはダメ. assignment operatorでDeepCopy実装しないとクラッシュする
		//std::copy( std::begin(src), std::end(src), std::begin(dst) );
		//memcpy( dst, src, sizeof(Val)*3 );


		// こっちはOK？ -> メモリリーク発生する -> デストラクタ呼び出し追加した.
		//OreOreLib::Mem::Copy( dst, src, 3 );

//		*src[2].value = -666.6f;

		//return 0;
	}


	{
		float aaa[3] = {1.0f, 2.0f, 3.0f};

		fArray faaa;//( std::begin(aaa), std::end(aaa) );
		faaa.Init( &aaa[0], &aaa[3] );

	}


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

	fArray arr2( &data[0], &data[10] );// constructor
	fArray arr2_1 = arr2;// copy constructor
	fArray arr2_2(arr2);// copy constructor

	arr2_1	= fArray(3);// move assignment operator. リファレンス型のarr2_1に実体型を代入すると、実体型に変わる

	arr2_1.Init(5);

	tcout << _T( "arr2_1.Resize(16): " ) << arr2_1.Resize( 16 ) << tendl;
	tcout << _T( "arr2_2.Resize(16): " ) << arr2_2.Resize( 16 ) << tendl;	

	//arr2_1.SetValues( 0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111 );
	arr2_1.SetValues( {0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111} );

	for( int i=0; i<arr2_1.Length<int>(); ++i )
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
	for( int i=0; i<view2.Length<int>(); ++i )	view2[i] *= -100;
	view2.Display();

	tcout << _T( "//============ view[0] = view[2] =============//\n" );
	view2[0] = view2[2];
	view2.Display();

	view2.Release();

	arr2.Display();

	
	return 0;

}
