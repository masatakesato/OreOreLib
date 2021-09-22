#include	<chrono>

#include	<oreore/common/TString.h>
#include	<oreore/meta/PeripheralTraits.h>

#include	"NDArray_proto.h"
#include	"NDArrayView_proto.h"
#include	"NDStaticArray_proto.h"
using namespace OreOreLib;



int main()
{
	std::chrono::system_clock::time_point  start, end; // 型は auto で可
	start = std::chrono::system_clock::now(); // 計測開始時間




	NDArray_proto<double, 2>	arr2d({4, 4}),// double2D(2, 3); is OK
								arr2d2(arr2d);


	NDArrayView_proto<double, 2>	view2d,
									view2d2;

	NDStaticArray_proto<double, 4, 4>	sarr2d,
										sarr2d2;


	// NDArray methods

	arr2d.Init({3, 3});
	arr2d.SetValues( 0, 1, 2, 3, 4, 5, 6, 7, 8 );
	arr2d.Display();



	view2d.Init( arr2d/*.begin(), arr2d.Shape()*/, {1, 1}, {2, 2} );//view2d.Init( arr2d.begin(), arr2d.Shape(), 1, 1, 2, 2 );//
	view2d.SetValues( -5, -6, -7, -8 );

	view2d.Display();
	//tcout << view2d(0, 0) << tendl;
	//view2d.begin();
	NDArrayView_proto<double, 2> view2d3( arr2d, 1, 1, 2, 2 );//view2d3( arr2d, {1, 3}, {2, 2} );//
	view2d3.Display();


	sarr2d = arr2d;
	{
		auto iter = &sarr2d(0, 0);
		for( int i=0; i<arr2d.Length(); ++i )
			(*iter++) = double(i);
	}
	sarr2d.Display();




	NDArray_proto<double, 2>	arr2d3(sarr2d);
	arr2d3.Display();

	/*
	NDArray_proto<double, 2>	arr2d3(view2d3);
	arr2d3.Display();
	*/

	{
		double value[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		NDStaticArray_proto<double, 4, 4> arr(10, value);
		arr.Display();
	}

	{
		NDStaticArray_proto<double, 4, 4> arr( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
		arr.Display();
	}

	{
		NDStaticArray_proto<double, 4, 4> arr( {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16} );
		arr.Display();
	}



/*
	// 
	sarr2d2 = sarr2d;// operator=( const NDStaticArray_proto& obj )
	sarr2d2 = arr2d;// operator=( const Memory<T>& obj )
	sarr2d2 = std::move(sarr2d);//operator=( NDStaticArray_proto&& obj )
*/








	end = std::chrono::system_clock::now();  // 計測終了時間
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
	tcout << "time elapsed: " << elapsed << "[ms].\n";

	tcout << tendl;



	return 0;

}