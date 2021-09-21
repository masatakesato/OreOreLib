﻿#include	<chrono>

#include	<oreore/common/TString.h>
#include	<oreore/meta/PeripheralTraits.h>

#include	"NDArray_proto.h"
#include	"NDArrayView_proto.h"
#include	"NDStaticArray_proto.h"
using namespace OreOreLib;



template < int N >
class Base
{
public:

	Base()
	{
		tcout << N << tendl;
	}


};



template<unsigned ... args>
class BBB : Base< sum_<args...>::value >
{
  public:
  
  BBB()
  {
      auto aa= { args... };
      
      for( auto v : aa )
        tcout << v << tendl;
  }
    
    
};







int main()
{
	std::chrono::system_clock::time_point  start, end; // 型は auto で可
	start = std::chrono::system_clock::now(); // 計測開始時間

/*
//	struct sum_<1, 2, 3, 4> aaa;
//	tcout << aaa.value;
	BBB<1, 2, 3, 4> bbb;

	return 0;
*/


	NDArray_proto<double, 2>	arr2d({4, 4}),// double2D(2, 3); is OK
								arr2d2;

	NDArrayView_proto<double, 2>	view2d,
									view2d2;

	NDStaticArray_proto<double, 4, 4>	sarr2d,
										sarr2d2;


	// NDArray methods


	arr2d.Init({3, 3});
	auto iter = &arr2d(0,0);
	for( int i=0; i<arr2d.Length(); ++i )
		(*iter++) = double(i);

	tcout << arr2d(0, 0) << tendl;
//	tcout << arr2d({0, 0}) << tendl;
	arr2d.Display();


//TODO: 連続メモリ領域ではなく、N次元空間上で窓枠領域を切り出す -> MatrixViewの多次元拡張
//	view2d.Init( arr2d );

	view2d.Init( arr2d, {1, 1}, {2, 2} );
	view2d.Display();

	tcout << view2d(0, 0) << tendl;
	

	sarr2d = arr2d;
	sarr2d.Display();
//	sarr2d[4] = ;

	tcout << sarr2d(0, 0) << tendl;
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