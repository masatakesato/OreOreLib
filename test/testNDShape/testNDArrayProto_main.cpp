#include	<chrono>

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

	NDArray_proto<double, 2>	double2D({2, 3});// double2D(2, 3); is OK


	tcout << double2D(0, 0) << tendl;

	double2D(0, 0) = 999.45;

	tcout << double2D(0, 0) << tendl;


	tcout << double2D({0, 0}) << tendl;


	NDArrayView_proto<double, 2>	view;

	view.Init( double2D.begin(), {2, 2} );

	end = std::chrono::system_clock::now();  // 計測終了時間
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
	tcout << "time elapsed: " << elapsed << "[ms].\n";

	tcout << tendl;



	NDStaticArray_proto<double, 2, 3> sarr2d;

	sarr2d.Display();


	return 0;
}