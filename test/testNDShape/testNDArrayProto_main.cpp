#include	<chrono>

#include	<oreore/common/TString.h>
#include	"NDArray_proto.h"
using namespace OreOreLib;



const int c_LoopCount = 1000000;//0;

int main()
{
	std::chrono::system_clock::time_point  start, end; // 型は auto で可
	start = std::chrono::system_clock::now(); // 計測開始時間



	NDArray_proto<double, 2>	double2D(2, 3);


	tcout << double2D(0, 0) << tendl;

	double2D(0, 0) = 999.45;

	tcout << double2D(0, 0) << tendl;


	tcout << double2D({0, 0}) << tendl;




	end = std::chrono::system_clock::now();  // 計測終了時間
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
	tcout << "time elapsed: " << elapsed << "[ms].\n";

	tcout << tendl;

	return 0;
}