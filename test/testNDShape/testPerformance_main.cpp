#include	<chrono>
#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/mathlib/GraphicsMath.h>

#include	<oreore/container/NDArray.h>
#include	<oreore/container/NDArrayView.h>
#include	<oreore/container/NDStaticArray.h>
using namespace OreOreLib;



//https://stackoverflow.com/questions/7230621/how-can-i-iterate-over-a-packed-variadic-template-argument-list/60136761

template <class F, class First, class... Rest>
void do_for(F f, First first, Rest... rest) {
    f(first);
    do_for(f, rest...);
}
template <class F>
void do_for(F f) {
    // Parameter pack is empty.
}

template <class... Args>
void doSomething(Args... args) {
    do_for([&](auto arg) {
        // You can do something with arg here.
    }, args...);
}





int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	{
		NDArray</*Vec4d*/double, 2>	rgbimage( 16384, 16384 );

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間


		for( int y=0; y<rgbimage.Dim<int>(0); ++y )
		{
			for( int x=0; x<rgbimage.Dim<int>(1); ++x )
			{
				double& pixel = //rgbimage({ y, x });// faster
								rgbimage( y, x );// slower
				pixel = 255.0;
				
				//Vec4d& pixel = rgbimage({y, x});
				//InitVec( pixel, 255.0, 0.5, 0.0, 1.0 );
			}
		}

		//tcout << rgbimage({ 16383, 16383 }) << tendl;

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
		tcout << "time elapsed: " << elapsed << "[ms].\n";

	}

	return 0;

}