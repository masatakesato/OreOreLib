#include	<chrono>
#include	<crtdbg.h>

#include	<oreore/common/TString.h>
//#include	<oreore/meta/PeripheralTraits.h>
#include	<oreore/mathlib/GraphicsMath.h>

#include	<oreore/container/NDArray.h>
#include	<oreore/container/NDArrayView.h>
#include	<oreore/container/NDStaticArray.h>
using namespace OreOreLib;



template < typename T, int64 ...Ns >
void Access( NDArrayBase<T, Ns...>& arr )
{
	for( int i=0; i<arr.Length(); ++i )
	{
		auto& val = arr[i];// = i;
		val = i;
	}
}



//#define PERFORMANCE_CHECK

#ifdef PERFORMANCE_CHECK

const int c_LoopCount = 100000000;//0;

#else
const int c_LoopCount = 1;

#endif




int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	NDArray<double, 2>	arr2d({4, 4}),// double2D(2, 3); is OK
						arr2d2(arr2d);


	NDArrayView<double, 2>	view2d,
							view2d2;

	NDStaticArray<double, 4, 4>	sarr2d,
								sarr2d2;


	// NDArray methods

	arr2d.Init({3, 3});
	arr2d.SetValues( 0, 1, 2, 3, 4, 5, 6, 7, 8 );
	arr2d.Display();


	{
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		NDArrayView<double, 2> view;
		view.Init( arr2d, 1, 1, 2, 2 );
		view.SetValues( -5, -6, -7, -8 );
//		view.Display();

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		NDArrayView<double, 2> view;
		view.Init( arr2d, {1, 1}, {2, 2} );
		view.SetValues( -5, -6, -7, -8 );

		for( int i=0; i<c_LoopCount; ++i )
		{

		#ifdef PERFORMANCE_CHECK
			Access( view );
		#else
			view.Display();
		#endif
		}


		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;



	//view2d.Display();
	//tcout << view2d(0, 0) << tendl;
	//view2d.begin();
	NDArrayView<double, 2> view2d3( arr2d, 1, 1, 2, 2 );//view2d3( arr2d, {1, 3}, {2, 2} );//
//	view2d3.Display();


	sarr2d = arr2d;
	{
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int i=0; i<c_LoopCount; ++i )
		{
			sarr2d.SetValues( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );

			#ifdef PERFORMANCE_CHECK
				Access( sarr2d );
			#else
				sarr2d.Display();
			#endif
		}
		
		//auto iter = &sarr2d(0, 0);
		//for( int i=0; i<arr2d.Length(); ++i )
		//	(*iter++) = double(i);

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
		tcout << "time elapsed: " << elapsed << "[ms].\n";

	}
//	sarr2d.Display();


	{
		NDArray<double, 2>	arr(sarr2d);
//		arr.Display();
	}
	
	{
		NDArray<double, 2>	arr(view2d3);
//		arr.Display();
	}

	{
		double value[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		NDStaticArray<double, 4, 4> arr(10, value);
//		arr.Display();
	}

	{
		NDStaticArray<double, 4, 4> arr( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
//		arr.Display();
	}

	{
		NDStaticArray<double, 4, 4> arr( {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16} );
//		arr.Display();
	}


	tcout << tendl;

	{
		NDArray</*Vec4f*/double, 2>	rgbimage( 16384, 16384 );

	//	rgbimage[0].x = 0.5f;
	//	rgbimage[1].x = 0.5f;

	////	rgbimage.Display();


		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間


		for( int y=0; y<16384; ++y )
		{
			for( int x=0; x<16384; ++x )
			{
				double& pixel = rgbimage( y, x);
				pixel = 255.0;
				//Vec4f& pixel = rgbimage( y, x);
				//InitVec( pixel, 255.0f, 0.5f, 0.0f, 1.0f );
			}
		}


		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
		tcout << "time elapsed: " << elapsed << "[ms].\n";

	}



	tcout << tendl;



	return 0;

}