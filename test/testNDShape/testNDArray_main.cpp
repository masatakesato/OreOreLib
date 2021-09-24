#include	<chrono>

#include	<oreore/common/TString.h>
//#include	<oreore/meta/PeripheralTraits.h>
#include	<oreore/mathlib/GraphicsMath.h>

#include	<oreore/container/NDArray.h>
#include	<oreore/container/NDArrayView.h>
#include	<oreore/container/NDStaticArray.h>
using namespace OreOreLib;



int main()
{
	std::chrono::system_clock::time_point  start, end; // 型は auto で可
	start = std::chrono::system_clock::now(); // 計測開始時間

	/*
	{
		const int X=2, Y=3, Z=4;
		double arr3d[Z][Y][X];

		double i=0;
		for( int z=0; z<Z; ++z )
			for( int y=0; y<Y; ++y )
				for( int x=0; x<X; ++x )				
				{
					arr3d[z][y][x] = i++;
					tcout << "[" << z << "][" << y << "][" << x << "]: " << arr3d[z][y][x] << tendl;
				}
	}
	*/
	tcout << tendl;

	{
		const int X=2, Y=3, Z=4;
		NDArray<double, 3>	arr3d( Z, Y, X );
		
		double i=0;
		for( int z=0; z<arr3d.Dim<int>(0); ++z )
			for( int y=0; y<arr3d.Dim<int>(1); ++y )
				for( int x=0; x<arr3d.Dim<int>(2); ++x )
				{
					arr3d(z,y,x) = i++;
					tcout << "[" << z << "][" << y << "][" << x << "]: " << arr3d(z, y, x) << tendl;
				}

		tcout << tendl;

		arr3d.Display();
	}


	tcout << tendl;


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
		NDArrayView<double, 2> view;
		view.Init( arr2d, 1, 1, 2, 2 );
		view.SetValues( -5, -6, -7, -8 );
		view.Display();
	}

	tcout << tendl;

	{
		NDArrayView<double, 2> view;
		view.Init( arr2d, {1, 1}, {2, 2} );
		view.SetValues( -5, -6, -7, -8 );
		view.Display();
	}

	tcout << tendl;



	//view2d.Display();
	//tcout << view2d(0, 0) << tendl;
	//view2d.begin();
	NDArrayView<double, 2> view2d3( arr2d, 1, 1, 2, 2 );//view2d3( arr2d, {1, 3}, {2, 2} );//
	view2d3.Display();


	sarr2d = arr2d;
	{
		sarr2d.SetValues( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
		//auto iter = &sarr2d(0, 0);
		//for( int i=0; i<arr2d.Length(); ++i )
		//	(*iter++) = double(i);
	}
	sarr2d.Display();


	{
		NDArray<double, 2>	arr(sarr2d);
		arr.Display();
	}
	
	{
		NDArray<double, 2>	arr(view2d3);
		arr.Display();
	}

	{
		double value[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		NDStaticArray<double, 4, 4> arr(10, value);
		arr.Display();
	}

	{
		NDStaticArray<double, 4, 4> arr( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
		arr.Display();
	}

	{
		NDStaticArray<double, 4, 4> arr( {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16} );
		arr.Display();
	}



/*
	// 
	sarr2d2 = sarr2d;// operator=( const NDStaticArray& obj )
	sarr2d2 = arr2d;// operator=( const Memory<T>& obj )
	sarr2d2 = std::move(sarr2d);//operator=( NDStaticArray&& obj )
*/




	NDStaticArray<Vec3f, 4, 4>	rgbimage;

	rgbimage[0].x = 0.5f;
	rgbimage[1].x = 0.5f;

	rgbimage.Display();

	Vec3f fff;

	tcout << fff << tendl;

	end = std::chrono::system_clock::now();  // 計測終了時間
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
	tcout << "time elapsed: " << elapsed << "[ms].\n";

	tcout << tendl;



	return 0;

}