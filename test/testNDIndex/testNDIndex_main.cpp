#include	<chrono>

#include	<oreore/common/TString.h>
#include	<oreore/container/NDShape.h>
using namespace OreOreLib;



const int c_LoopCount = 10000000;

int main()
{
	


	NDShape<3> idx( (size_t)4, (size_t)4, (size_t)2 );//, (size_t)6 );

	idx.Init( 3, 3, 3 );
	idx.Disiplay();



	{
		tcout << "array...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int i=0; i<c_LoopCount; ++i)
		for( int z=0; z<idx.Dim(2); ++z )
		{
			for( int y=0; y<idx.Dim( 1 ); ++y )
			{
				for( int x=0; x<idx.Dim(0); ++x )
				{
					int index[3] = {x, y, z };
					auto idx1d = idx.To1D( index );//idx.To1D( x, y, z );//
					//tcout << "(" << x << ", " << y << ", " << z << ") -> (" << idx1d << ")\n";
				}
			}
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "variadic templates...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int i=0; i<c_LoopCount; ++i)
		for( int z=0; z<idx.Dim(2); ++z )
		{
			for( int y=0; y<idx.Dim( 1 ); ++y )
			{
				for( int x=0; x<idx.Dim(0); ++x )
				{
					//int index[3] = {x, y, z };
					auto idx1d = idx.To1D( x, y, z );//idx.To1D( index );//
					//tcout << "(" << x << ", " << y << ", " << z << ") -> (" << idx1d << ")\n";
				}
			}
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "initializer list...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int i=0; i<c_LoopCount; ++i)
		for( int z=0; z<idx.Dim(2); ++z )
		{
			for( int y=0; y<idx.Dim( 1 ); ++y )
			{
				for( int x=0; x<idx.Dim(0); ++x )
				{
					auto idx1d = idx.To1D( {x, y, z} );
					//tcout << "(" << x << ", " << y << ", " << z << ") -> (" << idx1d << ")\n";
				}
			}
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "fixed arguments...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int i=0; i<c_LoopCount; ++i)
		for( int z=0; z<idx.Dim(2); ++z )
		{
			for( int y=0; y<idx.Dim( 1 ); ++y )
			{
				for( int x=0; x<idx.Dim(0); ++x )
				{
					auto idx1d = idx.From3DTo1D( x, y, z );//idx.To1D( index );//
					//tcout << "(" << x << ", " << y << ", " << z << ") -> (" << idx1d << ")\n";
				}
			}
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;




	for(int i=0; i<27; ++i )
	{
		int idx_[3];
		idx.ToND( i, idx_ );

		tcout << "(" << i << ") -> (" << idx_[0] << ", " << idx_[1] << ", " << idx_[2] << ")\n";
	}


	return 0;
}