#include	<chrono>

#include	<oreore/common/TString.h>
#include	<oreore/container/NDShape.h>
using namespace OreOreLib;



const int c_LoopCount = 1;//10000000;//0;

int main()
{
	


	NDShape<3> idx( (size_t)4, (size_t)4, (size_t)2 );//, (size_t)6 );

	idx.Init( 3, 3, 3 );
	idx.Disiplay();



	{
		tcout << "NDShape::To1D( const T (&indexND)[N] ) const...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int k=0; k<c_LoopCount; ++k)
		for( int z=0; z<idx.Dim(2); ++z )
		{
			for( int y=0; y<idx.Dim( 1 ); ++y )
			{
				for( int x=0; x<idx.Dim(0); ++x )
				{
					int index[3] = {x, y, z };
					auto idx1d = idx.To1D( index );
					tcout << "[" << z << "][" << y << "][" << x << "] -> [" << idx1d << "]\n";
				}
			}
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "NDArray::To1D( const Args& ... args ) const...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int k=0; k<c_LoopCount; ++k)
		for( int z=0; z<idx.Dim(2); ++z )
		{
			for( int y=0; y<idx.Dim(1); ++y )
			{
				for( int x=0; x<idx.Dim(0); ++x )
				{
					auto idx1d = idx.To1D( x, y, z );
					tcout << "[" << z << "][" << y << "][" << x << "] -> (" << idx1d << ")\n";
				}
			}
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "NDArray::To1D( std::initializer_list<T> indexND ) const...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int k=0; k<c_LoopCount; ++k)
		for( int z=0; z<idx.Dim(2); ++z )
		{
			for( int y=0; y<idx.Dim( 1 ); ++y )
			{
				for( int x=0; x<idx.Dim(0); ++x )
				{
					auto idx1d = idx.To1D( {x, y, z} );
					tcout << "[" << z << "][" << y << "][" << x << "] -> (" << idx1d << ")\n";
				}
			}
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "NDArray::From3DTo1D( const T& x, const T& y, const T& z ) const...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int k=0; k<c_LoopCount; ++k)
		for( int z=0; z<idx.Dim(2); ++z )
		{
			for( int y=0; y<idx.Dim( 1 ); ++y )
			{
				for( int x=0; x<idx.Dim(0); ++x )
				{
					auto idx1d = idx.From3DTo1D( x, y, z );
					tcout << "[" << z << "][" << y << "][" << x << "] -> (" << idx1d << ")\n";
				}
			}
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "NDArray::ToND( uint64 indexd1D, T indexND[] ) const...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int k=0; k<c_LoopCount; ++k)
		for(int i=0; i<27; ++i )
		{
			int idx_[3];
			idx.ToND( i, idx_ );
			tcout << "[" << i << "] -> [" << idx_[2] << "][" << idx_[1] << "][" << idx_[0] << "]\n";
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "NDArray::ToND( uint64 index1D, Args& ... args ) const...\n";
		
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int k=0; k<c_LoopCount; ++k)
		for(int i=0; i<27; ++i )
		{
			int x, y, z;
			idx.ToND( i, x, y, z );
			tcout << "[" << i << "] -> [" << z << "][" << y << "][" << x << "]\n";
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}


	return 0;
}