#include	<chrono>

#include	<oreore/common/TString.h>
#include	<oreore/container/NDShape.h>
//#include	<oreore/container/NDShape_colmajor.h>
using namespace OreOreLib;


const int64 N=4;

void LOOP_INCR()
{
	//for( auto i=1; i<N; ++i ){}
	int i=0;
	while( i<N )
		i++;
}

void LOOP_DECR()
{

	for( int i=N-1; i>=1; --i ){}
}





#define PERFORMANCE_CHECK

#ifdef PERFORMANCE_CHECK

const int c_LoopCount = 100000000;//0;

#else
const int c_LoopCount = 1;

#endif

int main()
{
	
	//{
	//	const int X=2, Y=3, Z=4;
	//	NDShape<3>	shape( Z, Y, X );
	//	
	//	double i=0;
	//	for( int z=0; z<shape.Dim(0); ++z )
	//		for( int y=0; y<shape.Dim(1); ++y )
	//			for( int x=0; x<shape.Dim(2); ++x )
	//			{
	//				tcout << "[" << z << "][" << y << "][" << x << "]: " << shape.From3DTo1D( z,y,x )  << tendl;
	//			}
	//	//arr3d.Display();
	//	

	//}



	NDShape<3, int32> idx( (size_t)4, (size_t)4, (size_t)2 );//, (size_t)6 );

	idx.Init( 4, 3, 2 );
//	idx.Disiplay();



	{
		tcout << "NDShape::To1D( const T indexND[] ) const...\n";

		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int k=0; k<c_LoopCount; ++k)
		for( int z=0; z<idx.Dim<int>(0); ++z )
		{
			for( int y=0; y<idx.Dim<int>(1); ++y )
			{
				for( int x=0; x<idx.Dim<int>(2); ++x )
				{
					int index[3] = { z, y, x };
					auto idx1d = idx.To1D( index );
					#ifndef PERFORMANCE_CHECK
					tcout << "[" << z << "][" << y << "][" << x << "] -> [" << idx1d << "]\n";
					#endif
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
		for( int z=0; z<idx.Dim<int>(0); ++z )
		{
			for( int y=0; y<idx.Dim<int>(1); ++y )
			{
				for( int x=0; x<idx.Dim<int>(2); ++x )
				{
					auto idx1d = idx.To1D( z, y, x );
					#ifndef PERFORMANCE_CHECK
					tcout << "[" << z << "][" << y << "][" << x << "] -> (" << idx1d << ")\n";
					#endif
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
		for( int z=0; z<idx.Dim<int>(0); ++z )
		{
			for( int y=0; y<idx.Dim<int>(1); ++y )
			{
				for( int x=0, X=idx.Dim<int>(0); x<X; ++x )
				{
					auto idx1d = idx.To1D( {z, y, x} );
					#ifndef PERFORMANCE_CHECK
					tcout << "[" << z << "][" << y << "][" << x << "] -> (" << idx1d << ")\n";
					#endif
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
		for( int z=0, Z=idx.Dim<int>(0); z<Z; ++z )
		{
			for( int y=0; y<idx.Dim<int>(1); ++y )
			{
				for( int x=0; x<idx.Dim<int>(2); ++x )
				{
					auto idx1d = idx.From3DTo1D( z, y, x );
					#ifndef PERFORMANCE_CHECK
					tcout << "[" << z << "][" << y << "][" << x << "] -> (" << idx1d << ")\n";
					#endif
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
		for(int i=0; i<idx.Size<int>(); ++i )
		{
			int idx_[3];
			idx.ToND( i, idx_ );
			#ifndef PERFORMANCE_CHECK
			tcout << "[" << i << "] -> [" << idx_[0] << "][" << idx_[1] << "][" << idx_[2] << "]\n";
			#endif
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;

	{
		tcout << "NDArray::ToND( uint64 indexd1D, int dim ) const...\n";
		
		std::chrono::system_clock::time_point  start, end; // 型は auto で可
		start = std::chrono::system_clock::now(); // 計測開始時間

		for( int k=0; k<c_LoopCount; ++k)
		for(int i=0; i<idx.Size<int>(); ++i )
		{
			int z, y, x;
			z = idx.ToND<int>(i, 0);
			y = idx.ToND<int>(i, 1);
			x = idx.ToND<int>(i, 2);
			#ifndef PERFORMANCE_CHECK
			tcout << "[" << i << "] -> [" << z << "][" << y << "][" << x << "]\n";
			#endif
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
		for(int i=0; i<idx.Size<int>(); ++i )
		{
			//LOOP_INCR();
			//LOOP_DECR();

			int z, y, x;
			idx.ToND( i, z, y, x );
			#ifndef PERFORMANCE_CHECK
			tcout << "[" << i << "] -> [" << z << "][" << y << "][" << x << "]\n";
			#endif
		}

		end = std::chrono::system_clock::now();  // 計測終了時間
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

		tcout << "time elapsed: " << elapsed << "[ms].\n";
	}

	tcout << tendl;


	return 0;
}