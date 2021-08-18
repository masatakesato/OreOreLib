// キャッシュデータ管理クラスのテスト

#include	<oreore/common/TString.h>
#include	<oreore/CacheDataManager.h>
#include	<oreore/IndexND.h>

using namespace OreOreLib;


int main()
{
	// データの確保はN次元



	//====================== test 2D index ======================//
	Index2D	idx2d( 5, 6 );

	tcout << idx2d.ArrayIdx1D( 2, 1 ) << tendl;

	Vec2ui idx_2d;
	idx2d.ArrayIdx2D( idx_2d.xy, idx2d.ArrayIdx1D( 2, 1 ) );
	tcout << idx_2d.x << ", " << idx_2d.y << tendl;


	//====================== test 3D index ======================//
	Index3D	idx3d( 3, 4, 5 );

	tcout << idx3d.ArrayIdx1D( 2, 1, 3 ) << tendl;

	Vec3ui idx_3d;
	idx3d.ArrayIdx3D( idx_3d.xyz, idx3d.ArrayIdx1D( 2, 1, 3 ) );
	tcout << idx_3d.x << ", " << idx_3d.y << ", " << idx_3d.z << tendl;



	//====================== test 4D index ======================//
	Index4D	idx4d( 3, 4, 5, 6 );

	tcout << idx4d.ArrayIdx1D( 2, 1, 3, 4 ) << tendl;

	Vec4ui idx_4d;
	idx4d.ArrayIdx4D( idx_4d.xyzw, idx4d.ArrayIdx1D( 2, 1, 3, 4 ) );
	tcout << idx_4d.x << ", " << idx_4d.y << ", " << idx_4d.z << ", " << idx_4d.w << tendl;


	//======================= test RingQueue =====================//

	RingQueue<int> queue;


	queue.Init( 16 );


	int i=0;

	tcout << "//======================= Enqueue Operation =======================//" << tendl;
	while( !queue.IsFull() )
	{
		tcout << "enqueue value " << i << " to ring queue" << tendl;
		queue.Enqueue(i++);
	}


	tcout << "//======================= Dequeue Operation =======================//" << tendl;
	while( !queue.IsEmpty() )
	{
		int elm = queue.Dequeue();
		tcout << "dequeued value " << elm << " from ring queue" << tendl;
	}


	queue.Release();



	//======================= test CacheDataManager =====================//

	CacheDataManager	m_CacheManager;

	m_CacheManager.Init( 16 );

	while( !m_CacheManager.IsFull() )
	{
		int slot_id = m_CacheManager.ReserveSlot();
		tcout << "Reserved Slot	(" << slot_id << ") from m_CacheManager" << tendl;
	}



	return 0;
}