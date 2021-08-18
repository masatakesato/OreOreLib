// キャッシュデータ管理クラスのテスト
#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/CacheDataManager.h>
#include	<oreore/IndexND.h>

using namespace OreOreLib;


int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	//======================= test CacheDataManager =====================//

	// 16スロットのキャッシュデータマネージャを作成する
	CacheDataManager	m_CacheManager;
	m_CacheManager.Init( 16 );

	// 全スロットを予約する
	while( !m_CacheManager.IsFull() )
	{
		int slot_id = m_CacheManager.ReserveSlot();
		tcout << "Reserved Slot	(" << slot_id << ") from m_CacheManager" << tendl;
	}

	// 偶数スロットだけ解放する
	for( int i=0; i<m_CacheManager.numMaxSlots(); i+=2 )
	{
		m_CacheManager.FreeSlot(i);
	}


	// まとめて4個確保する
	int slot_array_start	= m_CacheManager.ReserveSlot(4);

	// まとめて3個確保する
	int slot_array_start2	= m_CacheManager.ReserveSlot(3);

	// 確保した4個を解放する
	m_CacheManager.FreeSlot( slot_array_start );

	// もう1回4個確保する
	slot_array_start	= m_CacheManager.ReserveSlot(4);

	// 要素をクリーンにする
	m_CacheManager.SetClean( slot_array_start );


	// 要素をDirtyにする
	m_CacheManager.SetDirty( slot_array_start );

	// 連続解放した要素を途中から解放する(無効な操作)
	m_CacheManager.FreeSlot( 0 );


	m_CacheManager.FreeSlot( slot_array_start );


	return 0;
}