#include	"PoolManager.h"


#include	<oreore/common/TString.h>
#include	<oreore/common/Utility.h>


namespace OreOreLib
{


	PoolManager::PoolManager()
	{
		m_Capacity		= 0;
		m_numReservedSlots	= 0;
		m_CacheStatus		= NULL;
	}



	PoolManager::PoolManager( int max_slots )
	{
		//tcout << _T("TileCacheTexture::TileCacheTexture()...") << tendl;
		Init( max_slots );
	}



	PoolManager::~PoolManager()
	{
		//tcout << _T("TileCacheTexture::~TileCacheTexture()...") << tendl;
		Release();
	}



	void PoolManager::Init( int max_slots )
	{
		//================= パラメータ初期化 =================//	
		m_Capacity		= max_slots;// TileCacheに格納できるタイル数
		m_numReservedSlots	= 0;

		//=========== ステータス管理バッファの確保 ===========//
		SafeDeleteArray( m_CacheStatus );
		m_CacheStatus		= new SlotStatus[ m_Capacity ];
		m_FreeSlots.Init( max_slots );

		//================ バッファの初期化 ==================//
		Clear();
	}



	void PoolManager::Resize( int max_slots )
	{

	}



	void PoolManager::Release()
	{
		m_Capacity		= 0;
		m_numReservedSlots	= 0;
		SafeDeleteArray( m_CacheStatus );
		m_FreeSlots.Release();
	}



}// end of namespace