#include	"ObjectPool.h"


#include	<oreore/common/TString.h>
#include	<oreore/common/Utility.h>


namespace OreOreLib
{


	ObjectPool::ObjectPool()
	{
		m_Capacity		= 0;
		m_numReservedSlots	= 0;
		//m_BlockStatus		= NULL;
	}



	ObjectPool::ObjectPool( uint32 capacity )
	{
		//tcout << _T("TileCacheTexture::TileCacheTexture()...") << tendl;
		Init( capacity );
	}



	ObjectPool::~ObjectPool()
	{
		//tcout << _T("TileCacheTexture::~TileCacheTexture()...") << tendl;
		Release();
	}



	void ObjectPool::Init( uint32 capacity )
	{
		//================= パラメータ初期化 =================//	
		m_Capacity		= capacity;// TileCacheに格納できるタイル数
		m_numReservedSlots	= 0;

		//=========== ステータス管理バッファの確保 ===========//
		//SafeDeleteArray( m_BlockStatus );
		//m_BlockStatus		= new SlotStatus[ m_Capacity ];
		//m_FreeSlots.Init( capacity );

		//================ バッファの初期化 ==================//
		Clear();
	}



	void ObjectPool::Resize( uint32 capacity )
	{

	}



	void ObjectPool::Release()
	{
		m_Capacity		= 0;
		m_numReservedSlots	= 0;
		//SafeDeleteArray( m_BlockStatus );
		//m_FreeSlots.Release();
	}



}// end of namespace