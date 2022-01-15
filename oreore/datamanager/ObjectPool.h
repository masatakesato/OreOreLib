#ifndef	OBJECT_POOL_H
#define	OBJECT_POOL_H

#include	<windows.h>
#include	<process.h>

#include	<oreore/mathlib/GraphicsMath.h>
//#include	<oreore/container/RingQueue.h>
#include	<oreore/memory/Memory.h>



namespace OreOreLib
{

	//##################################################################################//
	//																					//
	//									ObjectPool										//
	//																					//
	//##################################################################################//

	class ObjectPool
	{
		static const uint16 INVALID = (uint16)~0u;

	public:

		static const uint16 N = 4096;

		ObjectPool();
		ObjectPool( uint32 capacity );
		~ObjectPool();
		ObjectPool( const ObjectPool& ) = delete;

		void Init( uint32 capacity );
		void Resize( uint32 capacity );
		void Release();

		int Capacity() const				{ return m_Capacity; }			// キャッシュが保持するスロット数を返す
		int numReservedSlots() const		{ return m_numReservedSlots; }		// 使用中のスロット数を返す

		bool IsFull() const					{ return m_numReservedSlots>=m_Capacity; }//m_FreeSlots.IsEmpty(); }	// スロットが満杯かどうかチェックする(m_FreeSlotsが空なら満杯)
		bool IsEmpty() const				{ return m_numReservedSlots==0; }//m_FreeSlots.IsFull(); }	// 全スロットが未使用かどうかチェックする(m_FreeSlotsが満杯ならtrue)
		bool IsRererved( int slot_id) const	{ return (m_BlockStatus + slot_id)->reserved; }
		bool IsDirty( int slot_id ) const	{ return (m_BlockStatus + slot_id)->reserved && (m_BlockStatus + slot_id)->dirty; }	// 最近データへのアクセスがあったかどうか調べる
		bool IsClean( int slot_id ) const	{ return (m_BlockStatus + slot_id)->reserved && !(m_BlockStatus + slot_id)->dirty; }// アクセスなしの放置状態になっているかどうか調べる
		
		int ReserveSlot();						// Reserve available slot
		bool ReserveSlots( Memory<int>& slots );// Reserve multiple slots

		bool FreeSlot( int slot_id );			// Free slot
		bool FreeSlots( Memory<int>& slots );	// Free multiple slots

		bool SetClean( int slot_id );		// 指定スロットのアクセスフラグをCleanに設定する
		bool SetDirty( int slot_id );		// 指定スロットのアクセスフラグをDirtyに設定する

		void Clear();						// キャッシュを初期状態（何もデータを保持しない）にする


	private:

		struct BlockStatus
		{
			bool dirty;		// true: recently accessed to slot, false: no recent access
			bool reserved;	// true: slot is reserved, false: slot is free to use
		};

		int				m_Capacity;			// データスロット最大数
		int				m_numReservedSlots;	// 使用中スロットの数
		BlockStatus		m_BlockStatus[ N-1 ];		// タイルキャッシュの使用状況
//		RingQueue<int>	m_FreeSlots;		// 空きスロットの番号リスト

		uint16			m_Front;
		uint16			m_FreeList[ N-1 ];// [0, 65534]. 65535は無効インデックス値

	};



	// 指定スロットのデータを「しばらくアクセスがない=汚れていない」状態にする -> 複数指定したいなら Variadic argments か initializer_listにする
	inline bool ObjectPool::SetClean( int slot )
	{
		ASSERT( slot < m_Capacity );

		m_BlockStatus[ slot ].dirty	= false;
		return true;
	}



	// 指定スロットのデータを「最近アクセスした=触って垢がついた」状態にする. -> 複数指定したいなら Variadic argments か initializer_listにする
	inline bool ObjectPool::SetDirty( int slot )
	{
		ASSERT( slot < m_Capacity );

		BlockStatus& status = m_BlockStatus[ slot ];

		if( status.reserved == false )
			return false;
		status.dirty = true;

		return true;
	}



	// キャッシュの未使用領域を探して予約する
	inline int ObjectPool::ReserveSlot()
	{	
		if( IsFull() ) //if( m_FreeSlots.IsEmpty() )
			return -1;	// フリースロットが空の場合は処理中止

		//=========== 空きスロットのIDを取得する ==============//
		//int freeSlot	= m_FreeSlots.Dequeue();

// Reserve index and invalidate next link
auto freeSlot = m_Front;
m_Front = m_FreeList[ m_Front ];
m_FreeList[ freeSlot ] = INVALID;

		//=========== 空きスロットを使用状態にする ===========//
		m_BlockStatus[ freeSlot ].reserved	= true;
		m_BlockStatus[ freeSlot ].dirty		= true;

		//======= 使用中スロット数をインクリメントする =======//
		++m_numReservedSlots;

		return freeSlot;
	}



	// キャッシュの未使用領域を指定数探して予約する。戻り値は先頭スロット(予約失敗した場合は-1を返す)
	inline bool ObjectPool::ReserveSlots( Memory<int>& slots )
	{	
		auto numslots = slots.Length();

		if(	IsFull() || //m_FreeSlots.IsEmpty() ||	// No free slots available 
			( numslots > m_Capacity - m_numReservedSlots ) )// Requested number of slots are too lagre
			return false;

		for( auto& slot : slots )
			slot = ReserveSlot();

		return true;
	}



	// 指定スロットを解放し、未使用状態にする
	inline bool ObjectPool::FreeSlot( int tile_id )
	{
		if( tile_id >= m_Capacity )
			return false;

		//========== 指定スロットを空き状態にする ============//
		m_BlockStatus[ tile_id ].reserved	= false; // タイル領域を予約する
		m_BlockStatus[ tile_id ].dirty		= false;// アップロード直後はアクセス痕跡を残す

		//========== 空きスロットリストに登録する ============//
		//m_FreeSlots.Enqueue( tile_id );

// Insert tile_id at at the top of free list
m_FreeList[ tile_id ] = m_Front;// connect tile_id's next to current front index
m_Front = tile_id;// overwrite m_Front with tile_id


		//========= 使用中スロット数をデクリメントする =======//
		--m_numReservedSlots;






		return true;
	}



	// 連続確保したスロット群を解放し、未使用状態にする
	inline bool ObjectPool::FreeSlots( Memory<int>& slots )
	{
		auto result = true;

		for( auto& slot : slots )
		{
			result &= FreeSlot( slot );
			slot = -1;
		}

		return result;
	}



	// 全スロットを未使用状態にする
	inline void ObjectPool::Clear()
	{
		//m_FreeSlots.Clear();
		m_Front = 0;
		for( uint32 i=0; i<m_Capacity; ++i )
		{
			m_FreeList[i] = i+1;
		}


		for( int i=0; i<m_Capacity; ++i )
		{
			m_BlockStatus[i].reserved	= false;
			m_BlockStatus[i].dirty		= false;

			//m_FreeSlots.Enqueue(i);
		}

		m_numReservedSlots	= 0;

	}


}// end of namespace



#endif	// OBJECT_POOL_H //