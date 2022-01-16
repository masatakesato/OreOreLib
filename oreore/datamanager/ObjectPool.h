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


	template < typename T, uint16 CAPACITY, typename = std::enable_if_t< ( CAPACITY < MaxLimit::uInt16 ) > >
	class ObjectPool
	{
		using SizeType = uint16;
		static const SizeType INVALID = CAPACITY;//(SizeType)~0u;

	public:

		// Default constructor
		ObjectPool()
			: m_Front( 0 )
			#ifdef _DEBUG
			, m_NumReservedSlots( 0 )
			#endif
		{
			for( SizeType i=0; i<CAPACITY; ++i )
			{
				m_FreeList[i] = i+1;
				m_Pool[i] = T();
			}
		}


		// Destructor
		~ObjectPool()
		{
			//tcout << _T("TileCacheTexture::~TileCacheTexture()...") << tendl;
		}


		ObjectPool( const ObjectPool& ) = delete;



		void Clear()
		{
			for( auto& data : m_Pool )
				data.~T();

			for( SizeType i=0; i<CAPACITY; ++i )
				m_FreeList[i] = i+1;
			m_Front = 0;

			#ifdef _DEBUG
			m_NumReservedSlots = 0;
			#endif
		}


		SizeType Capacity() const
		{
			return CAPACITY;
		}


		#ifdef _DEBUG
		SizeType NumReservedSlots() const
		{
			return m_NumReservedSlots;
		}
		#endif


		bool IsUsedup() const
		{
			return m_Front == CAPACITY;//m_NumReservedSlots >= CAPACITY;
		}


		T* const Reserve()
		{	
			if( IsUsedup() )
				return nullptr;	// フリースロットが空の場合は処理中止

			//=========== 空きスロットのIDを取得する ==============//
			// Reserve index and invalidate next link
			auto freeSlot = m_Front;
			m_Front = m_FreeList[ m_Front ];
			m_FreeList[ freeSlot ] = INVALID;

			//======= 使用中スロット数をインクリメントする =======//
			#ifdef _DEBUG
			++m_NumReservedSlots;
			#endif

			return &m_Pool[ freeSlot ];
		}


		// Free slot
		bool Free( T* ptr )
		{
			ASSERT( ptr >= m_Pool && ptr < CAPACITY + m_Pool );

			//========== Insert slot_id at at the top of free list ============//
			auto slot_id = static_cast<SizeType>( ptr - m_Pool );
			m_FreeList[ slot_id ] = m_Front;// connect slot_id's next to current front index
			m_Front = slot_id;// overwrite m_Front with slot_id

			//========= 使用中スロット数をデクリメントする =======//
			#ifdef _DEBUG
			--m_NumReservedSlots;
			#endif


			return true;
		}



	private:

		T			m_Pool[ CAPACITY ];

		SizeType	m_FreeList[ CAPACITY ];// [0, 65534]. 65535は無効インデックス値
		SizeType	m_Front;

		#ifdef _DEBUG
		SizeType	m_NumReservedSlots;	// 使用中スロットの数
		#endif

	};













}// end of namespace



#endif	// OBJECT_POOL_H //