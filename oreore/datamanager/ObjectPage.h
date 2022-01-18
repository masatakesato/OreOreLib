#ifndef	OBJECT_PAGE_H
#define	OBJECT_PAGE_H

#include	<windows.h>
#include	<process.h>

//#include	<oreore/mathlib/GraphicsMath.h>
//#include	<oreore/container/RingQueue.h>
#include	<oreore/memory/Memory.h>



namespace OreOreLib
{

	//##################################################################################//
	//																					//
	//									ObjectPage										//
	//																					//
	//##################################################################################//


	template < typename T, uint16 CAPACITY, typename = std::enable_if_t< ( CAPACITY < MaxLimit::uInt16 ) > >
	class ObjectPage
	{
		using SizeType = uint16;
		static const SizeType INVALID = CAPACITY;

	public:

		enum PageStates{ Clean, Dirty, Usedup, NumPageStates };

		// Default constructor
		ObjectPage()
			: m_Front( 0 )
			#ifdef _DEBUG
			, m_NumBlocksInUse( 0 )
			#endif
		{
			for( SizeType i=0; i<CAPACITY; ++i )
			{
				m_FreeList[i] = i+1;
				m_Pool[i] = T();
			}
		}


		// Destructor
		~ObjectPage()
		{
			Clear();
		}


		ObjectPage( const ObjectPage& ) = delete;



		void Clear()
		{
			for( auto& data : m_Pool )
				data.~T();

			for( SizeType i=0; i<CAPACITY; ++i )
				m_FreeList[i] = i+1;
			m_Front = 0;

			#ifdef _DEBUG
			m_NumBlocksInUse = 0;
			#endif
		}


		T* const Reserve()
		{	
			// bread if no avaiable blocks
			if( IsUsedup() )
				return nullptr;

			// Reserve free block id and invalidate next link
			auto reserved_block_id = m_Front;
			m_Front = m_FreeList[ m_Front ];
			m_FreeList[ reserved_block_id ] = INVALID;

			// Increment in-use block count
			#ifdef _DEBUG
			++m_NumBlocksInUse;
			#endif

			return &m_Pool[ reserved_block_id ];
		}


		// Free block
		bool Free( T* ptr )
		{
			ASSERT( ptr >= m_Pool && ptr < CAPACITY + m_Pool );

			// Insert block_id at at the top of free list
			auto block_id = static_cast<SizeType>( ptr - m_Pool );
			m_FreeList[ block_id ] = m_Front;// connect block_id's next to current front index
			m_Front = block_id;// overwrite m_Front with block_id

			// Decrement in-use block count
			#ifdef _DEBUG
			--m_NumBlocksInUse;
			#endif

			return true;
		}



		SizeType Capacity() const
		{
			return CAPACITY;
		}


		#ifdef _DEBUG
		SizeType NumBlocksInUse() const
		{
			return m_NumBlocksInUse;
		}
		#endif


		bool IsUsedup() const
		{
			return m_Front >= CAPACITY;
		}


		PageStates PageState() const
		{
			return ( m_NumBlocksInUse != 0 ) << uint8( m_NumBlocksInUse>=CAPACITY );
			// Clean: 00
			// Dirty: 01
			// Usedup:10
		}



	private:

		T			m_Pool[ CAPACITY ];

		SizeType	m_FreeList[ CAPACITY ];// [0, 65534]. 65535は無効インデックス値
		SizeType	m_Front;

		#ifdef _DEBUG
		SizeType	m_NumBlocksInUse;	// number of blocks in use
		#endif

	};













}// end of namespace



#endif	// OBJECT_PAGE_H //