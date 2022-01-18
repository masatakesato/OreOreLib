#ifndef	OBJECT_POOL_H
#define	OBJECT_POOL_H

#include	<oreore/memory/PoolAllocator.h>

#include	"ObjectPage.h"



namespace OreOreLib
{

	//##################################################################################//
	//																					//
	//									ObjectPool										//
	//																					//
	//##################################################################################//


//TODO: ページあたりのブロック数はどうやって指定する？
// 決め打ちでいい

	template < typename T, uint16 PageBloks >
	class ObjectPool
	{
		using ObjPage = ObjectPage<T, PageBloks >;

		template < typename T, uint16 CAPACITY >
		struct ObjPageNode
		{
			ObjectPool< T, CAPACITY > pool;//uint8 data[1];//
			ObjPageNode* next = nullptr;
			static const size_t HeaderSize = sizeof(next);// next byte length

TODO:  nextだけで実装可能か検討する. 2022.01.18

			void ConnectAfter( Page* pnode )
			{
				if( !IsAlone() )	return;
			
				// update connection of this node
				prev = pnode;
				next = pnode->next;

				// update connection of pnode
				pnode->next = this;

				// update connection of next node
				if( next )	next->prev = this;
			}


			void ConnectBefore( Page* pnode )
			{
				if( !IsAlone() )	return;
			
				// update connection of this node
				prev = pnode->prev;
				next = pnode;

				// update connection of pnode
				pnode->prev = this;

				// update connection of next node
				if( prev )	prev->next = this;
			}


			void Disconnect()
			{
				if( prev )	prev->next = next;
				if( next )	next->prev = prev;
				next = nullptr;
				prev = nullptr;
			}


			void DisconnectPrev()
			{
				if( prev )	prev->next = nullptr;
				prev = nullptr;
			}


			void DisconnectNext()
			{
				if( prev )	prev->next = nullptr;
				prev = nullptr;
			}


			bool IsAlone() const
			{
				return next && prev ? false : true;
			}

		};


	public:

		ObjectPool()
			: m_pAllocator( nullptr )

			, m_Nil{ &m_Nil, 0x00 }
			, m_CleanFront( &m_Nil )
			, m_DirtyFront( &m_Nil )
			, m_UsedupFront( &m_Nil )
		{

		}


		~ObjectPool()
		{

		}


		void Cleanup()
		{
			tcout << _T( "ObjectPool::Cleanup()...\n" );

			ObjPageNode* page = m_CleanFront;
			while( page != &m_Nil )
			{
				ObjPageNode* nextpage = page->next;
				void* base = OSAllocator::GetAllocationBase( page );					
				if( base == feed )
				{
					tcout << _T( "      " ) << (unsigned*)page << tendl;
					if( page == m_CleanFront )	m_CleanFront = nextpage;
					page->Disconnect();
				}
				page = nextpage;
			}

			//while( m_CleanFront->next != m_Nil )
			//{
			//}

		}


		T* const Allocate()
		{
			//==================== メモリブロック取得前の準備 ====================//
			// DirtyListが空の場合
			if( IsEmpty( m_DirtyFront ) )
			{
				// Extend clean pages if empty
				if( IsEmpty( m_CleanFront ) )
				{
					tcout << _T( "m_CleanFront is empty. Allocating new Clean page....\n" );
					AllocatePages( /*m_CommitBatchSize*/1 );

					//ASSERT( IsEmpty( m_CleanFront )==false );
					if( IsEmpty( m_CleanFront ) )
						return nullptr;
				}

				tcout << _T( "m_DirtyFront is empty. Acquiring a clean page from m_CleanFront...\n" );
				m_DirtyFront = m_CleanFront;
				m_CleanFront = m_CleanFront->next;
			}


			ObjPageNode* pagetag	= m_DirtyFront;
			ObjPage* page = pagetag->data;

			T* ptr = page->Reserve();

			if( page->IsUsedup() )// pageがUsedupになった場合
			{
				// Update m_DirtyFront
				if( m_DirtyFront->next != m_CleanFront )
					m_DirtyFront = m_DirtyFront->next;
				else
					m_DirtyFront = &m_Nil;

				// Update m_UsedupFront if necessary
				if( IsEmpty(m_UsedupFront) )
					m_UsedupFront = page;
			}


			return ptr;
		}



// TODO: オブジェクトを解放する
//		ObjectPageを見つける
//		ObjectPageのブロックをフリーリストに加える

		bool Free( void*& ptr )
		{
			if( !ptr )
				return false;

			tcout << _T( "ObjectPool::Free()..." ) << ptr << tendl;
		
			// Get ObjPage from ptr
			ObjPage* page = m_pAllocator->GetBlockBase( ptr );
		
			// return false if page not found. 
			if( !page )
				return false;

			// Clean: 0, Dirty: 1, Usedup: 2
			auto stateBefore = page->PageState();//PageStates stateBefore = GetPageState( page );
			FreeBlock( page, ptr );
			ptr = nullptr;
			auto stateAfter = page->PageState();//PageStates stateAfter = GetPageState( page );


			if( stateBefore==ObjectPage::Usedup && stateAfter==ObjectPage::Dirty )
			{
				tcout << _T( "  Usedup -> Dirty...\n" );

				ObjPageNode* pPivot = IsEmpty( m_DirtyFront ) ? m_CleanFront : m_DirtyFront;

				if( page->next != pPivot )
				{
					if( page == m_UsedupFront ) m_UsedupFront = page->next;// page is the first usedup element 
					page->Disconnect();
					page->ConnectBefore( pPivot );
				}
				else if( page == m_UsedupFront )// page is the only remaining usedup, and pPivot's neighbor.
				{
					m_UsedupFront = &m_Nil;
				}

				m_DirtyFront = page;
			}

			else if( stateBefore==ObjectPage::Dirty && stateAfter==ObjectPage::Clean )
			{
				tcout << _T( "  Dirty -> Clean...\n" );	

				if( page->next != m_CleanFront )
				{
					if( page == m_DirtyFront ) m_DirtyFront = page->next;
					page->Disconnect();
					page->ConnectBefore( m_CleanFront );
				}
				else if( page == m_DirtyFront )// page is the only remaining Dirty, and m_CleanFront's neighbor.
				{
					m_DirtyFront = &m_Nil;
				}

				m_CleanFront = page;
			}

			else if( stateBefore==ObjectPage::Usedup && stateAfter==ObjectPage::Clean )// Occurs only when Page contains single block.
			{
				tcout << _T( "  Usedup -> Clean...\n" );

				if( page->next != m_CleanFront )
				{
					if( page == m_UsedupFront )	m_UsedupFront = page->next;
					page->Disconnect();
					page->ConnectBefore( m_CleanFront );
				}
				else if( page == m_UsedupFront )// page is the only remaining Usedup, and m_CleanFront's neighbor.
				{
					m_UsedupFront = &m_Nil;
				}

				m_CleanFront	= page;
			}

			else if( stateBefore==ObjectPage::Dirty && stateAfter==ObjectPage::Dirty )// Do nothing.
			{
				tcout << _T( "  Dirty -> Dirty...\n" );
			}

			tcout << tendl;

			return true;
		}


		void ReleaseAllPages()
		{
			while( m_Nil.next )
			{
				ObjPageNode* next = m_Nil.next;
				m_Nil.DisconnectNext();
				((ObjPage*)next->data)->~ObjPage();

				m_pAllocator->Free( next );
			}

			m_Nil.next = m_CleanFront = m_DirtyFront = m_UsedupFront = &m_Nil;
		}



	private:

		int32 m_NumActiveBlocks;
		size_t m_PoolSize;
		PoolAllocator*	m_pAllocator;


		//enum PageStates{ Clean, Dirty, Usedup, NumPageStates };
		ObjPageNode	m_Nil;
		ObjPageNode*	m_CleanFront;
		ObjPageNode*	m_DirtyFront;
		ObjPageNode*	m_UsedupFront;


		// Private helper functions

		bool IsEmpty( const ObjPageNode* p ) const
		{
			return p==&m_Nil;
		}


		ObjPage* GetObjPage( const void* ptr )
		{
			// ptrから先頭ブロックへ戻る
			ObjPageNode* page = m_pAllocator->GetBlockBase( ptr );
			return ((ObjPage*)page->data);
		}


		void AllocatePages( uint32 numPages )
		{
			ObjPageNode* newPage = nullptr;

			for( uint32 i=0; i<numPages; ++i )
			{
				newPage = static_cast<ObjPageNode*>( m_pAllocator->Allocate() );

				// Deploy as "Clean" page.
				newPage->ConnectBefore( m_CleanFront );
				m_CleanFront = newPage;
			}// end of i loop
		}


	};



}// end of namespace



#endif	// OBJECT_POOL_H //