#include	<oreore/common/TString.h>
#include	<oreore/common/Utility.h>
#include	<oreore/mathlib/MathLib.h>



#define MAXELM 3


struct Page
{
	Page *next = nullptr;
	Page *prev = nullptr;
	int count=0;

	Page() : count(MAXELM)
	{
	}

	Page(int val) : count(val)
	{
	}


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


	bool IsAlone() const
	{
		return next && prev ? false : true;
	}

};



class PageList
{
	enum PageStates{ Clean, Dirty, Usedup, NumPageStates };


public:


	PageList()
	{
		m_Nil.next = m_Nil.prev = &m_Nil;
		m_CleanFront = m_DirtyFront = m_UsedupFront = &m_Nil;

	}



	void AddPage()
	{
		Page* newPage = new Page(MAXELM);

		newPage->ConnectBefore( &m_Nil );

		if( IsEmpty(m_CleanFront) )// .. -> m_CleanFront(newNode) -> m_Nil			
			m_CleanFront = newPage;
	}



	Page* Alloc()
	{

		if( IsEmpty(m_DirtyFront) )// m_DirtyFrontが空の場合は、m_CleanFrontから新たに取得する
		{
			if( !IsEmpty(m_CleanFront) )
			{
				m_DirtyFront = m_CleanFront;
				m_CleanFront = m_CleanFront->next;
			}
			else
			{
				return nullptr;
			}
		}
	

		// m_DirtyFrontの先頭からPageを取り出す
		Page* page = m_DirtyFront;
		page->count--;// カウントをデクリメント

		if( page->count <= 0 )// pageがUsedupになった場合
		{
			m_DirtyFront = m_DirtyFront->next;// m_DirtyFrontを後ろにずらす
			if( m_DirtyFront==m_CleanFront )	m_DirtyFront = &m_Nil;// m_DirtyFront残り要素がなくなった場合はnilに戻す

			if( IsEmpty(m_UsedupFront) )
				m_UsedupFront = page;
		}

		return page;

	}


	void Free_ver3( Page* page )
	{
		tcout << "//============== PageList::Free_ver3() =================//\n";

		// Clean: 0, Dirty: 1, Usedup: 2
		auto stateBefore = GetPageState( page );
		page->count++;// カウントをインクリメント
		auto stateAfter = GetPageState( page );


		if( stateBefore==Usedup && stateAfter==Dirty )
		{
			tcout << "  Usedup -> Dirty...\n";

			Page* pPivot = IsEmpty( m_DirtyFront ) ? m_CleanFront : m_DirtyFront;

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

		else if( stateBefore==Dirty && stateAfter==Clean )
		{
			tcout << "  Dirty -> Clean...\n";	

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

		else if( stateBefore==Usedup && stateAfter==Clean )// Occurs only when Page contains single block.
		{
			tcout << "  Usedup -> Clean...\n";

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

		else if( stateBefore==Dirty && stateAfter==Dirty )// Do nothing.
		{
			tcout << "  Dirty -> Dirty...\n";
		}



		tcout << tendl;
	}




	void Free_ver2( Page* page )
	{
		tcout << "//============== PageList::Free_ver2() =================//\n";

		// Clean: 0, Dirty: 1, Usedup: 2
		auto stateBefore = GetPageState( page );
		page->count++;// カウントをインクリメント
		auto stateAfter = GetPageState( page );


		if( stateBefore==Usedup && stateAfter==Dirty )
		{
			tcout << "  Usedup -> Dirty...\n";

			if( IsEmpty( m_DirtyFront ) )// m_DirtyFrontが存在しない場合
			{
				tcout << "    No Dirty exists...\n";

				if( page->next == m_CleanFront )
				{
					if( page == m_UsedupFront )	m_UsedupFront = &m_Nil;
					m_DirtyFront = page;
				}
				else
				{
					if( page == m_UsedupFront ) m_UsedupFront = page->next;
					page->Disconnect();
					page->ConnectBefore( m_CleanFront );
					m_DirtyFront = page;
				}
			}
			else// m_DirtyFrontが存在する場合
			{
				tcout << "    Dirty exists...\n";

				if( page->next == m_DirtyFront )
				{
					if( page == m_UsedupFront )	m_UsedupFront = &m_Nil;
					m_DirtyFront = page;
				}
				else
				{
					if( page == m_UsedupFront ) m_UsedupFront = page->next;
					page->Disconnect();
					page->ConnectBefore( m_DirtyFront );
					m_DirtyFront = page;
				}
			}
		}
		else if( stateBefore==Usedup && stateAfter==Clean )// Pageのブロック数が1個の場合にのみ発生.
		{
			tcout << "  Usedup -> Clean...\n";

			if( IsEmpty( m_CleanFront ) )// m_CleanFrontが存在しない場合
			{
				tcout << "    No Clean exists...\n";

				if( page->next == /*&m_Nil*/m_CleanFront )
				{
					if( page == m_UsedupFront )	m_UsedupFront = &m_Nil;
					m_CleanFront	= page;
				}
				else
				{
					if( page == m_UsedupFront )	m_UsedupFront = page->next;
					page->Disconnect();
					page->ConnectBefore( &m_Nil );
					m_CleanFront	= page;
				}
			}
			else// m_CleanFrontが存在する場合
			{
				tcout << "    Clean exists...\n";

				if( page->next == m_CleanFront )
				{
					if( page == m_UsedupFront )	m_UsedupFront = &m_Nil;
					m_CleanFront = page;
				}
				else
				{
					if( page == m_UsedupFront )	m_UsedupFront = page->next;
					page->Disconnect();
					page->ConnectBefore( m_CleanFront );
					m_CleanFront = page;
				}
			}
		}

		else if( stateBefore==Dirty && stateAfter==Dirty )
		{
		// TODO: test with MAXELM=3
			tcout << "  Dirty -> Dirty...\n";
			// 何もしなくていい
		}

		else if( stateBefore==Dirty && stateAfter==Clean )
		{
			tcout << "  Dirty -> Clean...\n";	

			if( IsEmpty( m_CleanFront ) )// m_CleanFrontが存在しない場合
			{
				tcout << "    No Clean exists...\n";

				if( page->next == /*&m_Nil*/m_CleanFront )
				{
					if( page == m_DirtyFront ) m_DirtyFront = &m_Nil;
					m_CleanFront = page;
				}
				else
				{
					if( page == m_DirtyFront ) m_DirtyFront = page->next;
					page->Disconnect();
					page->ConnectBefore( &m_Nil );
					m_CleanFront = page;
				}
			}
			else// m_CleanFrontが存在する場合
			{
				tcout << "    Clean exists...\n";

				if( page->next == m_CleanFront )
				{
					if( page == m_DirtyFront )	m_DirtyFront = &m_Nil;
					m_CleanFront = page;
				}
				else
				{
					if( page == m_DirtyFront ) m_DirtyFront = page->next;
					page->Disconnect();
					page->ConnectBefore( m_CleanFront );
					m_CleanFront = page;
				}
			}
		}

		tcout << tendl;
	}



	void Free_ver1( Page* page )
	{
		tcout << "//============== PageList::Free_ver1() =================//\n";

		// Clean: 0, Dirty: 1, Usedup: 2

		auto stateBefore = GetPageState( page );
		page->count++;// カウントをインクリメント
		auto stateAfter = GetPageState( page );


		if( stateBefore==Usedup && stateAfter==Dirty )
		{
			tcout << "  Usedup -> Dirty...\n";

			if( IsEmpty( m_DirtyFront ) )// m_DirtyFrontが存在しない場合
			{
				tcout << "    No Dirty exists...\n";

				if( m_UsedupFront->next == m_CleanFront )// m_UsedupFrontが1個だけで、m_CleanFrontと隣接している場合
				{
					tcout << "      unique element...\n";
					m_DirtyFront = m_UsedupFront;// m_UsedupFront位置をm_DirtyFrontにする
					m_UsedupFront = &m_Nil;// m_UsedupFrontは無効化する
				}
				else if( m_UsedupFront == page )// m_UsedupFrontの先頭要素を移動する場合
				{
					tcout << "      head element...\n";
					m_UsedupFront = m_UsedupFront->next;// m_UsedupFrontを後ろにずらす
					page->Disconnect();
					page->ConnectBefore( m_CleanFront );// m_CleanFrontの直前にpageを移動する
					m_DirtyFront = page;// m_DirtyFrontの位置をpageに設定する
				}
				else// m_UsedupFrontの先頭以外の要素を移動する場合
				{
					tcout << "      internal element...\n";
					if( page->next != m_CleanFront )// m_CleanFrontの直前にpageを移動する
					{
						page->Disconnect();
						page->ConnectBefore( m_CleanFront );
					}
					m_DirtyFront = page;
				}

			}
			else// m_DirtyFrontが存在する場合
			{
				tcout << "    Dirty exists...\n";

				if( m_UsedupFront->next == m_DirtyFront )// m_UsedupFrontが1個だけで、m_DirtyFrontと隣接している場合
				{
					tcout << "      unique element...\n";
					 m_DirtyFront = m_UsedupFront;//m_DirtyFrontの位置を一つ前にずらす
					 m_UsedupFront = &m_Nil;// m_UsedupFrontは無効化する
				}
				else if( m_UsedupFront == page )// m_UsedupFrontの先頭要素を移動する場合
				{
					tcout << "      head element...\n";
					m_UsedupFront = m_UsedupFront->next;
					page->Disconnect();
					page->ConnectAfter( m_DirtyFront );
				}
				else// m_UsedupFrontの先頭要素以外を移動する場合
				{
					tcout << "      internal element...\n";
					if( page->next != m_DirtyFront )
					{
						page->Disconnect();
						page->ConnectBefore( m_DirtyFront );
					}
					m_DirtyFront = page;
				}
			}

		}
		else if( stateBefore==Usedup && stateAfter==Clean )// Pageのブロック数が1個の場合にのみ発生.
		{
			tcout << "  Usedup -> Clean...\n";

			if( IsEmpty( m_CleanFront ) )// m_CleanFrontが存在しない場合
			{
				tcout << "    No Clean exists...\n";

				if( m_UsedupFront->next == m_DirtyFront )// m_UsedupFrontが1個だけで、m_DirtyFrontと隣接している場合
				{
				// ありえない. Usedup -> Cleanへの遷移は、Page要素が1個だけの場合のみ.
				// Page要素1個の場合はUsedUop/Cleanいずれかの状態しか取らない
					tcout << "      unique element...\n";
					//page->Disconnect();
					//page->ConnectBefore( &m_Nil );// 最後尾にpageを移動する
					// m_UsedupFront = &m_Nil;// m_UsedupFrontは無効化する
					// m_CleanFront = page;	// m_CleanFrontを新たに設定する
				}
				else if( m_UsedupFront == page )// m_UsedupFrontの先頭要素を移動する場合
				{
					tcout << "      head element...\n";
					m_UsedupFront = m_UsedupFront->next;
					page->Disconnect();
					page->ConnectBefore( &m_Nil );// 最後尾にpageを移動する
					m_CleanFront = page;	// m_CleanFrontを新たに設定する
				}
				else// m_UsedupFrontの先頭以外の要素を移動する場合
				{
					tcout << "      internal element...\n";
					if( page->next != /*m_CleanFront*/&m_Nil )
					{
						page->Disconnect();
						page->ConnectBefore( &m_Nil );// 最後尾にpageを移動する
					}
					m_CleanFront = page;	// m_CleanFrontを新たに設定する
				}

			}
			else// m_CleanFrontが存在する場合
			{
				tcout << "    Clean exists...\n";

				if( m_UsedupFront->next == m_CleanFront )// m_UsedupFrontが1個だけで、m_CleanFrontと隣接している場合
				{
					tcout << "      unique element...\n";
					 m_CleanFront = m_UsedupFront;//m_DirtyFrontの位置を一つ前にずらす
					 m_UsedupFront = &m_Nil;// m_UsedupFrontは無効化する
				}
				else if( m_UsedupFront == page )// m_UsedupFrontの先頭要素を移動する場合
				{
					tcout << "      head element...\n";
					m_UsedupFront = m_UsedupFront->next;
					page->Disconnect();
					page->ConnectAfter( m_CleanFront );
				}
				else// m_UsedupFrontの先頭要素以外を移動する場合
				{
					tcout << "      internal element...\n";
					if( page->next != m_CleanFront )
					{
						 page->Disconnect();
						 page->ConnectBefore( m_CleanFront );
					}
					m_CleanFront = page;
				}

			}

		}

		else if( stateBefore==Dirty && stateAfter==Dirty )
		{
		// TODO: test with MAXELM=3
			tcout << "  Dirty -> Dirty...\n";
			// 何もしなくていい
		}

		else if( stateBefore==Dirty && stateAfter==Clean )
		{
			tcout << "  Dirty -> Clean...\n";	

			if( IsEmpty( m_CleanFront ) )// m_CleanFrontが存在しない場合
			{
				tcout << "    No Clean exists...\n";

				if( m_DirtyFront->next == m_CleanFront )// m_DirtyFrontが1個だけで、m_CleanFrontと隣接している場合
				{
					tcout << "      unique element...\n";
					page->Disconnect();
					page->ConnectBefore( &m_Nil );// 最後尾にpageを移動する
					 m_DirtyFront = &m_Nil;// m_DirtyFrontは無効化する
					 m_CleanFront = page;	// m_CleanFrontを新たに設定する
				}
				else if( m_DirtyFront == page )// m_DirtyFrontの先頭要素を移動する場合
				{
					tcout << "      head element...\n";
					m_DirtyFront = m_DirtyFront->next;
					page->Disconnect();
					page->ConnectBefore( &m_Nil );// 最後尾にpageを移動する
					m_CleanFront = page;	// m_CleanFrontを新たに設定する
				}
				else// m_DirtyFrontの先頭以外の要素を移動する場合
				{
					tcout << "      internal element...\n";
					if( page->next != /*m_CleanFront*/&m_Nil )
					{
						page->Disconnect();
						page->ConnectBefore( &m_Nil );// 最後尾にpageを移動する
					}
					m_CleanFront = page;	// m_CleanFrontを新たに設定する
				}

			}
			else// m_CleanFrontが存在する場合
			{
				tcout << "    Clean exists...\n";

				if( m_DirtyFront->next == m_CleanFront )// m_DirtyFrontが1個だけで、m_CleanFrontと隣接している場合
				{
					tcout << "      unique element...\n";
					 m_CleanFront = m_DirtyFront;//m_DirtyFrontの位置を一つ前にずらす
					 m_DirtyFront = &m_Nil;// m_DirtyFrontは無効化する
				}
				else if( m_DirtyFront == page )// m_DirtyFrontの先頭要素を移動する場合
				{
					tcout << "      head element...\n";
					m_DirtyFront = m_DirtyFront->next;
					page->Disconnect();
					page->ConnectAfter( m_CleanFront );
				}
				else// m_DirtyFrontの先頭要素以外を移動する場合
				{
					tcout << "      internal element...\n";
					if( page->next != m_CleanFront )
					{
						page->Disconnect();
						page->ConnectBefore( m_CleanFront );
					}
					m_CleanFront = page;
				}

			}

		}

		tcout << tendl;

	}




	void Info()
	{
		tcout << "//=========== PageList ===========//\n";

		tcout << "UsedUp Pages...\n";
		for( Page* page=m_UsedupFront; page!=m_DirtyFront && page!=m_CleanFront && page!=&m_Nil; page=page->next )
		{
			tcout << "[" << (unsigned*)page << "]: " << page->count << "/" << MAXELM << tendl;
		}

		tcout << "Dirty Pages...\n";
		for( Page* page=m_DirtyFront; page!=m_CleanFront && page!=&m_Nil; page=page->next )
		{
			tcout << "[" << (unsigned*)page << "]: " << page->count << "/" << MAXELM << tendl;
		}

		tcout << "Clean Pages...\n";
		for( Page* page=m_CleanFront; page!=&m_Nil; page=page->next )
		{
			tcout << "[" << (unsigned*)page << "]: " << page->count << "/" << MAXELM << tendl;
		}

		tcout << tendl;

	}



	bool IsEmpty( Page* p ) const { return p==&m_Nil; }

	PageStates GetPageState( Page* p ) const { return PageStates( (p->count<MAXELM) << uint8(p->count<=0) ); }


	Page* m_UsedupFront = nullptr;
	Page* m_DirtyFront = nullptr;
	Page* m_CleanFront = nullptr;

	Page m_Nil;

};




int main()
{

	tcout << ((uint8)true << (uint8)true) << tendl;


	{
		tcout << "//############################ Free_ver1 test ###############################//\n\n";
		PageList pagelist;

		tcout << "//==================== Add 4 Pages =====================//\n\n";

		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.Info();

		tcout << tendl;

	
		tcout << "//====================== Alloc x 4 ========================//\n\n";

		Page* p1 = pagelist.Alloc();
		pagelist.Info();
	
		Page* p2 = pagelist.Alloc();
		pagelist.Info();
	
		Page* p3 = pagelist.Alloc();
		pagelist.Info();

		Page* p4 = pagelist.Alloc();
		pagelist.Info();

		Page* p5 = pagelist.Alloc();
		pagelist.Info();

		Page* p6 = pagelist.Alloc();
		pagelist.Info();

		tcout << tendl;

		tcout << "//====================== Free x 4 ========================//\n\n";

		pagelist.Free_ver1(p6);
		pagelist.Info();

		pagelist.Free_ver1(p4);
		pagelist.Info();

		pagelist.Free_ver1(p1);
		pagelist.Info();

		pagelist.Free_ver1(p2);
		pagelist.Info();

		pagelist.Free_ver1(p5);
		pagelist.Info();

		pagelist.Free_ver1(p3);
		pagelist.Info();

	}

	tcout << tendl;

	{
		tcout << "//############################ Free_ver2 test ###############################//\n\n";

		PageList pagelist;

		tcout << "//==================== Add 4 Pages =====================//\n\n";

		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.Info();

		tcout << tendl;

	
		tcout << "//====================== Alloc x 4 ========================//\n\n";

		Page* p1 = pagelist.Alloc();
		pagelist.Info();
	
		Page* p2 = pagelist.Alloc();
		pagelist.Info();
	
		Page* p3 = pagelist.Alloc();
		pagelist.Info();

		Page* p4 = pagelist.Alloc();
		pagelist.Info();

		Page* p5 = pagelist.Alloc();
		pagelist.Info();

		Page* p6 = pagelist.Alloc();
		pagelist.Info();

		tcout << tendl;

		tcout << "//====================== Free x 4 ========================//\n\n";

		pagelist.Free_ver2(p6);
		pagelist.Info();

		pagelist.Free_ver2(p4);
		pagelist.Info();

		pagelist.Free_ver2(p1);
		pagelist.Info();

		pagelist.Free_ver2(p2);
		pagelist.Info();

		pagelist.Free_ver2(p5);
		pagelist.Info();

		pagelist.Free_ver2(p3);
		pagelist.Info();

	}

	tcout << tendl;

	{
		tcout << "//############################ Free_ver3 test ###############################//\n\n";

		PageList pagelist;

		tcout << "//==================== Add 4 Pages =====================//\n\n";

		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.AddPage();
		pagelist.Info();

		tcout << tendl;

	
		tcout << "//====================== Alloc x 4 ========================//\n\n";

		Page* p1 = pagelist.Alloc();
		pagelist.Info();
	
		Page* p2 = pagelist.Alloc();
		pagelist.Info();
	
		Page* p3 = pagelist.Alloc();
		pagelist.Info();

		Page* p4 = pagelist.Alloc();
		pagelist.Info();

		Page* p5 = pagelist.Alloc();
		pagelist.Info();

		Page* p6 = pagelist.Alloc();
		pagelist.Info();

		tcout << tendl;

		tcout << "//====================== Free x 4 ========================//\n\n";

		pagelist.Free_ver3(p6);
		pagelist.Info();

		pagelist.Free_ver3(p4);
		pagelist.Info();

		pagelist.Free_ver3(p1);
		pagelist.Info();

		pagelist.Free_ver3(p2);
		pagelist.Info();

		pagelist.Free_ver3(p5);
		pagelist.Info();

		pagelist.Free_ver3(p3);
		pagelist.Info();

	}


	return 0;














	tcout << tendl;



	return 0;
}