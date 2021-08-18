#include	"TLSF.h"


#include	"../mathlib/MathLib.h"



namespace OreOreLib
{
	static const uint32 N = 2;


	TLSF::TLSF()
		: m_SplitFactor( N )
		, m_NumSLI( 0 )
	 	, m_AllocSize( 0 )
		, m_DataSize( 0 )
		, m_pData( nullptr )
		, m_FreeListLength( 0 )
		, m_FreeList( nullptr )
		, m_FreeSlots( nullptr )
		, m_SLIFreeSlotsLength( 0 )
		, m_SLIFreeSlots( nullptr )
	{

	}


	TLSF::TLSF( uint32 allocSize/*, uint32 splitFactor*/ )
		: m_SplitFactor( N )
		, m_FreeList( nullptr )
		, m_FreeSlots( nullptr )
		, m_SLIFreeSlots( nullptr )
		, m_pData( nullptr )
	{
		Init( allocSize/*, splitFactor*/ );
	}


	TLSF::~TLSF()
	{
		Release();
	}


	void TLSF::Init( uint32 allocSize/*, uint32 splitFactor*/ )
	{
		//tcout << "//============== TLSF::Init... ==============//\n";

		// TODO: プール可能なメモリ容量の下限値がある
		//  - 第2カテゴリ分割数以上のバイト数は必須
		//  - 

		Release();

		//m_SplitFactor = splitFactor;
		m_NumSLI = (uint32)pow( (uint32)2, m_SplitFactor );
		// TODO: 確保するメモリ総容量を計算する. allocSize + BoundaryTagヘッダー/フッター領域(sizeof(BoundaryTag)+sizeof(uint32)) * 最大分割数
		m_DataSize	= Max( (uint32)pow( 2, m_SplitFactor ), allocSize );
		m_AllocSize	= m_DataSize + c_BoundaryTagAllocSize;

		m_pData	= new uint8[ m_AllocSize ];
		//memset( m_pData, 0, m_AllocSize );
		//std::fill_n( m_pData, allocSize, 0 );

		//=============== Initialize FreeList =================//
		uint32 fli = GetMSB( m_AllocSize );
		uint32 sli = GetSLI( m_AllocSize, fli );
		uint32 freeListID = GetFreeListIndex( fli, sli );
		
		m_FreeListLength = freeListID + 1;
		m_FreeList		= new BoundaryTagBlock[m_FreeListLength];
		m_FreeSlots		= new bool[m_FreeListLength];
		std::fill_n( m_FreeSlots, m_FreeListLength, false );//memset( m_FreeSlots, 0, sizeof( bool ) * m_FreeListLength );//

		//tcout << "Num of 2nd Level Split: 2^" << m_SplitFactor << tendl;
		//tcout << "Requested memory size: " << allocSize << tendl;
		//tcout << "Actual memory pool size(includes boundary tag space): " << m_AllocSize << tendl;
		//tcout << "First Level Index: " << fli << tendl;
		//tcout << "Second Level Index: " << sli << tendl;
		//tcout << "Free List Length: " << freeListLength << tendl;
		
		m_FLIFreeSlots	= 0;
		m_SLIFreeSlotsLength = fli + 1;
		m_SLIFreeSlots	= new uint8[m_SLIFreeSlotsLength];//new uint32[m_SLIFreeSlotsLength];//new uint16[m_SLIFreeSlotsLength];//
		std::fill_n( m_SLIFreeSlots, m_SLIFreeSlotsLength, 0 );//memset( m_SLIFreeSlots, 0, sizeof( uint8 ) * ( m_SLIFreeSlotsLength ) );//


		//=============== Register entire memory to free list ================//
		BoundaryTagBlock* newblock	= new( m_pData ) BoundaryTagBlock( m_pData + c_BBHeaderSize, m_DataSize, true );

		m_FreeList[freeListID].Register( newblock );
		
		// Set Free slot flag
		m_FreeSlots[freeListID]	= true;
		m_FLIFreeSlots			|= ( 1 << fli );
		m_SLIFreeSlots[fli]		|= ( 1 << sli );
	}


	void TLSF::Release()
	{
		//m_SplitFactor			= 0;
		m_NumSLI				= 0;
		m_AllocSize				= 0;
		m_DataSize				= 0;
		SafeDeleteArray( m_pData );
		m_FreeListLength		= 0;
		SafeDeleteArray( m_FreeList );
		SafeDeleteArray( m_FreeSlots );
		m_FLIFreeSlots			= 0;
		m_SLIFreeSlotsLength	= 0;
		SafeDeleteArray( m_SLIFreeSlots );
	}


	void TLSF::Clear()
	{
		//=============== Initialize FreeList =================//
		ClearFreeList();
		
		//=============== Register entire memory to free list ================//
		BoundaryTagBlock* newblock	= new( m_pData ) BoundaryTagBlock( m_pData + c_BBHeaderSize, m_DataSize, true );
		AddtoFreeList( newblock );
	}


	// TODO: m_pDataを参照しているポインタ変数も更新が必要. 2018.11.22
	void TLSF::Resize( uint32 allocSize )
	{
		tcout << _T( "TLSF::Resize.\n" );

		if( allocSize == m_AllocSize )
			return;

		if( allocSize < m_AllocSize )
		{
			allocSize = Max( Compact(), allocSize );
		}
		
		uint8* newData = new uint8[ allocSize ];
		
		std::copy( m_pData, m_pData + Min( m_AllocSize, allocSize ), newData );
	}


	//###########################################################################//
	//
	// before: |- header -|-------------------- Data ---------------------|- footer -|
	//
	//                                                          <- size ->
	// after:  |- header -|---- Data ----|- footer -|- header -|-- Data --|- footer -|
	//
	// 実際には、size + header + footer[バイト]よりも大きい空き領域が必要
	uint8* TLSF::Allocate( uint32 size )
	{
		//tcout << "//============== TLSF::Reserve... ==============//\n";

		uint32 requiredSize = size + c_BoundaryTagAllocSize;

		uint32 fli = GetMSB( requiredSize );
		uint32 sli = GetSLI( requiredSize, fli );
		uint32 freeListID = GetFreeListIndex( fli, sli );

		if( m_FreeSlots[freeListID]==false )// fli/sliに該当する空き領域が見つかった場合は開き領域を探索する
		{
			// 同一第1段階カテゴリ内で利用可能な未使用ブロックを探す
			sli = SearchFreeSLI( fli, sli );
			
			if( sli==-1 )// それでも見つからなかったら、更に容量が大きい第1段階カテゴリのフリーリストを検出する
			{
				fli = SearchFreeFLI( fli );
				if( fli==-1 )	return nullptr;
				sli = SearchFreeSLI( fli, 0 );
				if( sli==-1 )	return nullptr;// それでもなお見つからない -> メモリ空き領域不足
			}

			freeListID	= GetFreeListIndex( fli, sli );
		}

		//tcout << "First Level Index: " << fli << tendl;
		//tcout << "Second Level Index: " << sli << tendl;
		//tcout << "Free List Index: " << freeListID << tendl;

		//================= Remove BoundaryTagBlock from m_FreeList. ================//
		BoundaryTagBlock* pBB = m_FreeList[freeListID].Next();
		pBB->Remove();
		
		// Update m_FreeSlots, FLIFreeSlots, and SLIFreeSlot status.
		if( m_FreeList[freeListID].HasNext()==false )//( m_FreeList[freeListID].next == &m_FreeList[freeListID] ) //
		{
			// Set Freeslot[ (fli, sli)] to false
			m_FreeSlots[freeListID]	= false;
			// Set SLIFreeSlot to false
			m_SLIFreeSlots[fli] &= ~( 1 << sli );
			// Set FLIFleeSlot[fli] to false only if all sli slots are empty.
			if( m_SLIFreeSlots[fli] == 0 )
				m_FLIFreeSlots &= ~( 1 << fli );
		}

//	pBBサイズが分割可能な容量を保持している(分割後でも管理タグサイズ+1[uint8]以上残る)場合は、pBBの一部を切り出してnewBBを作成する
//		
//		         | <---------------------- pBB->TotalSize() -------------------------->|
//		 before: |- header -|--------------------- Data --------------------|- footer -|
//		
//		         |<------   postPbbTotalSize   ------>|<------   requiredSize   ------>|
//		 after:  |- header -|---- Data ----|- footer -|- header -|-- Data --|- footer -|
//		                                            newBBPos

		uint32 postPbbTotalSize = pBB->TotalSize() - requiredSize;// 0: 要求サイズがpBBと一致. 0<c_BoundaryTagAllocSize: pBBサイズ不足で分割できない

		if( postPbbTotalSize <= c_BoundaryTagAllocSize )// pBBを分割する必要がない(orできない)場合はpBBのデータ領域先頭ポインタを返して終了
		{
			pBB->SetFree( false );
			//pBB->Clear();
			return pBB->Data();
		}

		// まずpBBの占有サイズを縮小してメモリ空間を空ける
		pBB->Resize( postPbbTotalSize - c_BoundaryTagAllocSize );// 
		AddtoFreeList( pBB );// Register remaining pBB to freelist

		// 空いたメモリ空間に新たなBoundaryTagBlockを生成する
		uint8* newBBPos = (uint8*)pBB + postPbbTotalSize;
		BoundaryTagBlock *newBB = new( newBBPos ) BoundaryTagBlock( newBBPos + c_BBHeaderSize, size, false );
		
		
		// メモリ空間をクリアする
		//pBB->Clear();
		//newBB->Clear();

		// rerurn pointer to reserved memory.
		return newBB->Data();
	}


	void TLSF::Free( uint8* data )
	{
		//tcout << _T( "TLSF::Release...\n" );

		BoundaryTagBlock *pBBcurr	= (BoundaryTagBlock *)( data - c_BBHeaderSize );

		// TODO: pBBcurrがメモリプールの先頭/後続かどうかはどうやって判断する? -> m_pDataの範囲に収まっているかどうか
		// pBBがメモリプール先頭にある場合:　pBBcurr == m_pData:
		// pBBがメモリプール末尾にある場合: (uint8*)pBBcurr + pBBCurr->TotalSize() == m_pData + m_AllocSize
		
		//======== 先行するBoundaryTagBlock (pBBprev)が空き領域の場合はマージする. ========//
		if( (uint8*)pBBcurr > m_pData )
		{
			BoundaryTagBlock *pBBprev = (BoundaryTagBlock *)( (uint8*)pBBcurr - *(uint32*)( (uint8*)pBBcurr - sizeof( uint32 ) ) );
			if( pBBprev->IsFree() )
			{
				RemovefromFreeList( pBBprev );
				pBBprev->SetFree( false );
				
				// pBBprevにpBBcurrをマージする
				pBBprev->Resize( pBBprev->DataSize() + pBBcurr->TotalSize() );
				// pBBcurrのアドレス参照位置をpBBprevに入れ替える
				pBBcurr	= pBBprev;
			}
		}

		//======== 後続するBoundaryTagBlock (pBBnext)が空き領域の場合はマージする. ========//
		if( (uint8*)pBBcurr + pBBcurr->TotalSize() < m_pData + m_AllocSize )
		{
			BoundaryTagBlock *pBBnext = (BoundaryTagBlock *)( (uint8*)pBBcurr + pBBcurr->TotalSize() );
			if( pBBnext->IsFree() )
			{
				RemovefromFreeList( pBBnext );
				pBBnext->SetFree( false );
				
				// pBBcurrにpBBnextをマージする
				pBBcurr->Resize( pBBcurr->DataSize() + pBBnext->TotalSize() );
			}
		}

		//============ Register pBBcurr to FreeList =========//
		AddtoFreeList( pBBcurr );
		pBBcurr->SetFree( true );
		
		data	= nullptr;
		//tcout << __T( "    Released memory block: " ) << pBBcurr->m_DataSize << _T( " [bytes]\n" );
	}



	uint32 TLSF::Compact()
	{
		//tcout << _T( "//========== TLSF::Compact... ==========//\n" );

		BoundaryTagBlock *pBBcurr = (BoundaryTagBlock *)m_pData;
		uint8 *pFree = (uint8*)pBBcurr;
		uint8 *pNext = nullptr;

		ClearFreeList();

		while( true )
		{
			pNext = (uint8*)pBBcurr + pBBcurr->TotalSize();

			if( pBBcurr->IsFree() == false )// 使用中BoundaryTagBlockが見つかった場合
			{
				if( pFree < (uint8*)pBBcurr )// メモリ空間前方に空き領域がある場合はBoundaryTagBlockを移動する
				{
					//tcout << _T( "Free space found at [" ) << pFree << ", " << pFree + pBBcurr->TotalSize() << "]\n";
					memmove( pFree, pBBcurr, pBBcurr->TotalSize() );//std::copy( (uint8*)pBBcurr, (uint8*)pBBcurr + pBBcurr->TotalSize(), pFree );//

					BoundaryTagBlock* bb = (BoundaryTagBlock*)pFree;
					bb->SetDataPointer( pFree + c_BBHeaderSize );
				}

				pFree += pBBcurr->TotalSize();
				//tcout << _T( "Reserved block found. Moving pFree to " ) << pFree << tendl;
			}
			//else// 未使用BoundaryTagBlockが見つかった場合
			//{
			//	//Unregister( pBBcurr );
			//	//tcout << _T( "Free block found. pFree fixed at = " ) << pFree << tendl;
			//}
			
			if( pNext >= m_pData + m_AllocSize )
				break;

			pBBcurr = (BoundaryTagBlock *)pNext;

			//tcout << tendl;
		}
		
		// Initialize remaining memory space and register to freelist.
		BoundaryTagBlock* pBBend = new( pFree ) BoundaryTagBlock( pFree + c_BBHeaderSize, uint32(m_pData + m_AllocSize - pFree - c_BoundaryTagAllocSize), true );
		AddtoFreeList( pBBend );

		return m_AllocSize - pBBend->TotalSize();
	}



	void TLSF::Info()
	{
		tcout << _T( "//=== TLSF::Info... ===//\n" );

		BoundaryTagBlock *pBB = (BoundaryTagBlock *)m_pData;

		while( true )
		{
			pBB->Info();

			if( (uint8*)pBB + pBB->TotalSize() >= m_pData + m_AllocSize )
				break;

			pBB = (BoundaryTagBlock *)( (uint8*)pBB + pBB->TotalSize() );
		}

	}







}// end of namespace