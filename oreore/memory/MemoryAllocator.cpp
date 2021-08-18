#include	"MemoryAllocator.h"


#include	"../mathlib/MathLib.h"
#include	"../common/TString.h"



namespace OreOreLib
{
	static const uint32 N = 2;


	MemoryAllocator::MemoryAllocator()
		: m_AllocSize( 0 )
		, m_DataSize( 0 )
		, m_pData( nullptr )
		, m_FreeListLength( 0 )
	{

	}


	MemoryAllocator::MemoryAllocator( uint32 allocSize )
		: m_pData( nullptr )
	{
		Init( allocSize );
	}


	MemoryAllocator::~MemoryAllocator()
	{
		Release();
	}


	void MemoryAllocator::Init( uint32 allocSize )
	{
		Release();

		// TODO: 確保するメモリ総容量を計算する. allocSize + BoundaryTagヘッダー/フッター領域(sizeof(BoundaryTag)+sizeof(uint32)) * 最大分割数
		m_DataSize	= allocSize;
		m_AllocSize	= m_DataSize + c_BoundaryTagAllocSize;

		m_pData	= new uint8[ m_AllocSize ];
		//memset( m_pData, 0, m_AllocSize );
		//std::fill_n( m_pData, allocSize, 0 );

		//=============== Initialize FreeList =================//		
		m_FreeListLength = 1;


		//=============== Register entire memory to free list ================//
		BoundaryTagBlock* newblock	= new( m_pData ) BoundaryTagBlock( m_pData + c_BBHeaderSize, m_DataSize, true );

		m_FreeList.Register( newblock );
	}


	void MemoryAllocator::Release()
	{
		m_AllocSize				= 0;
		m_DataSize				= 0;
		SafeDeleteArray( m_pData );
		m_FreeListLength		= 0;
	}


	void MemoryAllocator::Clear()
	{
		//=============== Initialize FreeList =================//
		ClearFreeList();
		
		//=============== Register entire memory to free list ================//
		BoundaryTagBlock* newblock	= new( m_pData ) BoundaryTagBlock( m_pData + c_BBHeaderSize, m_DataSize, true );
		AddtoFreeList( newblock );
	}


	// TODO: m_pDataを参照しているポインタ変数も更新が必要. 2018.11.22
	void MemoryAllocator::Resize( uint32 allocSize )
	{
		tcout << _T( "MemoryAllocator::Resize.\n" );

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

	void* MemoryAllocator::Allocate( size_t size )
	{
		//tcout << "//============== MemoryAllocator::Reserve... ==============//\n";

		//================= Remove BoundaryTagBlock from m_FreeList. ================//
		BoundaryTagBlock* pBB = GetFreeBoundaryTag( size );
		if( !pBB )	return nullptr;

		pBB->Remove();


		//	pBBサイズが分割可能な容量を保持している(分割後でも管理タグサイズ+1[uint8]以上残る)場合は、pBBの一部を切り出してnewBBを作成する
		//		
		//		         | <---------------------- pBB->TotalSize() -------------------------->|
		//		 before: |- header -|--------------------- Data --------------------|- footer -|
		//		
		//		         |<------   postPbbTotalSize   ------>|<------   requiredSize   ------>|
		//		 after:  |- header -|---- Data ----|- footer -|- header -|-- Data --|- footer -|
		//		                                            newBBPos

		uint32 requiredSize = size + c_BoundaryTagAllocSize;
		uint32 postPbbTotalSize = pBB->TotalSize() - requiredSize;// 0: 要求サイズがpBBと一致. 0<c_BoundaryTagAllocSize: pBBサイズ不足で分割できない

		if( postPbbTotalSize <= c_BoundaryTagAllocSize )// pBBを分割する必要がない(orできない)場合はpBBのデータ領域先頭ポインタを返して終了
		{
			pBB->SetFree( false );
			return pBB->Data();
		}

		// まずpBBの占有サイズを縮小してメモリ空間を空ける
		pBB->Resize( postPbbTotalSize - c_BoundaryTagAllocSize );// 
		AddtoFreeList( pBB );// Register remaining pBB to freelist

		// 空いたメモリ空間に新たなBoundaryTagBlockを生成する
		uint8* newBBPos = (uint8*)pBB + postPbbTotalSize;
		BoundaryTagBlock *newBB = new( newBBPos ) BoundaryTagBlock( newBBPos + c_BBHeaderSize, size, false );
		
		// rerurn pointer to reserved memory.
		return newBB->Data();
	}




	void* MemoryAllocator::AlignedAllocate( size_t size, size_t alignment )
	{
		//tcout << "//============== MemoryAllocator::AlignedAllocate... ==============//\n";

		// Remove BoundaryTagBlock from m_FreeList
		BoundaryTagBlock* pBB = GetFreeBoundaryTag( RoundUp( size + c_BoundaryTagAllocSize, alignment ) );
		
		if( !pBB )	return nullptr;

		pBB->Remove();


		uint8* ptrEnd = (uint8*)pBB + pBB->TotalSize();
		uint8* dataEnd = ptrEnd - sizeof uint32;

		uint8* alignedDataStart = (uint8*)Round( size_t(dataEnd - size), alignment );
		size_t alignedDataSize = size_t(dataEnd - alignedDataStart);
		size_t alignedRequiredSize = size_t( ptrEnd - alignedDataStart - c_BBHeaderSize );


	
		//		         | <---------------------- pBB->TotalSize() -------------------------->|
		//		 before: |- header -|--------------------- Data --------------------|- footer -|
		//		
		//
		//		         |<------   postPbbTotalSize   ------>|<------   alignedRequiredSize   ------>|
		//		 after:  |- header -|---- Data ----|- footer -|- header -|-- Data --|- footer -|
		//		                                            newBBPos

		uint32 postPbbTotalSize = pBB->TotalSize() - alignedRequiredSize;// 0: 要求サイズがpBBと一致. 0<c_BoundaryTagAllocSize: pBBサイズ不足で分割できない

		if( postPbbTotalSize <= c_BoundaryTagAllocSize )// pBBを分割する必要がない(orできない)場合はpBBのデータ領域先頭ポインタを返して終了
		{
TTODO: アラインメントなしでポインタ返す???
			pBB->SetFree( false );
			return pBB->Data();
		}

		// まずpBBの占有サイズを縮小してメモリ空間を空ける
		pBB->Resize( postPbbTotalSize - c_BoundaryTagAllocSize );// 
		AddtoFreeList( pBB );// Register remaining pBB to freelist

		// 空いたメモリ空間に新たなBoundaryTagBlockを生成する
		uint8* newBBPos = (uint8*)pBB + postPbbTotalSize;
		BoundaryTagBlock *newBB = new( newBBPos ) BoundaryTagBlock( newBBPos + c_BBHeaderSize, alignedDataSize, false );
		

		//tcout << "------- pBB: --------\n";
		//pBB->Info();
		//tcout << "------- NewPB: --------\n";
		//newBB->Info();
		//tcout << tendl;


		// rerurn pointer to reserved memory.
		return newBB->Data();
	}









	void MemoryAllocator::Free( uint8* data )
	{
		//tcout << _T( "MemoryAllocator::Free...\n" );

		BoundaryTagBlock *pBBcurr	= (BoundaryTagBlock *)( data - c_BBHeaderSize );

		// pBBcurrがメモリプールの先頭/後続かどうかはどうやって判断する? -> m_pDataの範囲に収まっているかどうか
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



	uint32 MemoryAllocator::Compact()
	{
		//tcout << _T( "//========== MemoryAllocator::Compact... ==========//\n" );

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



	void MemoryAllocator::Info()
	{
		tcout << _T( "//=== MemoryAllocator::Info... ===//\n" );

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