#ifndef BOUNDARY_TAG_BLOCK_H
#define	BOUNDARY_TAG_BLOCK_H


#include	"../common/Utility.h"



namespace OreOreLib
{

	class BoundaryTagBlock
	{
	public:

		BoundaryTagBlock();
		BoundaryTagBlock( uint8* data, uint32 size, bool isfree );
		~BoundaryTagBlock();
		
		// defined as inline methods
		BoundaryTagBlock* Register( BoundaryTagBlock* pnext );
		BoundaryTagBlock* Remove();
		BoundaryTagBlock* RemovePrevious();
		BoundaryTagBlock* RemoveNext();
		bool HasPrevious() const;
		bool HasNext() const;
		BoundaryTagBlock* Previous() const;
		BoundaryTagBlock* Next() const;
		uint8* const Data() const;
		uint32 DataSize() const;// data size
		uint32 TotalSize() const;// size of all tags + data.
		void Resize( uint32 size );
		void SetDataPointer( uint8* data );
		void Clear();
		void Reserve();
		void SetFree( bool flag );
		bool IsFree();

		void Info();

		//###################### TODO: experimental. 2018.11.14 #####################//
		BoundaryTagBlock* Transfer( int mem_offset );


	private:

		// this        m_pData       m_pEnd
		//  | [m_DataSize] | [ data ] |
		bool	m_IsFree : 1;	// used/unused flag bit field
		uint32	m_DataSize;	// data size in uint32

		uint8	*m_pData;	// pointer to data
		uint32	*m_pEnd;	// pointer to end tag


		BoundaryTagBlock *prev, *next;
	};


	const size_t c_BBHeaderSize = sizeof( BoundaryTagBlock );// header only size (excepts footer size)
	const size_t c_BoundaryTagAllocSize = c_BBHeaderSize + sizeof( uint32 );// header + footer size



	inline BoundaryTagBlock* BoundaryTagBlock::Register( BoundaryTagBlock* pnext )
	{
		// connect current next and pnext
		pnext->next	= this->next;
		this->next->prev = pnext;

		// connect self and pnext
		this->next	= pnext;
		pnext->prev	= this;

		return this;
	}

	inline BoundaryTagBlock* BoundaryTagBlock::Remove()
	{
		next->prev	= this->prev;
		prev->next	= this->next;

		this->next	= this;
		this->prev	= this;

		return this;
	}

	inline BoundaryTagBlock* BoundaryTagBlock::RemovePrevious()
	{
		BoundaryTagBlock *pprev = prev;

		// conect self and new prev
		this->prev = pprev->prev;
		this->prev->next = this;

		// isolate pprev
		pprev->next	= pprev;
		pprev->prev	= pprev;

		return pprev;
	}

	inline BoundaryTagBlock* BoundaryTagBlock::RemoveNext()
	{
		BoundaryTagBlock *pnext = next;

		// connect self and new next
		this->next = pnext->next;
		this->next->prev = this;

		// isolate pnext
		pnext->next = pnext;
		pnext->prev = pnext;

		return pnext;
	}




	inline bool BoundaryTagBlock::HasPrevious() const
	{
		return this != prev;
	}

	inline bool BoundaryTagBlock::HasNext() const
	{
		return this != next;
	}

	inline BoundaryTagBlock* BoundaryTagBlock::Previous() const
	{
		return next;
	}

	inline BoundaryTagBlock* BoundaryTagBlock::Next() const
	{
		return prev;
	}

	inline uint8* const BoundaryTagBlock::Data() const
	{
		return m_pData;
	}

	inline uint32 BoundaryTagBlock::DataSize() const// data size
	{
		return m_DataSize;
	}

	inline uint32 BoundaryTagBlock::TotalSize() const// size of all tags + data.
	{
		return *m_pEnd;
	}

	inline void BoundaryTagBlock::Resize( uint32 size )
	{
		m_DataSize	= size;
		m_pEnd	= (uint32 *)( m_pData + size );// set end tag address.
		*m_pEnd	= c_BoundaryTagAllocSize + size;// set reserved memory amount ( boundary tag obj + datasize + end tag size)
	}

	inline void BoundaryTagBlock::SetDataPointer( uint8* data )
	{
		m_pData	= data;
		m_pEnd	= (uint32 *)( m_pData + m_DataSize );// set end tag address.
		*m_pEnd	= c_BoundaryTagAllocSize + m_DataSize;// set reserved memory amount ( boundary tag obj + datasize + end tag size)
	}

	inline void BoundaryTagBlock::Clear()
	{
		memset( m_pData, 0, m_DataSize );
	}

	inline void BoundaryTagBlock::Reserve()
	{
		m_IsFree = false;
	}

	inline void BoundaryTagBlock::SetFree( bool flag )
	{
		m_IsFree = flag;
	}

	inline bool BoundaryTagBlock::IsFree()
	{
		return m_IsFree;
	}


	//###################### TODO: experimental. 2018.11.14 #####################//

	// TODO: 元々のメモリ空間のベースアドレスが分からないと、prevとnextの変更ができない.けどちょっと待てよ...
	// TODO: prev, nextはFreeListに登録されたブロックの探索に使う
	// Resizeで領域縮小する場合はFreeList空っぽ状態で行う→prev,nextに何が入ってても関係ない
	// Resizeで領域拡大する場合は? 

	inline BoundaryTagBlock* BoundaryTagBlock::Transfer( int mem_offset )
	{
		uint8* mem_base = (uint8*)this + mem_offset;
		BoundaryTagBlock* pBB = new( mem_base ) BoundaryTagBlock( mem_base + c_BBHeaderSize, m_DataSize, m_IsFree );
		pBB->next = (BoundaryTagBlock*)( (uint8*)next + mem_offset );
		pBB->prev = (BoundaryTagBlock*)( (uint8*)prev + mem_offset );

		return pBB;
	}


}// end of namespace


#endif // !BOUNDARY_TAG_BLOCK_H
