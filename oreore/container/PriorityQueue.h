#ifndef PRIORITY_QUEUE_H
#define	PRIORITY_QUEUE_H


#include	"../common/TString.h"
#include	"../memory/Memory.h"



namespace OreOreLib
{


	template< typename T, typename IndexType = typename MemSizeType >
	class PriorityQueue
	{
	public:

		PriorityQueue();
		PriorityQueue( IndexType max_size );
		virtual ~PriorityQueue();

		void Init( IndexType max_size );
		void Clear();
		void Release();

		void Extend( IndexType numelms );
		bool Shrink( IndexType numelms );

		void Enqueue( T elm );
		T Dequeue();

		IndexType next( IndexType index ) const	{ return ( index + 1 ) % m_HeapArray.Length(); }
		bool IsFull() const						{ return m_ActiveSize>=m_HeapArray.Length(); }
		bool IsEmpty() const					{ return m_ActiveSize==0; }

		void Display();



	private:
		
		IndexType					m_ActiveSize;
		MemoryBase<T, IndexType>	m_HeapArray;

		void TrickleUp( IndexType index );		// 下にある小さな値の要素を上に上げる
		void TrickleDown( IndexType index );		// 上にある大きな値の要素を下に沈める

	};




	template< typename T,  typename  IndexType >
	PriorityQueue<T, IndexType>::PriorityQueue()
		: m_ActiveSize( 0 )
	{
	}



	template< typename T,  typename  IndexType >
	PriorityQueue<T, IndexType>::PriorityQueue( IndexType max_size )
		: m_ActiveSize( 0 )
		, m_HeapArray( max_size )
	{
	}



	template< typename T,  typename  IndexType >
	PriorityQueue<T, IndexType>::~PriorityQueue()
	{
		Release();
	}



	template< typename T,  typename  IndexType >
	void PriorityQueue<T, IndexType>::Init( IndexType max_size )
	{
		Release();
		
		m_ActiveSize	= 0;
		m_HeapArray.Init( max_size );
	}



	template< typename T,  typename  IndexType >
	void PriorityQueue<T, IndexType>::Clear()
	{
		m_ActiveSize	= 0;
	}



	template< typename T,  typename  IndexType >
	void PriorityQueue<T, IndexType>::Release()
	{
		m_HeapArray.Release();
		m_ActiveSize	= 0;
	}



	template< typename T,  typename  IndexType >
	void PriorityQueue<T, IndexType>::Extend( IndexType numelms )
	{
		m_HeapArray.Extend( numelms );
	}



	template< typename T,  typename  IndexType >
	bool PriorityQueue<T, IndexType>::Shrink( IndexType numelms )
	{
		auto new_length = m_HeapArray.Length() - Min( m_HeapArray.Length(), numelms );
		if( new_length < m_ActiveSize )// 縮小可能な下限値を割り込んだ場合は中止.
			return false;

		m_HeapArray.Shrink( numelms );

		return true;
	}



	// heapに要素を1つ追加する
	template< typename T,  typename  IndexType >
	void PriorityQueue<T, IndexType>::Enqueue( T elm )
	{
		if( IsFull() )
		{
#ifdef _DEBUG
			tcout << "cannot enqueue. queue is full" << tendl;
#endif
			return;
		}

		// 配列の最後尾に新しい値を追加する
		m_HeapArray[ m_ActiveSize ]	= elm;
		m_ActiveSize++;

		// 上に上げていく
		TrickleUp( m_ActiveSize-1 );
	}

	

	// 先頭ノードを削除する.要素が入っていることが前提
	template< typename T,  typename  IndexType >
	T PriorityQueue<T, IndexType>::Dequeue()
	{
		T root = 0;

		if( IsEmpty() )
		{
#ifdef _DEBUG
			tcout << "cannot enqueue. queue is empty" << tendl;
#endif
			return root;// INF
		}

		// 先頭ノードをセーブ
		root = m_HeapArray[0];

		// 末尾要素を先頭に移動
		m_ActiveSize--;
		m_HeapArray[0] = m_HeapArray[ m_ActiveSize ];
	
		// 先頭に移動した末尾要素を下へ下げていく
		TrickleDown(0);

		return root;
	}
	


	// 下にある小さな値の要素を上に上げる
	template< typename T,  typename  IndexType >
	void PriorityQueue<T, IndexType>::TrickleUp( IndexType index )
	{
		IndexType parent = (index-1) / 2;
		T bottom = m_HeapArray[index];

		while( index > 0 && m_HeapArray[parent] > bottom )
		{
			m_HeapArray[index] = m_HeapArray[parent];// ノードを下へ移動
			index = parent;// インデックスを上へ移動
			parent = (parent-1) / 2;
		}// end of while

		m_HeapArray[index] = bottom;
	}



	// 上にある大きな値を下に沈める
	template< typename T,  typename  IndexType >
	void PriorityQueue<T, IndexType>::TrickleDown( IndexType index )
	{
		if( m_ActiveSize <= 1 )	return;

		IndexType smallerChild;
		T top = m_HeapArray[index];// 移動対象の要素を一時退避する

		while( index < m_ActiveSize/2 )
		{
			IndexType leftChild = 2*index+1;
			IndexType rightChild = leftChild+1;
		
			// より小さい子を見つける
			if( rightChild < m_ActiveSize && m_HeapArray[leftChild] > m_HeapArray[rightChild] )
			{
				smallerChild = rightChild;
			}
			else
			{
				smallerChild = leftChild;
			}

			// topがlargerChildよりも小さければ終了
			if( top <= m_HeapArray[smallerChild] )
				break;

			// 子を上へシフトする
			m_HeapArray[index] = m_HeapArray[smallerChild];
			// 下へ行く
			index = smallerChild;
	
		}// end of while

		m_HeapArray[index] = top;// 一時退避した要素を目的位置に入れる
	}



	template< typename T,  typename  IndexType >
	void PriorityQueue<T, IndexType>::Display()
	{
		tcout << typeid(*this).name() << "[" << m_ActiveSize << "]" << tendl;
		for( IndexType i=0; i<m_ActiveSize; ++i )
			tcout << "[" << i << "]: " << m_HeapArray[i] << tendl;
	}




}// end of namespace



#endif // !PRIORITY_QUEUE_H
