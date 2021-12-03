#ifndef PRIORITY_QUEUE_H
#define	PRIORITY_QUEUE_H


#include	"../common/TString.h"
#include	"../memory/Memory.h"



namespace OreOreLib
{


	template< typename T >
	class PriorityQueue
	{
		using SizeType = typename MemSizeType;

	public:

		PriorityQueue();
		PriorityQueue( int max_size );
		virtual ~PriorityQueue();

		void Init( SizeType max_size );
		void Clear();
		void Release();

		void Extend( SizeType numelms );
		bool Shrink( SizeType numelms );

		void Enqueue( T elm );
		T Dequeue();

		int next( sizeType index ) const	{ return ( index + 1 ) % m_HeapArray.Length(); }
		bool IsFull() const					{ return m_ActiveSize>=m_HeapArray.Length(); }
		bool IsEmpty() const				{ return m_ActiveSize==0; }

		void Display();



	private:
		
		SizeType			m_ActiveSize;
		/*Array*/Memory<T>	m_HeapArray;

		void TrickleUp( SizeType index );		// 下にある小さな値の要素を上に上げる
		void TrickleDown( SizeType index );		// 上にある大きな値の要素を下に沈める

	};




	template< typename T >
	PriorityQueue<T>::PriorityQueue()
		: m_ActiveSize( 0 )
	{
	}



	template< typename T >
	PriorityQueue<T>::PriorityQueue( int max_size )
		: m_ActiveSize( 0 )
		, m_HeapArray( max_size )
	{
	}



	template< typename T >
	PriorityQueue<T>::~PriorityQueue()
	{
		Release();
	}



	template< typename T >
	void PriorityQueue<T>::Init( SizeType max_size )
	{
		Release();
		
		m_ActiveSize	= 0;
		m_HeapArray.Init( max_size );
	}



	template< typename T >
	void PriorityQueue<T>::Clear()
	{
		m_ActiveSize	= 0;
	}



	template< typename T >
	void PriorityQueue<T>::Release()
	{
		m_HeapArray.Release();
		m_ActiveSize	= 0;
	}



	template< typename T >
	void PriorityQueue<T>::Extend( SizeType numelms )
	{
		m_HeapArray.Extend( numelms );
	}



	template< typename T >
	bool PriorityQueue<T>::Shrink( SizeType numelms )
	{
		auto new_length = m_HeapArray.Length() - Min( m_HeapArray.Length(), numelms );
		if( new_length < m_ActiveSize )// 縮小可能な下限値を割り込んだ場合は中止.
			return false;

		m_HeapArray.Shrink( numelms );

		return true;
	}



	// heapに要素を1つ追加する
	template< typename T >
	void PriorityQueue<T>::Enqueue( T elm )
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
	template< typename T >
	T PriorityQueue<T>::Dequeue()
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
	template< typename T >
	void PriorityQueue<T>::TrickleUp( SizeType index )
	{
		int parent = (index-1) / 2;
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
	template< typename T >
	void PriorityQueue<T>::TrickleDown( SizeType index )
	{
		if( m_ActiveSize <= 1 )	return;

		int smallerChild;
		T top = m_HeapArray[index];// 移動対象の要素を一時退避する

		while( index < m_ActiveSize/2 )
		{
			SizeType leftChild = 2*index+1;
			SizeType rightChild = leftChild+1;
		
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



	template< typename T >
	void PriorityQueue<T>::Display()
	{
		tcout << typeid(*this).name() << "[" << m_ActiveSize << "]" << tendl;
		for( SizeType i=0; i<m_ActiveSize; ++i )
			tcout << "[" << i << "]: " << m_HeapArray[i] << tendl;
	}




}// end of namespace



#endif // !PRIORITY_QUEUE_H
