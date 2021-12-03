#ifndef RING_QUEUE_H
#define	RING_QUEUE_H


#include	"../common/TString.h"
#include	"../memory/Memory.h"



namespace OreOreLib
{


	template< typename T >
	class RingQueue
	{
		using SizeType = typename MemSizeType;

	public:

		RingQueue();
		RingQueue( SizeType max_size );
		virtual ~RingQueue();

		void Init( SizeType max_size );
		void Clear();
		void Release();

		void Extend( SizeType numelms );
		bool Shrink( SizeType numelms );

		void Enqueue( T elm );
		T Dequeue();

		int next( SizeType index ) const { return ( index + 1 ) % m_Queue.Length(); }
		bool IsFull() const { return m_ActiveSize >= m_Queue.Length(); }
		bool IsEmpty() const { return m_ActiveSize==0; }

		void Display();


	private:
		
		SizeType	m_ActiveSize;
		Memory<T>	m_Queue;
		SizeType	front;// キュー先頭のオブジェクトが入っている要素のインデックス
		SizeType	rear;// キュー最後尾の、オブジェクトを登録可能な空要素のインデックス

	};




	template< typename T >
	RingQueue<T>::RingQueue()
		: m_ActiveSize( 0 )
		, front( ~0u )
		, rear( ~0u )
	{
	}



	template< typename T >
	RingQueue<T>::RingQueue( SizeType max_size )
		: m_ActiveSize( 0 )
		, m_Queue( max_size )
		, front( 0 )
		, rear( 0 )
	{
	}



	template< typename T >
	RingQueue<T>::~RingQueue()
	{
		Release();
	}



	template< typename T >
	void RingQueue<T>::Init( SizeType max_size )
	{
		Release();
		
		m_ActiveSize	= 0;
		m_Queue.Init( max_size );
		front			= 0;
		rear			= 0;
	}



	template< typename T >
	void RingQueue<T>::Clear()
	{
		front			= 0;
		rear			= 0;
		m_ActiveSize	= 0;
	}



	template< typename T >
	void RingQueue<T>::Release()
	{
		m_Queue.Release();
		m_ActiveSize	= 0;
		front			= ~0u;
		rear			= ~0u;
	}



	template< typename T >
	void RingQueue<T>::Extend( SizeType numelms )
	{
		m_Queue.Extend( numelms );

		if( rear < front )
		{
			int32 newfront = front + static_cast<int32>( numelms );
			memmove( &m_Queue[newfront], &m_Queue[front], (m_ActiveSize - rear) * sizeof(T) );
			front = newfront;
		}
	}
	/*
    こういう場合は
    |-----+++++++++++++++++++---------------|
          ^front=5           ^rear=24
	こうする
    |-----+++++++++++++++++++---------------===============|
          ^front=5           ^rear=24

    こういう場合は
    |++++++---------------------++++++++++++|
           ^rear=6              ^front=27
    こうする
    |++++++---------------------===============++++++++++++|
           ^rear=6                             ^front=42
	*/


	template< typename T >
	bool RingQueue<T>::Shrink( SizeType numelms )
	{
		auto new_length = m_Queue.Length() - Min( numelms, m_Queue.Length() );
		if( new_length < m_ActiveSize )// 縮小可能な下限値を割り込んだ場合は中止.
			return false;
		
		if( front <= rear )
		{
			memmove( &m_Queue[0], &m_Queue[front], m_ActiveSize * sizeof(T) );// 使用中領域を配列先頭にスライドさせる
			// rear/frontのインデックスもスライドする
			rear -= front;
			front = 0;
		}
		else
		{
			int newfront = front - numelms;
			memmove( &m_Queue[newfront], &m_Queue[front], (m_Queue.Length()-front) * sizeof(T) );
			front = newfront;
		}

		m_Queue.Shrink( numelms );


		return true;
	}

	/*
    こういう場合は、、、、
    |-----+++++++++++++++++++---------------|
          ^front=5           ^rear=24
    こうする
	|+++++++++++++++++++----------xxxxxxxxxx|
     ^front=0           ^rear=19

    こういう場合は、、、、
    |++++++---------------------++++++++++++|
           ^rear=6              ^front=27
    こうする
	|++++++-----------++++++++++++xxxxxxxxxx|
           ^rear=6    ^front=17
	*/




	template< typename T >
	void RingQueue<T>::Enqueue( T elm )
	{
		if( IsFull() )
		{
#ifdef _DEBUG
			tcout << "cannot enqueue. queue is full" << tendl;
#endif
			return;
		}

		m_Queue[rear]	= elm;
		rear = next( rear );

		m_ActiveSize++;
	}



	template< typename T >
	T RingQueue<T>::Dequeue()
	{
		T elm = 0;

		if( IsEmpty() )
		{
#ifdef _DEBUG
			tcout << "cannot enqueue. queue is empty" << tendl;
#endif
			return elm;// INF
		}

		elm		= m_Queue[front];
		front	= next( front );
		m_ActiveSize--;

		return elm;
	}



	template< typename T >
	void RingQueue<T>::Display()
	{
		tcout << typeid(*this).name() << "[" << front << ", " << rear << "]" << tendl;

		for( SizeType i=0; i<m_ActiveSize; ++i )
		{
			SizeType idx = (front + i) % m_Queue.Length();
			tcout << "[" << idx << "]: " << m_Queue[idx] << tendl;
			if((idx+1)% m_Queue.Length()==rear)	break;
		}
	}


}// end of namespace



#endif // !RING_QUEUE_H
