#ifndef RING_QUEUE_H
#define	RING_QUEUE_H


#include	"../common/TString.h"
#include	"Array.h"
#include	"../memory/Memory.h"



namespace OreOreLib
{


	template< typename T, typename IndexType = MemSizeType  >
	class RingQueue
	{
	public:

		RingQueue();
		RingQueue( IndexType max_size );
		virtual ~RingQueue();

		void Init( IndexType max_size );
		void Clear();
		void Release();

		void Extend( IndexType numelms );
		bool Shrink( IndexType numelms );

		//void Enqueue( T elm );
		void Enqueue( const T& elm );
		void Enqueue( T&& elm );

		T Dequeue();// 要素をコピーして返す
		void Dequeue( T& elm );// 要素を委譲して返す

		template < typename  ...Args >
		void Emplace( Args&&... args );

		IndexType next( IndexType index ) const { return ( index + 1 ) % m_Queue.Length(); }
		bool IsFull() const { return m_ActiveSize >= m_Queue.Length(); }
		bool IsEmpty() const { return m_ActiveSize==0; }

		operator bool() const { return m_Queue; }

		void Display();


	private:
		
		IndexType					m_ActiveSize;
		ArrayImpl<T, IndexType>		m_Queue;//MemoryBase<T, IndexType>	m_Queue;//
		IndexType					front;// キュー先頭のオブジェクトが入っている要素のインデックス
		IndexType					rear;// キュー最後尾の、オブジェクトを登録可能な空要素のインデックス

	};




	template< typename T, typename IndexType >
	RingQueue<T, IndexType>::RingQueue()
		: m_ActiveSize( 0 )
		, front( ~0u )
		, rear( ~0u )
	{
	}



	template< typename T, typename IndexType >
	RingQueue<T, IndexType>::RingQueue( IndexType max_size )
		: m_ActiveSize( 0 )
		, m_Queue( max_size )
		, front( 0 )
		, rear( 0 )
	{
	}



	template< typename T, typename IndexType >
	RingQueue<T, IndexType>::~RingQueue()
	{
		Release();
	}



	template< typename T, typename IndexType >
	void RingQueue<T, IndexType>::Init( IndexType max_size )
	{
		Release();
		
		m_ActiveSize	= 0;
		m_Queue.Init( max_size );
		front			= 0;
		rear			= 0;
	}



	template< typename T, typename IndexType >
	void RingQueue<T, IndexType>::Clear()
	{
		front			= 0;
		rear			= 0;
		m_ActiveSize	= 0;
	}



	template< typename T, typename IndexType >
	void RingQueue<T, IndexType>::Release()
	{
		m_Queue.Release();
		m_ActiveSize	= 0;
		front			= ~0u;
		rear			= ~0u;
	}



	/*
    If( front < rear ):
    |-----+++++++++++++++++++---------------|
          ^front=5           ^rear=24
	Just extend array without changing front/rear position
    |-----+++++++++++++++++++---------------===============|
          ^front=5           ^rear=24

    ElseIf( rear < front ):
    |++++++---------------------++++++++++++|
           ^rear=6              ^front=27
    Right shift [ front : m_Length ] elements
    |++++++---------------------===============++++++++++++|
           ^rear=6                             ^front=42
	*/
	template< typename T, typename IndexType >
	void RingQueue<T, IndexType>::Extend( IndexType numelms )
	{
		m_Queue.Resize( m_Queue.Length() + numelms );

		if( rear < front )
		{
			m_Queue.RightShiftElements( front, m_ActiveSize - rear, numelms );

			front += numelms;
		}
	}



	/*
    If( front < rear ):
    |-----+++++++++++++++++++---------------|
          ^front=5           ^rear=24
    Left shift [ front, rear ]
	|+++++++++++++++++++----------xxxxxxxxxx|
     ^front=0           ^rear=19   numelms=10

    Else If( rear < front ):
    |++++++---------------------++++++++++++|
           ^rear=6              ^front=27
    Left shift [front : -1]
	|++++++-----------++++++++++++xxxxxxxxxx|
           ^rear=6    ^front=17    numelms=10
	*/
	template< typename T, typename IndexType >
	bool RingQueue<T, IndexType>::Shrink( IndexType numelms )
	{
		auto new_length = m_Queue.Length() - Min( numelms, m_Queue.Length() );
		if( new_length < m_ActiveSize )// 縮小可能な下限値を割り込んだ場合は中止.
			return false;
		
		if( front <= rear )
		{
			m_Queue.LeftShiftElements( front, m_ActiveSize, front );

			rear -= front;
			front = 0;
		}
		else
		{
			m_Queue.LeftShiftElements( front, m_Queue.Length()-front, numelms );

			front -= numelms;
		}

		m_Queue.Resize( new_length );


		return true;
	}



	template< typename T, typename IndexType >
	void RingQueue<T, IndexType>::Enqueue( const T& elm )
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



	template< typename T, typename IndexType >
	void RingQueue<T, IndexType>::Enqueue( T&& elm )
	{
		if( IsFull() )
		{
#ifdef _DEBUG
			tcout << "cannot enqueue. queue is full" << tendl;
#endif
			return;
		}

		m_Queue[rear]	= (T&&)elm;
		rear = next( rear );

		m_ActiveSize++;
	}



	template< typename T, typename IndexType >
	T RingQueue<T, IndexType>::Dequeue()
	{
		if( IsEmpty() )
		{
#ifdef _DEBUG
			tcout << "cannot enqueue. queue is empty" << tendl;
#endif
			return T();//elm;// INF
		}

		T elm	= (T&&)m_Queue[front];
		front	= next( front );
		m_ActiveSize--;

		return elm;
	}



	template< typename T, typename IndexType >
	void RingQueue<T, IndexType>::Dequeue( T& elm )
	{
		if( IsEmpty() )
		{
#ifdef _DEBUG
			tcout << "cannot enqueue. queue is empty" << tendl;
#endif
			return;
		}

		elm		= (T&&)m_Queue[front];
		front	= next( front );
		m_ActiveSize--;
	}



	template< typename T, typename IndexType >
	template < typename  ...Args >
	void RingQueue<T, IndexType>::Emplace( Args&&... args )
	{
		if( IsFull() )
		{
#ifdef _DEBUG
			tcout << "cannot enqueue. queue is full" << tendl;
#endif
			return;
		}

		m_Queue[rear]	= T(args...);//elm;
		rear = next( rear );

		m_ActiveSize++;
	}



	template< typename T, typename IndexType >
	void RingQueue<T, IndexType>::Display()
	{
		tcout << typeid(*this).name() << "[" << front << ", " << rear << "]" << tendl;

		for( IndexType i=0; i<m_ActiveSize; ++i )
		{
			IndexType idx = (front + i) % m_Queue.Length();
			tcout << "[" << idx << "]: " << m_Queue[idx] << tendl;
			if((idx+1)% m_Queue.Length()==rear)	break;
		}
	}


}// end of namespace



#endif // !RING_QUEUE_H
