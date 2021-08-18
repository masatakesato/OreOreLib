#ifndef	LINKED_LIST_H
#define	LINKED_LIST_H


#include	"../common/Utility.h"


// https://www.podoliaka.org/2013/01/20/iterators-en/

namespace OreOreLib
{

	//######################################################################//
	//								List Node								//
	//######################################################################//

	template< typename T >
	struct ListNode
	{
		ListNode *next = nullptr;
		ListNode *prev = nullptr;

		T data;


		//ListNode()
		//	: next( nullptr )
		//	, prev( nullptr )
		//{
		//	tcout << _T( "ListNode default constructor...\n" );
		//}


		//// Constructor with data copy
		//ListNode( const T& data, ListNode* next=nullptr, ListNode* prev=nullptr )
		//	: data( data )
		//	, next( next )
		//	, prev( prev )
		//{
		//	tcout << _T( "ListNode constructor with data copy...\n" );
		//}


		//// Constructor with data move
		//ListNode( T&& data, ListNode* next=nullptr, ListNode* prev=nullptr )
		//	: data( std::forward<T>(data) )
		//	, next( next )
		//	, prev( prev )
		//{
		//	tcout << _T( "ListNode constructor with data move...\n" );
		//}


		void ConnectAfter( ListNode* pnode )
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


		void ConnectBefore( ListNode* pnode )
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



	//######################################################################//
	//						Iterator for LinkedList							//
	//######################################################################//

	template< typename T >
	class ListIterator
	{
	public:

		// Default constructor
		ListIterator()
			: m_pCurrent( nullptr )
		{
		}


		// Constructor
		ListIterator( ListNode<T>* pnode )
			: m_pCurrent( pnode )
		{
		}


		// Copy constructor
		ListIterator( const ListIterator<T>& obj )
			: m_pCurrent( obj.m_pCurrent )
		{
			tcout << "ListIterator copy constructor called...\n";
		}


		const ListNode<T>* node() const
		{
			return m_pCurrent;
		}


		ListIterator& operator++()
		{
			m_pCurrent = m_pCurrent->next;
			return *this;
		}


		ListIterator& operator--()
		{
			m_pCurrent = m_pCurrent->prev;
			return *this;
		}

		
		T& operator*()
		{
			return m_pCurrent->data;
		}


		const T& operator*() const
		{
			return m_pCurrent->data;
		}


		bool operator==( const ListIterator& it ) const
		{
			return m_pCurrent == it.node();
		}


		bool operator!=( const ListIterator& it ) const
		{
			return m_pCurrent != it.node();
		}



	private:

		ListNode<T>*	m_pCurrent;

	};





	//######################################################################//
	//								Linked List								//
	//######################################################################//

	template< typename T >
	class LinkedList
	{
		using Node = ListNode<T>;

	public:

		// Default constructor
		LinkedList()
			: m_NumElements( 0 )
			, m_pHead( nullptr )
			, m_pTail( nullptr )
		{
		}


		// Destructor
		~LinkedList()
		{
			Clear();
		}


		// Copy constructor
		LinkedList( const LinkedList& obj )
			: m_NumElements( 0 )
			, m_pHead( nullptr )
			, m_pTail( nullptr )
		{
			for( const auto& data : obj )
				PushBack( data );
		}


		// Move constructor
		LinkedList( LinkedList&& obj )
			: m_NumElements( obj.m_NumElements )
			, m_pHead( obj.m_pHead )
			, m_pTail( obj.m_pTail )
		{
			obj.m_NumElements	= 0;
			obj.m_pHead			= nullptr;
			obj.m_pTail			= nullptr;
		}


		// Copy assignment operator
		LinkedList& operator=( const LinkedList& obj )
		{
			if( this != &obj )
			{
				Clear();

				for( const auto& data : obj )
					PushBack( data );
			}

			return *this;
		}


		// Move assignment operator
		LinkedList& operator=( LinkedList&& obj )
		{
			if( this != &obj )
			{
				Clear();

				m_NumElements		= obj.m_NumElements;
				m_pHead				= obj.m_pHead;
				m_pTail				= obj.m_pTail;
		
				obj.m_NumElements	= 0;
				obj.m_pHead			= nullptr;
				obj.m_pTail			= nullptr;
			}

			return *this;
		}



		void PushFront()
		{
			Node *node = new Node;

			if( m_pHead )
			{
				node->next = m_pHead;
				node->next->prev = node;
			}
			else
			{
				m_pTail = node;
			}

			m_pHead = node;

			++m_NumElements;
		}


		void PushFront( const T& data )
		{
			PushFront();
			m_pHead->data = data;
		}


		void PushFront( T&& data )
		{
			PushFront();
			m_pHead->data = std::forward<T>(data);
		}


		void PushBack()
		{
			Node *node = new Node;

			if( m_pTail )// insert new node before tail
			{
				node->prev = m_pTail;
				m_pTail->next = node;
			}
			else
			{
				m_pHead = node;
			}

			m_pTail = node;

			++m_NumElements;
		}


		void PushBack( const T& data )
		{
			PushBack();
			m_pTail->data = data;
		}


		void PushBack( T&& data )
		{
			PushBack();
			m_pTail->data = std::forward<T>(data);
		}


		void PopFront()
		{
			if( !m_pHead )	return;

			if( m_pHead==m_pTail )
			{
				ReleaseNode( m_pHead );
				m_pHead = m_pTail = nullptr;
			}
			else
			{
				m_pHead = m_pHead->next;
				ReleaseNode( m_pHead->prev );
			}

			--m_NumElements;
		}


		void PopBack()
		{
			if( !m_pTail )	return;

			if( m_pTail==m_pHead )
			{
				ReleaseNode( m_pTail );
				m_pHead = m_pTail = nullptr;
			}
			else
			{
				m_pTail = m_pTail->prev;
				ReleaseNode( m_pTail->next );
			}

			--m_NumElements;
		}


		const T& Front() const
		{
			return m_pHead->data;
		}


		const T& Back() const
		{
			return	m_pTail->data;
		}


		void Resize( int numelms )
		{
			assert( numelms >= 0 );

			while( numelms > m_NumElements )
				PushBack();

			while( numelms < m_NumElements )
				PopBack();
		}


		void Resize( int numelms, const T& data )
		{
			assert( numelms >= 0 );

			while( numelms > m_NumElements )
			{
				PushBack();
				m_pTail->data = data;
			}

			while( numelms < m_NumElements )
			{
				PopBack();
			}
		}


		void Clear()
		{
			while( m_pHead )
			{
				Node* next = m_pHead->next;
				delete m_pHead;
				m_pHead = next;
			}
			m_pTail = nullptr;

			m_NumElements = 0;
		}


		int Length() const
		{
			return m_NumElements;
		}



		ListIterator<T> begin() const
		{
			return ListIterator<T>( m_pHead );
		}


		ListIterator<T> end() const
		{
			return ListIterator<T>( nullptr );
		}



		void Display() const
		{
			tcout << typeid(*this).name() << _T("[") << m_NumElements << "]" << tendl;

			int i=0;
			for( const auto& data : *this )
			{
				tcout << _T("[") << i++ << _T("]: ") << data << tendl;
			}
			//Node *curr = m_pHead;

			//for( int i=0; i<m_NumElements; ++i )
			//{
			//	tcout << curr->data << tendl;
			//	curr = curr->next;
			//}

		}



	private:

		int		m_NumElements;
		Node*	m_pHead;
		Node*	m_pTail;


		void ReleaseNode( Node* pnode )
		{
			if( pnode->prev )	pnode->prev->next = pnode->next;
			if( pnode->next )	pnode->next->prev = pnode->prev;
			SafeDelete( pnode );
		}

	};


}// end of namespace



#endif	// LINKED_LIST_H //