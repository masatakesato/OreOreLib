﻿#ifndef SET_H
#define	SET_H

#include	<exception>

#include	"../common/HashCode.h"
#include	"../memory/Memory.h"



// https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1134/materials/cppdoc/hashmap-h.html
// https://aozturk.medium.com/simple-hash-map-hash-m_pTable-implementation-in-c-931965904250
// https://github.com/aozturk/HashMap



namespace OreOreLib
{

	//######################################################################//
	//																		//
	//								SetNode									//
	//																		//
	//######################################################################//

	template < typename T >
	struct SetNode
	{
		T value;
		SetNode* next = nullptr;


		SetNode( const T& value )
			: value( value )
			, next( nullptr )
		{

		}



		template < typename T, typename F, typename IndexType >
		friend class SetIterator;

		template < typename T, typename F, typename IndexType >
		friend class Set;

	};




	//######################################################################//
	//																		//
	//							Iterator for Set							//
	//																		//
	//######################################################################//

	template< typename T, typename F, typename IndexType >
	class SetIterator
	{
	public:

		// Default constructor
		SetIterator()
			: m_pMap( nullptr )
			, m_pCurrentNode( nullptr )
			, m_TableIndex( 0 )
		{

		}


		// Constructor
		SetIterator( Set<T, F, IndexType>* pmap )
			: m_pMap( pmap )
			, m_pCurrentNode( nullptr )
			, m_TableIndex( 0 )			
		{
			if( pmap )
			{
				while( m_pCurrentNode==nullptr && m_TableIndex < pmap->m_pTable.Length<IndexType>() )
				{
					m_pCurrentNode = pmap->m_pTable[ m_TableIndex++ ];
				}
			}
			else
			{
				m_TableIndex = (IndexType)HashConst::DefaultHashSize;
			}
		}


		// Copy constructor
		SetIterator( const SetIterator& obj )
			: m_pMap( obj.m_pMap )
			, m_pCurrentNode( obj.m_pCurrentNode )
			, m_TableIndex( obj.m_TableIndex )
		{
			//tcout << "SetIterator copy constructor called...\n";
		}


		SetIterator& operator++()
		{
			m_pCurrentNode = m_pCurrentNode->next;

			while( m_pCurrentNode==nullptr && m_TableIndex < m_pMap->m_pTable.Length() )
				m_pCurrentNode = m_pMap->m_pTable[ m_TableIndex++ ];

			return *this;
		}


		const T& operator*() const
		{
			return m_pCurrentNode->value;
		}


		const T* operator->() const
		{
			return &m_pCurrentNode->value;
		}


		bool operator==( const SetIterator& it ) const
		{
			return m_pCurrentNode == it.m_pCurrentNode;
		}


		bool operator!=( const SetIterator& it ) const
		{
			return m_pCurrentNode != it.m_pCurrentNode;
		}



	private:

		Set<T, F, IndexType>*	m_pMap;
		SetNode<T>*				m_pCurrentNode;
		IndexType				m_TableIndex;

	};




	//######################################################################//
	//																		//
	//									Set									//
	//																		//
	//######################################################################//

	template < typename T, typename F = KeyHash<T>, typename IndexType = MemSizeType >
	class Set
	{
	public:

		// Default constructor
		Set( size_t hashSize=HashConst::DefaultHashSize )
			: m_pTable( static_cast<IndexType>(hashSize) )
			, hashFunc()
			, m_numElements( 0 )
		{

		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same< Pair<T>, Args...>::value >* = nullptr >
		//Set( Args const & ... args )
		//	: m_pTable()
		//	, hashFunc()
		//	, m_numElements( 0 )
		//{

		//}


		Set( std::initializer_list<T> ilist )
			: m_pTable( static_cast<IndexType>( ilist.size() ) )
			, hashFunc()
			, m_numElements( 0 )
		{
			for( const auto& val : ilist )
				Put( val );
		}


		template < typename Iter >
		Set( Iter first, Iter last )
			: m_pTable( static_cast<IndexType>(last - first) )
			, hashFunc()
			, m_numElements( 0 )
		{
			for(; first != last; ++first )
				Put( *first );
		}


		// Destructor
		~Set()
		{
			Clear();
		}


		// Copy constructor
		Set( const Set& obj )
			: m_pTable( obj.m_pTable.Length() )
			, hashFunc( obj.hashFunc )
			, m_numElements( obj.m_numElements )
		{

			for( int i=0; i<m_pTable.Length<int>(); ++i )
			{
				SetNode<T>* objentry = obj.m_pTable[i];
				SetNode<T>* entry = m_pTable[i];

				while( objentry )
				{
					SetNode<T>* newNode = new SetNode<T>( objentry->value );
	
					if( !entry )
						m_pTable[i] = newNode;
					else
						entry->next = newNode;

					entry = newNode;
					objentry = objentry->next;
				}
			}

		}


		// Move constructor
		Set( Set&& obj )
			: m_pTable( (Memory<SetNode<T>*>)obj.m_pTable )
			, hashFunc( obj.hashFunc )
			, m_numElements( obj.m_numElements )
		{
			obj.m_numElements = 0;
		}


		// Copy Assignment opertor =
		Set& operator=( const Set& obj )
		{
			if( this != &obj )
			{
				m_pTable.Init( obj.m_pTable.Length() );

				hashFunc		= obj.hashFunc;
				m_numElements	= obj.m_numElements;

				for( int i=0; i<m_pTable.Length<int>(); ++i )
				{
					SetNode<T>* objentry = obj.m_pTable[i];
					SetNode<T>* entry = m_pTable[i];

					while( objentry )
					{
						SetNode<T>* newNode = new SetNode<T>( objentry->value );
	
						if( !entry )
							m_pTable[i] = newNode;
						else
							entry->next = newNode;

						entry = newNode;
						objentry = objentry->next;
					}
				}
			}
		
			return *this;
		}


		// Move assignment opertor =
		Set& operator=( Set&& obj )
		{
			if( this != &obj )
			{
				Clear();

				hashFunc		= obj.hashFunc;
				m_numElements	= obj.m_numElements;
				m_pTable		= (Memory<SetNode<T>*, IndexType>&&)obj.m_pTable;

				obj.m_numElements = 0;
			}

			return *this;
		}


		void Put( const T& value )
		{
			IndexType hashValue = hashFunc.Get<IndexType>( value, m_pTable.Length() );
			SetNode<T>* prev = nullptr;
			SetNode<T>* entry = m_pTable[ hashValue ];

			while( entry != nullptr && entry->value != value )
			{
				prev = entry;
				entry = entry->next;
			}

			if( entry == nullptr )
			{
				++m_numElements;
				entry = new SetNode<T>( value );

				if( prev == nullptr )
					m_pTable[ hashValue ] = entry;
				else
					prev->next = entry;
			}
			else
			{
				//entry->second = value;
			}

		}


		void Remove( const T& value )
		{
			IndexType hashValue = hashFunc.Get<IndexType>( value, m_pTable.Length() );
			SetNode<T>* prev = nullptr;
			SetNode<T>* entry = m_pTable[ hashValue ];

			while( entry != nullptr && entry->value != value )
			{
				prev = entry;
				entry = entry->next;
			}

			if( entry == nullptr )
			{
				return;
			}
			else
			{
				if( prev == nullptr )
					m_pTable[ hashValue ] = entry->next;
				else
					prev->next = entry->next;

				SafeDelete( entry );
				--m_numElements;
			}

		}


		void Clear()
		{
			for( int i=0; i<m_pTable.Length<int>(); ++i )
			{
				SetNode<T>* entry = m_pTable[i];

				while( entry )
				{
					SetNode<T>* prev = entry;
					entry = entry->next;
					SafeDelete( prev );
				}

				m_pTable[i] = nullptr;
			}

			m_numElements = 0;
		}


		bool Exists( const T& value ) const
		{
			IndexType hashValue = hashFunc.Get<IndexType>( value, m_pTable.Length() );
			SetNode<T>* entry = m_pTable[ hashValue ];

			for( auto entry = m_pTable[ hashValue ]; entry != nullptr; entry=entry->next )
			{
				if( entry->value == value )
					return true;
			}

			return false;
		}


		int Length() const
		{
			return m_numElements;
		}


		bool Empty() const
		{
			return m_numElements<=0;
		}


		SetIterator<T, F, IndexType> begin() const
		{
			return SetIterator<T, F, IndexType>( (Set*)this );
		}


		SetIterator<T, F, IndexType> end() const
		{
			return SetIterator<T, F, IndexType>( nullptr );
		}



	private:

		Memory<SetNode<T>*, IndexType>	m_pTable;
		F hashFunc;
		int	m_numElements = 0;


		template < typename T, typename F, typename IndexType >
		friend class SetIterator;

	};



}// end of namespace


#endif // !SET_H
