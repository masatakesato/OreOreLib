#ifndef SET_H
#define	SET_H

#include	<exception>

#include	"../common/HashCode.h"



// https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1134/materials/cppdoc/hashmap-h.html
// https://aozturk.medium.com/simple-hash-map-hash-m_pTable-implementation-in-c-931965904250
// https://github.com/aozturk/HashMap



namespace OreOreLib
{

	//######################################################################//
	//																		//
	//								Exception								//
	//																		//
	//######################################################################//

	class OutOfBoundsException : public std::exception
	{
	public:
		const char* what() const noexcept override
		{
			return "Out of bounds exception.";
		}
	};




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



		template < typename T, size_t TableSize, typename F >
		friend class SetIterator;

		template < typename T, size_t TableSize, typename F >
		friend class Set;

	};



	//######################################################################//
	//																		//
	//					KewyHash class implementation						//
	//																		//
	//######################################################################//

	template < typename T, size_t TableSize >
	struct KeyHash
	{
		uint64 operator()( const T& value ) const
		{
			return HashCode( value ) % TableSize;
			//return *(uint64*)( &value ) % TableSize;
		}

	};



	// https://stackoverflow.com/questions/8094790/how-to-get-hash-code-of-a-string-in-c

	inline uint64 hashCode( const tstring& text )
	{
		uint64 hash = 0, strlen = text.length(), i;
		TCHAR character;

		if( strlen == 0 )
			return hash;

		for( i=0; i<strlen; ++i )
		{
			character = text.at(i);
			hash = (31 * hash) + character;
		}

		return hash;
	}


	// https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1178/lectures/27-Inheritance/code/Inheritance/lib/StanfordCPPLib/collections/hashcode.h
	// https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1178/lectures/27-Inheritance/code/Inheritance/lib/StanfordCPPLib/collections/hashcode.cpp






	//######################################################################//
	//																		//
	//							Iterator for Set							//
	//																		//
	//######################################################################//

	template< typename T, size_t TableSize, typename F >
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
		SetIterator( Set<T, TableSize, F>* pmap )
			: m_pMap( pmap )
			, m_pCurrentNode( nullptr )
			, m_TableIndex( 0 )			
		{
			if( pmap )
			{
				while( m_pCurrentNode==nullptr && m_TableIndex < TableSize )
				{
					m_pCurrentNode = pmap->m_pTable[ m_TableIndex++ ];
				}
			}
			else
			{
				m_TableIndex = TableSize;
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

			while( m_pCurrentNode==nullptr && m_TableIndex < TableSize )
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

		Set<T, TableSize, F>*	m_pMap;
		SetNode<T>*				m_pCurrentNode;
		int						m_TableIndex;

	};




	//######################################################################//
	//																		//
	//									Set									//
	//																		//
	//######################################################################//

	template < typename T, size_t TableSize = 64, typename F = KeyHash<T, TableSize> >
	class Set
	{
	public:

		// Default constructor
		Set()
			: m_pTable()
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
			: m_pTable()
			, hashFunc()
			, m_numElements( 0 )
		{
			for( const auto& val : ilist )
				Put( val );
		}


		template < typename Iter >
		Set( Iter first, Iter last )
			: m_pTable()
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
			: m_pTable()
			, hashFunc( obj.hashFunc )
			, m_numElements( obj.m_numElements )
		{

			for( int i=0; i<TableSize; ++i )
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
			: hashFunc( obj.hashFunc )
			, m_numElements( obj.m_numElements )
		{
			memcpy( m_pTable, obj.m_pTable, sizeof (SetNode<T>*) * TableSize );

			memset( obj.m_pTable, 0, sizeof (SetNode<T>*) * TableSize );
			obj.m_numElements = 0;
		}


		// Copy Assignment opertor =
		Set& operator=( const Set& obj )
		{
			if( this != &obj )
			{
				hashFunc		= obj.hashFunc;
				m_numElements	= obj.m_numElements;

				for( int i=0; i<TableSize; ++i )
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
				hashFunc		= obj.hashFunc;
				m_numElements	= obj.m_numElements;
				memcpy( m_pTable, obj.m_pTable, sizeof (SetNode<T>*) * TableSize );

				memset( obj.m_pTable, 0, sizeof (SetNode<T>*) * TableSize );
				obj.m_numElements = 0;
			}

			return *this;
		}


		void Put( const T& value )
		{
			uint64 hashValue = hashFunc( value );
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
			uint64 hashValue = hashFunc( value );
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
			for( int i=0; i<TableSize; ++i )
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
			uint64 hashValue = hashFunc( value );
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


		SetIterator<T, TableSize, F> begin() const
		{
			return SetIterator<T, TableSize, F>( (Set*)this );
		}


		SetIterator<T, TableSize, F> end() const
		{
			return SetIterator<T, TableSize, F>( nullptr );
		}



	private:

		SetNode<T>*	m_pTable[ TableSize ];
		F hashFunc;
		int	m_numElements = 0;


		template < typename T, size_t TableSize, typename F >
		friend class SetIterator;

	};



}// end of namespace


#endif // !SET_H
