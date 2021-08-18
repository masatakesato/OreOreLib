#ifndef HASH_MAP_H
#define	HASH_MAP_H

#include	<exception>

#include	"../common/Utility.h"
#include	"../common/HashCode.h"
#include	"../meta/TypeTraits.h"
#include	"Pair.h"



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
	//								HashNode								//
	//																		//
	//######################################################################//

	template < typename K, typename V >
	class HashNode : public Pair<K, V>
	{
		HashNode* next = nullptr;


		HashNode( const K& key )
			: Pair<K, V>( key )
			, next( nullptr )
		{

		}


		HashNode( const K& key, const V& value )
			: Pair<K, V>( key, value )
			, next( nullptr )
		{

		}



		template < typename K, typename V, size_t TableSize, typename F >
		friend class HashMapIterator;

		template < typename K, typename V, size_t TableSize, typename F >
		friend class HashMap;

	};



	//######################################################################//
	//																		//
	//					KewyHash class implementation						//
	//																		//
	//######################################################################//

	template < typename K, size_t TableSize >
	struct KeyHash
	{
		uint64 operator()( const K& key ) const
		{
			return HashCode( key ) % TableSize;
			//return *(uint64*)( &key ) % TableSize;
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
	//						Iterator for HashMap							//
	//																		//
	//######################################################################//

	template< typename K, typename V, size_t TableSize, typename F >
	class HashMapIterator
	{
	public:

		// Default constructor
		HashMapIterator()
			: m_pMap( nullptr )
			, m_pCurrentNode( nullptr )
			, m_TableIndex( 0 )
		{

		}


		// Constructor
		HashMapIterator( HashMap<K, V, TableSize, F>* pmap )
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
		HashMapIterator( const HashMapIterator& obj )
			: m_pMap( obj.m_pMap )
			, m_pCurrentNode( obj.m_pCurrentNode )
			, m_TableIndex( obj.m_TableIndex )
		{
			//tcout << "HashMapIterator copy constructor called...\n";
		}


		HashMapIterator& operator++()
		{
			m_pCurrentNode = m_pCurrentNode->next;

			while( m_pCurrentNode==nullptr && m_TableIndex < TableSize )
				m_pCurrentNode = m_pMap->m_pTable[ m_TableIndex++ ];

			return *this;
		}

		
		HashNode<K, V>& operator*()
		{
			return *m_pCurrentNode;
		}


		const HashNode<K, V>& operator*() const
		{
			return *m_pCurrentNode;
		}


		HashNode<K, V>* operator->()
		{
			return m_pCurrentNode;
		}


		const HashNode<K, V>* operator->() const
		{
			return m_pCurrentNode;
		}



		bool operator==( const HashMapIterator& it ) const
		{
			return m_pCurrentNode == it.m_pCurrentNode;
		}


		bool operator!=( const HashMapIterator& it ) const
		{
			return m_pCurrentNode != it.m_pCurrentNode;
		}



	private:

		HashMap<K, V, TableSize, F>*	m_pMap;
		HashNode<K, V>*					m_pCurrentNode;
		int								m_TableIndex;

	};




	//######################################################################//
	//																		//
	//								HashMap									//
	//																		//
	//######################################################################//

	template < typename K, typename V, size_t TableSize, typename F = KeyHash<K, TableSize> >
	class HashMap
	{
	public:

		// Default constructor
		HashMap()
			: m_pTable()
			, hashFunc()
			, m_numElements( 0 )
		{

		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same< Pair<K, V>, Args...>::value >* = nullptr >
		//HashMap( Args const & ... args )
		//	: m_pTable()
		//	, hashFunc()
		//	, m_numElements( 0 )
		//{

		//}



		HashMap( std::initializer_list< Pair<K, V> > list )
			: m_pTable()
			, hashFunc()
			, m_numElements( int(list.size()) )
		{

			for( const auto& pair : list )
				Put( pair.first, pair.second );
		}


		// Destructor
		~HashMap()
		{
			Clear();
		}


		// Copy constructor
		HashMap( const HashMap& obj )
			: m_pTable()
			, hashFunc( obj.hashFunc )
			, m_numElements( obj.m_numElements )
		{

			for( int i=0; i<TableSize; ++i )
			{
				HashNode<K, V>* objentry = obj.m_pTable[i];
				HashNode<K, V>* entry = m_pTable[i];

				while( objentry )
				{
					HashNode<K, V>* newNode = new HashNode<K, V>( objentry->first, objentry->second );
	
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
		HashMap( HashMap&& obj )
			: hashFunc( obj.hashFunc )
			, m_numElements( obj.m_numElements )
		{
			memcpy( m_pTable, obj.m_pTable, sizeof (HashNode<K,V>*) * TableSize );

			memset( obj.m_pTable, 0, sizeof (HashNode<K,V>*) * TableSize );
			obj.m_numElements = 0;
		}


		// Copy Assignment opertor =
		HashMap& operator=( const HashMap& obj )
		{
			if( this != &obj )
			{
				hashFunc		= obj.hashFunc;
				m_numElements	= obj.m_numElements;

				for( int i=0; i<TableSize; ++i )
				{
					HashNode<K, V>* objentry = obj.m_pTable[i];
					HashNode<K, V>* entry = m_pTable[i];

					while( objentry )
					{
						HashNode<K, V>* newNode = new HashNode<K, V>( objentry->first, objentry->second );
	
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
		HashMap& operator=( HashMap&& obj )
		{
			if( this != &obj )
			{
				hashFunc		= obj.hashFunc;
				m_numElements	= obj.m_numElements;
				memcpy( m_pTable, obj.m_pTable, sizeof (HashNode<K,V>*) * TableSize );

				memset( obj.m_pTable, 0, sizeof (HashNode<K,V>*) * TableSize );
				obj.m_numElements = 0;
			}

			return *this;
		}


		// Subscription operator for read only.( called if HashMap is const )
		inline const V& operator[]( const K& key ) const&
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		// Subscription operator for read-write.( called if HashMap is non-const )
		inline V& operator[]( const K& key ) &
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* entry = m_pTable[ hashValue ];
			HashNode<K, V>* prev = entry;

			while( entry && entry->first != key )
			{
				prev = entry;
				entry = entry->next;
			}


			// create new hasnode if key does not exist.
			if( !entry )
			{
				++m_numElements;
				entry = new HashNode<K, V>( key );

				if( prev == nullptr )
					m_pTable[ hashValue ] = entry;
				else
					prev->next = entry;
			}
			
			return entry->second;
		}


		// Subscription operator. ( called by following cases: "T a = HashMap<tstring, int, T>()[n]", "auto&& a = HashMap<tstring, int, T>()[n]" )
		inline V operator[]( const K& key ) const&&
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}



		// At method for non-const HashMap 
		V& At( const K& key )
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}



		// At method for const HashMap 
		const V& At( const K& key ) const
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}



		bool Get( const K& key, V& value )
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry )
			{
				if( entry->first == key )
				{
					value = entry->second;
					return true;
				}

				entry = entry->next;
			}

			return false;
		}


		void Put( const K& key, const V& value )
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* prev = nullptr;
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry != nullptr && entry->first != key )
			{
				prev = entry;
				entry = entry->next;
			}

			if( entry == nullptr )
			{
				++m_numElements;
				entry = new HashNode<K, V>( key, value );

				if( prev == nullptr )
					m_pTable[ hashValue ] = entry;
				else
					prev->next = entry;
			}
			else
			{
				entry->second = value;
			}

		}


		void Remove( const K& key )
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* prev = nullptr;
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry != nullptr && entry->first != key )
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
				HashNode<K, V>* entry = m_pTable[i];

				while( entry )
				{
					HashNode<K, V>* prev = entry;
					entry = entry->next;
					SafeDelete( prev );
				}

				m_pTable[i] = nullptr;
			}

			m_numElements = 0;
		}


		bool Exists( const K& key ) const
		{
			uint64 hashValue = hashFunc( key );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			for( auto entry = m_pTable[ hashValue ]; entry != nullptr; entry=entry->next )
			{
				if( entry->first == key )
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


		HashMapIterator<K, V, TableSize, F> begin() const
		{
			return HashMapIterator<K, V, TableSize, F>( (HashMap*)this );
		}


		HashMapIterator<K, V, TableSize, F> end() const
		{
			return HashMapIterator<K, V, TableSize, F>( nullptr );
		}



	private:

		HashNode<K, V>*	m_pTable[ TableSize ];
		F hashFunc;
		int	m_numElements = 0;


		template < typename K, typename V, size_t TableSize, typename F >
		friend class HashMapIterator;

	};



}// end of namespace


#endif // !HASH_MAP_H
