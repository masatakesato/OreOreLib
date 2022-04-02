#ifndef HASH_MAP_H
#define	HASH_MAP_H

#include	<exception>

#include	"../common/HashCode.h"

#include	"Array.h"
#include	"StaticArray.h"
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
//TODO: Setと実装重複
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
	//							Class declaration							//
	//																		//
	//######################################################################//

	// HashMapBase 
	template < typename K, typename V, typename IndexType, typename F, sizeType HashSize >	class HashMapBase;


	// Dynamic HashMap
	template < typename K, typename V, typename IndexType = MemSizeType, typename F = KeyHash<K> >
	using HashMap = HashMapBase< K, V, IndexType, F, detail::DynamicSize >;

	// Static HashMap
	template < typename K, typename V, sizeType HashSize, typename IndexType = MemSizeType, typename F = KeyHash<K> >
	using StaticHashMap = HashMapBase< K, V, IndexType, F, HashSize >;




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



		template < typename K, typename V, typename IndexType, typename F, sizeType HashSize >
		friend class HashMapIterator;

		template < typename K, typename V, typename IndexType, typename F, sizeType HashSize >
		friend class HashMapBase;

	};




	//######################################################################//
	//																		//
	//						Iterator for HashMapBase						//
	//																		//
	//######################################################################//

	template< typename K, typename V, typename IndexType, typename F, sizeType HashSize >
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
		HashMapIterator( HashMapBase<K, V, IndexType, F, HashSize>* pmap )
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

			while( m_pCurrentNode==nullptr && m_TableIndex < m_pMap->m_pTable.Length<IndexType>() )
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

		HashMapBase<K, V, IndexType, F, HashSize>*	m_pMap;
		HashNode<K, V>*					m_pCurrentNode;
		IndexType						m_TableIndex;

	};




	//######################################################################//
	//																		//
	//						HashMap(Dynamic hash size)						//
	//																		//
	//######################################################################//

	template < typename K, typename V, typename IndexType, typename F >
	class HashMapBase< K, V, IndexType, F, detail::DynamicSize >
	{
		using Iter = HashMapIterator< K, V, IndexType, F, detail::DynamicSize >;

	public:

		// Default constructor
		HashMapBase( size_t hashSize=HashConst::DefaultHashSize )
			: m_pTable( static_cast<IndexType>(hashSize) )
			, m_HashFunc()
			, m_numElements( 0 )
		{

		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same< Pair<K, V>, Args...>::value >* = nullptr >
		//HashMapBase( Args const & ... args )
		//	: m_pTable()
		//	, m_HashFunc()
		//	, m_numElements( 0 )
		//{

		//}


		HashMapBase( std::initializer_list< Pair<K, V> > list )
			: m_pTable( static_cast<IndexType>( list.size() ) )
			, m_HashFunc()
			, m_numElements( 0 )
		{
			for( const auto& pair : list )
				Put( pair.first, pair.second );
		}


		// Destructor
		~HashMapBase()
		{
			Clear();
		}


		// Copy constructor
		HashMapBase( const HashMapBase& obj )
			: m_pTable( obj.m_pTable.Length() )
			, m_HashFunc( obj.m_HashFunc )
			, m_numElements( obj.m_numElements )
		{

			for( int i=0; i<m_pTable.Length<int>(); ++i )
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
		HashMapBase( HashMapBase&& obj )
			: m_pTable( (ArrayImpl<HashNode<K, V>*, IndexType>&&) obj.m_pTable )
			, m_HashFunc( obj.m_HashFunc )
			, m_numElements( obj.m_numElements )
		{
			obj.m_numElements = 0;
		}


		// Copy Assignment opertor =
		HashMapBase& operator=( const HashMapBase& obj )
		{
			if( this != &obj )
			{
				Clear();

				m_pTable.Init( obj.m_pTable.Length() );
				m_HashFunc		= obj.m_HashFunc;
				m_numElements	= obj.m_numElements;

				for( int i=0; i<m_pTable.Length<int>(); ++i )
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
		HashMapBase& operator=( HashMapBase&& obj )
		{
			if( this != &obj )
			{
				Clear();

				m_pTable		= (ArrayImpl<HashNode<K, V>*, IndexType>&&) obj.m_pTable;
				m_HashFunc		= obj.m_HashFunc;
				m_numElements	= obj.m_numElements;

				obj.m_numElements = 0;
			}

			return *this;
		}


		// Subscription operator for read only.( called if HashMapBase is const )
		inline const V& operator[]( const K& key ) const&
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		// Subscription operator for read-write.( called if HashMapBase is non-const )
		inline V& operator[]( const K& key ) &
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
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


		// Subscription operator. ( called by following cases: "T a = HashMapBase<tstring, int, T>()[n]", "auto&& a = HashMapBase<tstring, int, T>()[n]" )
		inline V operator[]( const K& key ) const&&
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		// At method for non-const HashMapBase 
		V& At( const K& key )
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		// At method for const HashMapBase 
		const V& At( const K& key ) const
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		bool Get( const K& key, V& value )
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
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
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
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
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
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
			for( int i=0; i<m_pTable.Length<int>(); ++i )
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
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			for( auto entry = m_pTable[ hashValue ]; entry != nullptr; entry=entry->next )
			{
				if( entry->first == key )
					return true;
			}

			return false;
		}


		IndexType Length() const
		{
			return m_numElements;
		}


		bool Empty() const
		{
			return m_numElements<=0;
		}


		Iter begin() const
		{
			return Iter( (HashMapBase*)this );
		}


		Iter end() const
		{
			return Iter( nullptr );
		}



	private:

		ArrayImpl<HashNode<K, V>*, IndexType>	m_pTable;
		F										m_HashFunc;
		IndexType								m_numElements;


		void Rehash()
		{
			// Create new hash table
			ArrayImpl<HashNode<K, V>*, IndexType>	newTable( (m_numElements + 1) * 2 );
			tcout << _T("HashMapBase::Rehash()... ") << m_pTable.Length() << _T("->") << newTable.Length() << tendl;

			// transfer nodes from m_pTable to newTable
			for( int i=0; i<m_pTable.Length<int>(); ++i )
			{
				HashNode<K, V>* entry = m_pTable[i];

				while( entry )
				{
					HashNode<K, V>* prev = entry;
					entry = entry->next;
					if( TransferNode( prev, newTable ) == false )
						SafeDelete( prev );
				}

				m_pTable[i] = nullptr;
			}

			m_pTable = (ArrayImpl<HashNode<K, V>*, IndexType>&&)newTable;
		}


		bool TransferNode( HashNode<K, V>* node, ArrayImpl<HashNode<K, V>*, IndexType>& pTable )
		{
			// Rehash if load facter exceeds limit value
			if( (float32)(m_numElements + 1) / m_pTable.Length<float32>() > HashConst::MaxLoadFactor )
				Rehash();

			// Put value into pTable
			IndexType hashValue = m_HashFunc.Get<IndexType>( node->value, pTable.Length<IndexType>() );
			HashNode<K, V>* prev = nullptr;
			HashNode<K, V>* entry = pTable[ hashValue ];

			// move to last element
			while( entry != nullptr && entry->value != node->value )
			{
				prev = entry;
				entry = entry->next;
			}

			if( entry == nullptr )
			{
				// disconnect node from current linklist
				node->next = nullptr;

				if( prev == nullptr )
					pTable[ hashValue ] = node;
				else
					prev->next = node;

				return true;
			}

			return false;
		}


		friend class Iter;

	};




	//######################################################################//
	//																		//
	//						HashMap( Static hash size )						//
	//																		//
	//######################################################################//

	template < typename K, typename V, sizeType HashSize, typename IndexType, typename F >
	class HashMapBase< K, V, IndexType, F, HashSize >
	{
		using Iter = HashMapIterator< K, V, IndexType, F, HashSize >;

	public:

		// Default constructor
		HashMapBase()
			: m_pTable()
			, m_HashFunc()
			, m_numElements( 0 )
		{

		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same< Pair<K, V>, Args...>::value >* = nullptr >
		//HashMapBase( Args const & ... args )
		//	: m_pTable()
		//	, m_HashFunc()
		//	, m_numElements( 0 )
		//{

		//}



		HashMapBase( std::initializer_list< Pair<K, V> > list )
			: m_pTable()
			, m_HashFunc()
			, m_numElements( 0 )
		{
			for( const auto& pair : list )
				Put( pair.first, pair.second );
		}


		// Destructor
		~HashMapBase()
		{
			Clear();
		}


		// Copy constructor
		HashMapBase( const HashMapBase& obj )
			: m_pTable()
			, m_HashFunc( obj.m_HashFunc )
			, m_numElements( obj.m_numElements )
		{

			for( int i=0; i<HashSize; ++i )
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
		HashMapBase( HashMapBase&& obj )
			: m_HashFunc( obj.m_HashFunc )
			, m_numElements( obj.m_numElements )
		{
			memcpy( m_pTable.begin(), obj.m_pTable.begin(), sizeof (HashNode<K,V>*) * HashSize );

			memset( obj.m_pTable.begin(), 0, sizeof (HashNode<K,V>*) * HashSize );
			obj.m_numElements = 0;
		}


		// Copy Assignment opertor =
		HashMapBase& operator=( const HashMapBase& obj )
		{
			if( this != &obj )
			{
				Clear();

				m_HashFunc		= obj.m_HashFunc;
				m_numElements	= obj.m_numElements;

				for( int i=0; i<HashSize; ++i )
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
		HashMapBase& operator=( HashMapBase&& obj )
		{
			if( this != &obj )
			{
				Clear();

				memcpy( m_pTable.begin(), obj.m_pTable.begin(), sizeof (HashNode<K,V>*) * HashSize );
				m_HashFunc		= obj.m_HashFunc;
				m_numElements	= obj.m_numElements;

				memset( obj.m_pTable.begin(), 0, sizeof (HashNode<K,V>*) * HashSize );
				obj.m_numElements = 0;
			}

			return *this;
		}


		// Subscription operator for read only.( called if HashMapBase is const )
		inline const V& operator[]( const K& key ) const&
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		// Subscription operator for read-write.( called if HashMapBase is non-const )
		inline V& operator[]( const K& key ) &
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
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


		// Subscription operator. ( called by following cases: "T a = HashMapBase<tstring, int, T>()[n]", "auto&& a = HashMapBase<tstring, int, T>()[n]" )
		inline V operator[]( const K& key ) const&&
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		// At method for non-const HashMapBase 
		V& At( const K& key )
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		// At method for const HashMapBase 
		const V& At( const K& key ) const
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			while( entry && entry->first != key )
				entry = entry->next;

			if( entry==nullptr )
				throw OutOfBoundsException();

			return entry->second;
		}


		bool Get( const K& key, V& value )
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
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
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
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
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
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
			for( int i=0; i<HashSize; ++i )
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
			IndexType hashValue = m_HashFunc.Get<IndexType>( key, m_pTable.Length() );
			HashNode<K, V>* entry = m_pTable[ hashValue ];

			for( auto entry = m_pTable[ hashValue ]; entry != nullptr; entry=entry->next )
			{
				if( entry->first == key )
					return true;
			}

			return false;
		}


		IndexType Length() const
		{
			return m_numElements;
		}


		bool Empty() const
		{
			return m_numElements<=0;
		}


		Iter begin() const
		{
			return Iter( (HashMapBase*)this );
		}


		Iter end() const
		{
			return Iter( nullptr );
		}



	private:

		StaticArrayImpl<HashNode<K, V>*, HashSize, IndexType>	m_pTable;
		F														m_HashFunc;
		IndexType												m_numElements;


		friend class Iter;

	};




}// end of namespace


#endif // !HASH_MAP_H
