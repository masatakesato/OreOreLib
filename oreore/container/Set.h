#ifndef SET_H
#define	SET_H

#include	<exception>

#include	"../common/HashCode.h"

#include	"Array.h"
#include	"StaticArray.h"



// https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1134/materials/cppdoc/hashmap-h.html
// https://aozturk.medium.com/simple-hash-map-hash-m_pTable-implementation-in-c-931965904250
// https://github.com/aozturk/HashMap



namespace OreOreLib
{
	
	//######################################################################//
	//																		//
	//							Class declaration							//
	//																		//
	//######################################################################//

	// SetBase 
	template < typename T, typename IndexType, typename F, sizeType HashSize >	class SetBase;


	// Dynamic Set
	template < typename T, typename IndexType = MemSizeType, typename F = KeyHash<T> >
	using Set = SetBase< T, IndexType, F, detail::DynamicSize >;

	// Static Set
	template < typename T, sizeType HashSize, typename IndexType = MemSizeType, typename F = KeyHash<T> >
	using StaticSet = SetBase< T, IndexType, F, HashSize >;




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



		template < typename T, typename IndexType, typename F, sizeType HashSize >
		friend class SetIterator;

		template < typename T, typename IndexType, typename F, sizeType HashSize >
		friend class SetBase;

	};




	//######################################################################//
	//																		//
	//							Iterator for Set							//
	//																		//
	//######################################################################//

	template< typename T, typename IndexType, typename F, sizeType HashSize >
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
		SetIterator( SetBase<T, IndexType, F, HashSize >* pmap )
			: m_pMap( pmap )
			, m_pCurrentNode( nullptr )
			, m_TableIndex( 0 )			
		{
			if( pmap )
			{
				while( m_pCurrentNode==nullptr && m_TableIndex < pmap->m_pTable.Length() )
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

		SetBase<T, IndexType, F, HashSize>*	m_pMap;
		SetNode<T>*				m_pCurrentNode;
		IndexType				m_TableIndex;

	};




	//######################################################################//
	//																		//
	//						Set(Dynamic hash size)							//
	//																		//
	//######################################################################//

	template < typename T, typename IndexType, typename F >
	class SetBase< T, IndexType, F, detail::DynamicSize >
	{
		using Iter = SetIterator< T, IndexType, F, detail::DynamicSize >;

	public:

		// Default constructor
		SetBase( size_t hashSize=HashConst::DefaultHashSize )
			: m_pTable( static_cast<IndexType>(hashSize) )
			, m_HashFunc()
			, m_numElements( 0 )
		{

		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same< Pair<T>, Args...>::value >* = nullptr >
		//SetBase( Args const & ... args )
		//	: m_pTable()
		//	, m_HashFunc()
		//	, m_numElements( 0 )
		//{

		//}


		SetBase( std::initializer_list<T> ilist )
			: m_pTable( static_cast<IndexType>( Ceil( ilist.size() / HashConst::MaxLoadFactor ) ) )
			, m_HashFunc()
			, m_numElements( 0 )
		{
			for( const auto& val : ilist )
				Put( val );
		}


		template < typename Iter >
		SetBase( Iter first, Iter last )
			: m_pTable( static_cast<IndexType>( Ceil( (last - first) / HashConst::MaxLoadFactor ) ) )
			, m_HashFunc()
			, m_numElements( 0 )
		{
			for(; first != last; ++first )
				Put( *first );
		}


		// Destructor
		~SetBase()
		{
			Clear();
		}


		// Copy constructor
		SetBase( const SetBase& obj )
			: m_pTable( obj.m_pTable.Length() )
			, m_HashFunc( obj.m_HashFunc )
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
		SetBase( SetBase&& obj )
			: m_pTable( (ArrayImpl<SetNode<T>*, IndexType>)obj.m_pTable )
			, m_HashFunc( obj.m_HashFunc )
			, m_numElements( obj.m_numElements )
		{
			obj.m_numElements = 0;
		}


		// Copy Assignment opertor =
		SetBase& operator=( const SetBase& obj )
		{
			if( this != &obj )
			{
				Clear();

				m_pTable.Init( obj.m_pTable.Length() );
				m_HashFunc		= obj.m_HashFunc;
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
		SetBase& operator=( SetBase&& obj )
		{
			if( this != &obj )
			{
				Clear();

				m_pTable		= (ArrayImpl<SetNode<T>*, IndexType>&&)obj.m_pTable;
				m_HashFunc		= obj.m_HashFunc;
				m_numElements	= obj.m_numElements;

				obj.m_numElements = 0;
			}

			return *this;
		}


		void Put( const T& value )
		{
			// Rehash if load facter exceeds limit value
			if( (float32)(m_numElements + 1) / m_pTable.Length<float32>() > HashConst::MaxLoadFactor )
				Rehash();

			// Put value into m_pTable
			IndexType hashValue = m_HashFunc.Get<IndexType>( value, m_pTable.Length() );
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
			//else
			//{
			//	//entry->second = value;
			//}

		}


		void Remove( const T& value )
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( value, m_pTable.Length() );
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
			IndexType hashValue = m_HashFunc.Get<IndexType>( value, m_pTable.Length() );
			SetNode<T>* entry = m_pTable[ hashValue ];

			for( auto entry = m_pTable[ hashValue ]; entry != nullptr; entry=entry->next )
			{
				if( entry->value == value )
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
			return Iter( (SetBase*)this );
		}


		Iter end() const
		{
			return Iter( nullptr );
		}



	private:

		ArrayImpl<SetNode<T>*, IndexType>	m_pTable;
		F									m_HashFunc;
		IndexType							m_numElements;



		void Rehash()
		{
			// Create new hash table
			ArrayImpl<SetNode<T>*, IndexType>	newTable( (m_numElements + 1) * 2 );
			tcout << _T("SetBase::Rehash()... ") << m_pTable.Length() << _T("->") << newTable.Length() << tendl;

			// transfer nodes from m_pTable to newTable
			for( int i=0; i<m_pTable.Length<int>(); ++i )
			{
				SetNode<T>* entry = m_pTable[i];

				while( entry )
				{
					SetNode<T>* prev = entry;
					entry = entry->next;
					if( TransferSetNode( prev, newTable ) == false )
						SafeDelete( prev );
				}

				m_pTable[i] = nullptr;
			}

			m_pTable = (ArrayImpl<SetNode<T>*, IndexType>&&)newTable;
		}


		bool TransferSetNode( SetNode<T>* node, ArrayImpl<SetNode<T>*, IndexType>& pTable )
		{
			// Put value into pTable
			IndexType hashValue = m_HashFunc.Get<IndexType>( node->value, pTable.Length<IndexType>() );
			SetNode<T>* prev = nullptr;
			SetNode<T>* entry = pTable[ hashValue ];

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
	//						Set(static hash size)							//
	//																		//
	//######################################################################//

	template < typename T, sizeType HashSize, typename IndexType, typename F >
	class SetBase< T, IndexType, F, HashSize >
	{
		using Iter = SetIterator< T, IndexType, F, HashSize >;

	public:

		// Default constructor
		SetBase()
			: m_pTable()
			, m_HashFunc()
			, m_numElements( 0 )
		{

		}


		// Constructor
		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same< Pair<T>, Args...>::value >* = nullptr >
		//SetBase( Args const & ... args )
		//	: m_pTable()
		//	, hashFunc()
		//	, m_numElements( 0 )
		//{

		//}


		SetBase( std::initializer_list<T> ilist )
			: m_pTable()
			, m_HashFunc()
			, m_numElements( 0 )
		{
			for( const auto& val : ilist )
				Put( val );
		}


		template < typename Iter >
		SetBase( Iter first, Iter last )
			: m_pTable()
			, m_HashFunc()
			, m_numElements( 0 )
		{
			for(; first != last; ++first )
				Put( *first );
		}


		// Destructor
		~SetBase()
		{
			Clear();
		}


		// Copy constructor
		SetBase( const SetBase& obj )
			: m_pTable()
			, m_HashFunc( obj.m_HashFunc )
			, m_numElements( obj.m_numElements )
		{

			for( int i=0; i<HashSize; ++i )
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
		SetBase( SetBase&& obj )
			: m_HashFunc( obj.m_HashFunc )
			, m_numElements( obj.m_numElements )
		{
			memcpy( m_pTable.begin(), obj.m_pTable.begin(), sizeof (SetNode<T>*) * HashSize );

			memset( obj.m_pTable.begin(), 0, sizeof (SetNode<T>*) * HashSize );
			obj.m_numElements = 0;
		}


		// Copy Assignment opertor =
		SetBase& operator=( const SetBase& obj )
		{
			if( this != &obj )
			{
				Clear();

				m_HashFunc		= obj.m_HashFunc;
				m_numElements	= obj.m_numElements;

				for( int i=0; i<HashSize; ++i )
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
		SetBase& operator=( SetBase&& obj )
		{
			if( this != &obj )
			{
				Clear();

				m_HashFunc		= obj.m_HashFunc;
				m_numElements	= obj.m_numElements;
				memcpy( m_pTable.begin(), obj.m_pTable.begin(), sizeof (SetNode<T>*) * HashSize );

				memset( obj.m_pTable.begin(), 0, sizeof (SetNode<T>*) * HashSize );
				obj.m_numElements = 0;
			}

			return *this;
		}


		void Put( const T& value )
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( value, m_pTable.Length() );
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
			//else
			//{
			//	//entry->second = value;
			//}

		}


		void Remove( const T& value )
		{
			IndexType hashValue = m_HashFunc.Get<IndexType>( value, m_pTable.Length() );
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
			for( int i=0; i<HashSize; ++i )
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
			IndexType hashValue = m_HashFunc.Get<IndexType>( value, m_pTable.Length() );
			SetNode<T>* entry = m_pTable[ hashValue ];

			for( auto entry = m_pTable[ hashValue ]; entry != nullptr; entry=entry->next )
			{
				if( entry->value == value )
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
			return Iter( (SetBase*)this );
		}


		Iter end() const
		{
			return Iter( nullptr );
		}



	private:

		StaticArrayImpl< SetNode<T>*, HashSize, IndexType >	m_pTable;
		F													m_HashFunc;
		IndexType											m_numElements;


		friend class Iter;

	};



}// end of namespace


#endif // !SET_H
