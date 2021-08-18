#ifndef	UNIQUE_PTR_H
#define	UNIQUE_PTR_H


// https://android.googlesource.com/trusty/app/keymaster/+/a0337ca60a7714bc69365b5103a2ded2b76c870d/UniquePtr.h


#include	"../common/Utility.h"



namespace OreOreLib
{

	template< typename T > class SharedPtr;


	// default deleter for pointer
	template< typename T >
	class DefaultDeleter
	{
	public:

		void operator()( T*& ptr ) const
		{
			//tcout << _T("DefaultDeleter<T>\n");
			SafeDelete( ptr );
		}

	};


	// default deleter specialized for dynamic array
	template< typename T >
	class DefaultDeleter< T[] >
	{
	public:

		void operator()( T*& ptr ) const
		{
			//tcout << _T("DefaultDeleter<T[]>\n");
			SafeDeleteArray( ptr );
		}
	};



	// https://www.zhblog.net/clanguage/cpp-unique-ptr.html

	template< typename T, typename DELETER=DefaultDeleter<T> >
	class UniquePtr
	{
	public:

		friend class SharedPtr<T>;


		// Default constructor
		UniquePtr()
			: m_Ptr( nullptr )
			, m_Deleter( DELETER() )
		{
		}


		// Constructor
		UniquePtr( T* ptr, const DELETER& del=DELETER() )
			: m_Ptr( ptr )
			, m_Deleter( del )
		{
		}


		// Copy Constructor
		UniquePtr( const UniquePtr& obj ) = delete;


		// Move constructor
		UniquePtr( UniquePtr&& obj )
			: m_Ptr( obj.m_Ptr )
		{
			obj.m_Ptr	= nullptr;
		}


		// Destructor
		~UniquePtr()
		{
			m_Deleter( m_Ptr );
		}


		// Copy Assignment Operator
		UniquePtr& operator=( const UniquePtr& obj ) = delete;


		// Move Assignment Operator
		UniquePtr& operator=( UniquePtr&& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				SafeDelete( m_Ptr );
				m_Ptr		= obj.m_Ptr;
				obj.m_Ptr	= nullptr;
			}

			return *this;
		}


		bool operator==( const UniquePtr& obj ) const
		{
			return m_Ptr == obj.m_Ptr;
		}


		// Indirection operator
		T& operator*()
		{
			//tcout << "T& operator*();\n";
			return *m_Ptr;
		}


		// Indirection operator( const object )
		const T& operator*() const
		{
			//tcout << "const T& operator*() const;\n";
			return *m_Ptr;
		}


		// member access operator from const object
		T* operator->()
		{
			//tcout << "T* operator->();\n";
			return m_Ptr;
		}


		// member access operator( const object )
		const T* operator->() const
		{
			//tcout << "const T* operator->() const;\n";
			return m_Ptr;
		}


		operator bool() const
		{
			return m_Ptr != nullptr;
		}


		T* Get() const
		{
			return m_Ptr;
		}


		void Swap( UniquePtr& obj )
		{
			if( this != obj )
			{
				T* ptr		= obj.m_Ptr;
				obj.m_Ptr	= m_Ptr;
				m_Ptr		= ptr;
			}
		}


		void Reset()
		{
			m_Deleter( m_Ptr );
		}



	private:

		T*		m_Ptr;
		DELETER	m_Deleter;

	};



	
	template< typename T, typename DELETER >
	class UniquePtr< T[], DELETER >
	{
	public:

		friend class SharedPtr<T[]>;


		// Default constructor
		UniquePtr()
			: m_Ptr( nullptr )
		{
		}


		// Constructor
		UniquePtr( T* ptr, const DELETER& del=DELETER() )
			: m_Ptr( ptr )
			, m_Deleter( del )
		{
		}


		// Copy Constructor
		UniquePtr( const UniquePtr& obj ) = delete;


		// Move constructor
		UniquePtr( UniquePtr&& obj )
			: m_Ptr( obj.m_Ptr )
		{
			obj.m_Ptr		= nullptr;
		}


		// Destructor
		~UniquePtr()
		{
			m_Deleter( m_Ptr );
		}


		// Copy Assignment Operator
		UniquePtr& operator=( const UniquePtr& obj ) = delete;


		// Move Assignment Operator
		UniquePtr& operator=( UniquePtr&& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				SafeDeleteArray( m_Ptr );
				m_Ptr		= obj.m_Ptr;
				obj.m_Ptr	= nullptr;
			}

			return *this;
		}


		bool operator==( const UniquePtr& obj ) const
		{
			return m_Ptr == obj.m_Ptr;
		}


		// Indirection operator
		T& operator*()
		{
			//tcout << "T& operator*();\n";
			return *m_Ptr;
		}


		// Indirection operator( const object )
		const T& operator*() const
		{
			//tcout << "const T& operator*() const;\n";
			return *m_Ptr;
		}


		// member access operator from const object
		T* operator->()
		{
			//tcout << "T* operator->();\n";
			return m_Ptr;
		}


		// member access operator( const object )
		const T* operator->() const
		{
			//tcout << "const T* operator->() const;\n";
			return m_Ptr;
		}


		// Subscription operator for read only.( called if SharedPtr is const )
		const T& operator[]( std::size_t n ) const&
		{
			return m_Ptr[n];
		}


		// Subscription operator for read-write.( called if SharedPtr is non-const )
		T& operator[]( std::size_t n ) &
		{
			return m_Ptr[n];
		}


		// Subscription operator. ( called by following cases: "T a = SharedPtr<T>(...)[n]", "auto&& a = SharedPtr<T>(...)[n]" )
		T operator[]( std::size_t n ) const&& = delete;
		//{
		//	return std::move(m_Ptr[n]);// return object
		//}


		operator bool() const
		{
			return m_Ptr != nullptr;
		}



		T* Get() const
		{
			return m_Ptr;
		}


		void Swap( UniquePtr& obj )
		{
			if( this != obj )
			{
				T* ptr		= obj.m_Ptr;
				obj.m_Ptr	= m_Ptr;
				m_Ptr		= ptr;
			}
		}


		void Reset()
		{
			m_Deleter( m_Ptr );
		}



	private:

		T*		m_Ptr;
		DELETER	m_Deleter;

	};
	


}// end of namespace


#endif // !UNIQUE_PTR_H
