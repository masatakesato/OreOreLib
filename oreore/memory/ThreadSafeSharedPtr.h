#ifndef	SHARED_PTR_H
#define	SHARED_PTR_H

// http://hp.vector.co.jp/authors/VA025590/handle.html
// https://gist.github.com/tejainece/cb872b24828491c32dbd
// https://medium.com/swlh/c-smart-pointers-and-how-to-write-your-own-c0adcbdce04f


// shared pointer with deleter
// https://codereview.stackexchange.com/questions/166395/custom-stdshared-ptrt-implementation <-

// thread safe shared pointer
// https://codereview.stackexchange.com/questions/186410/custom-thread-safe-shared-weak-pointer-implementation-in-c98/186420

// thread-safe constructor
// https://stackoverflow.com/questions/4037853/thread-safe-copy-constructor-assignment-operator

#include	<functional>
#include	<mutex>
#include	<atomic>

#include	"../common/Utility.h"
#include	"UniquePtr.h"




namespace OreOreLib
{
	template< typename T > class WeakPtr;
	/*
	// Reference count
	class ReferenceCount
	{
	public:

		std::atomic<int> sharedCount = 0;
		std::atomic<int> weakCount = 0;
	};
	*/



	// Reference Counter with mutex
	struct ReferenceCount
	{
		int sharedCount = 0;
		int weakCount = 0;
		std::mutex	Mutex;
	};



	template< typename T >
	class SharedPtr
	{
	public:

		friend class WeakPtr<T>;


		// Default constructor
		SharedPtr()
			: m_Ptr( nullptr )
			, m_pCount( new ReferenceCount() )
		{
		}


		// Constructor
		SharedPtr( T* ptr, const std::function<void(T*)> &del=(DefaultDeleter<T>()) )
			: m_Ptr( ptr )
			, m_pCount( new ReferenceCount() )
			, m_Deleter( del )
		{
			IncShared();//m_pCount->sharedCount = 1;
		}
		

		// Constructor
		template< typename D >
		SharedPtr( UniquePtr<T, D>&& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( new ReferenceCount() )
			, m_Deleter( obj.m_Deleter )
		{
			if( obj.m_Ptr != nullptr )
			{
				obj.m_Ptr	= nullptr;
				IncShared();//m_pCount->sharedCount = 1;
			}
		}


		// Constructor
		SharedPtr( const WeakPtr<T>& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
		{
			if( obj.Expired() )
				m_Ptr = nullptr;
			else
				IncShared();//++m_pCount->sharedCount;
		}


		// Copy constructor
		SharedPtr( const SharedPtr& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
			, m_Deleter( obj.m_Deleter )
		{
			IncShared();//++m_pCount->sharedCount;
		}


		// Move constructor
		SharedPtr( SharedPtr&& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
			, m_Deleter( std::move(obj.m_Deleter) )
		{
			obj.m_Ptr		= nullptr;
			obj.m_pCount	= nullptr;
		}


		// Destructor
		~SharedPtr()
		{
			Release();
		}


		// Copy assingment operator
		SharedPtr& operator=( const SharedPtr& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr		= obj.m_Ptr;
				m_pCount	= obj.m_pCount;
				m_Deleter	= obj.m_Deleter;

				IncShared();//++m_pCount->sharedCount;
			}

			return *this;
		}
		

		// Move assingment operator
		SharedPtr& operator=(  SharedPtr&& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr			= obj.m_Ptr;
				m_pCount		= obj.m_pCount;
				m_Deleter		= std::move( obj.m_Deleter );

				obj.m_Ptr		= nullptr;
				obj.m_pCount	= nullptr;
			}

			return *this;
		}


		bool operator==( const SharedPtr& obj ) const
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


		T* Get() const
		{
			return m_Ptr;
		}


		int UseCount() const
		{
			if( m_pCount != nullptr )
				return m_pCount->sharedCount;
			else
				return 0;
		}


		void Swap( SharedPtr& obj )
		{
			ReferenceCount* pcount	= obj.m_pCount;
			obj.m_pCount	= m_pCount;
			m_pCount		= pcount;

			T* ptr		= obj.m_Ptr;
			obj.m_Ptr	= m_Ptr;
			m_Ptr		= ptr;
		}


		void Reset()
		{
			Release();

			m_Ptr		= nullptr;
			m_pCount	= nullptr;
		}



	private:

		T*							m_Ptr;
		ReferenceCount*				m_pCount;
		std::function< void(T*) >	m_Deleter;


		void IncShared()
		{
			assert( m_pCount!=nullptr );

			std::lock_guard<std::mutex>	lock( m_pCount->Mutex );
			++m_pCount->sharedCount;
		}


		//void Release()
		//{
		//	if( m_pCount != nullptr && --m_pCount->sharedCount <= 0 )
		//	{
		//		m_Deleter( m_Ptr );
		//		
		//		if( m_pCount->weakCount <= 0 )
		//			SafeDelete( m_pCount );
		//	}
		//}

		void Release()
		{
			if( m_pCount==nullptr ) return;

			bool bDeletePtr = false;
			bool bDeleteCounter = false;

			{// lock
				std::lock_guard<std::mutex> lock(m_pCount->Mutex);

				if( --m_pCount->sharedCount <= 0 )
				{
					bDeletePtr = true;
				
					if( m_pCount->weakCount <= 0 )
						bDeleteCounter = true;
				}
			}// unlock

			if( bDeletePtr )
				m_Deleter( m_Ptr );

			if( bDeleteCounter )
				SafeDelete( m_pCount );
		}


	};



	
	template< typename T >
	class SharedPtr< T[] >
	{
	public:

		friend class WeakPtr<T[]>;


		// Default constructor
		SharedPtr()
			: m_Ptr( nullptr )
			, m_pCount( new ReferenceCount() )
		{
		}


		// Constructor
		SharedPtr( T* ptr, const std::function<void(T*)>& del=(DefaultDeleter<T[]>()) )
			: m_Ptr( ptr )
			, m_pCount( new ReferenceCount() )
			, m_Deleter( del )
		{
			IncShared();//m_pCount->sharedCount = 1;
		}
		

		// Constructor
		template< typename D >
		SharedPtr( UniquePtr<T[], D>&& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( new ReferenceCount() )
			, m_Deleter( obj.m_Deleter )
		{
			if( obj.m_Ptr != nullptr )
			{
				obj.m_Ptr	= nullptr;
				IncShared();//m_pCount->sharedCount = 1;
			}
		}


		// Constructor
		SharedPtr( const WeakPtr<T[]>& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
		{
			if( obj.Expired() )
				m_Ptr = nullptr;
			else
				IncShared();//++m_pCount->sharedCount;
		}


		// Copy constructor
		SharedPtr( const SharedPtr& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
			, m_Deleter( obj.m_Deleter )
		{
			IncShared();//++m_pCount->sharedCount;
		}


		// Move constructor
		SharedPtr( SharedPtr&& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
			, m_Deleter( std::move(obj.m_Deleter) )
		{
			obj.m_Ptr		= nullptr;
			obj.m_pCount	= nullptr;
		}


		// Destructor
		~SharedPtr()
		{
			Release();
		}


		// Copy assignment operator
		SharedPtr& operator=( const SharedPtr& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr		= obj.m_Ptr;
				m_pCount	= obj.m_pCount;
				m_Deleter	= obj.m_Deleter;

				IncShared();//++m_pCount->sharedCount;
			}
			return *this;
		}


		// Move assingment operator
		SharedPtr& operator=( const SharedPtr&& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr			= obj.m_Ptr;
				m_pCount		= obj.m_pCount;
				m_Deleter		= std::move( obj.m_Deleter );

				obj.m_Ptr		= nullptr;
				obj.m_pCount	= nullptr;
			}

			return *this;
		}


		bool operator==( const SharedPtr& obj ) const
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


		T* Get() const
		{
			return m_Ptr;
		}


		int UseCount() const
		{
			if( m_pCount != nullptr )
				return m_pCount->sharedCount;
			else
				return 0;
		}


		void Swap( SharedPtr& obj )
		{
			ReferenceCount* pcount	= obj.m_pCount;
			obj.m_pCount	= m_pCount;
			m_pCount		= pcount;

			T* ptr		= obj.m_Ptr;
			obj.m_Ptr	= m_Ptr;
			m_Ptr		= ptr;
		}


		void Reset()
		{
			Release();

			m_Ptr		= nullptr;
			m_pCount	= nullptr;
		}



	private:

		T*							m_Ptr;
		ReferenceCount*				m_pCount;
		std::function< void(T*) >	m_Deleter;


		void IncShared()
		{
			assert( m_pCount!=nullptr );

			std::lock_guard<std::mutex>	lock( m_pCount->Mutex );
			++m_pCount->sharedCount;
		}


		//void Release()
		//{
		//	if( m_pCount != nullptr && --m_pCount->sharedCount <= 0 )
		//	{
		//		m_Deleter( m_Ptr );
		//		
		//		if( m_pCount->weakCount <= 0 )
		//			SafeDelete( m_pCount );
		//	}
		//}


		void Release()
		{
			if( m_pCount==nullptr ) return;

			bool bDeletePtr = false;
			bool bDeleteCounter = false;

			{// lock
				std::lock_guard<std::mutex> lock(m_pCount->Mutex);

				if( --m_pCount->sharedCount <= 0 )
				{
					bDeletePtr = true;
				
					if( m_pCount->weakCount <= 0 )
						bDeleteCounter = true;
				}
			}// unlock

			if( bDeletePtr )
				m_Deleter( m_Ptr );

			if( bDeleteCounter )
				SafeDelete( m_pCount );
		}


	};
	



}// end of namespace


#endif // !SMART_PTR_H
