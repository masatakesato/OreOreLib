#ifndef WEAK_PTR_H
#define	WEAK_PTR_H

// https://programmersought.com/article/67783265796/
//


#include	<atomic>

#include	"SharedPtr.h"



namespace OreOreLib
{

	template< typename T >
	class WeakPtr
	{
	public:

		friend class SharedPtr<T>;


		// Default constructor
		WeakPtr()
			: m_Ptr( nullptr )
			, m_pCount( nullptr )
		{
		}


		// Constructor
		WeakPtr( const SharedPtr<T>& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
		{
			++m_pCount->weakCount;
		}


		// Copy constructor
		WeakPtr( const WeakPtr& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
		{
			++m_pCount->weakCount;
		}


		// Move constructor
		WeakPtr( WeakPtr&& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
		{
			obj.m_Ptr		= nullptr;
			obj.m_pCount	= nullptr;
		}


		// Destructor
		~WeakPtr()
		{
			Release();
		}


		// Copy assingment operator
		WeakPtr& operator=( const WeakPtr& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr		= obj.m_Ptr;
				m_pCount	= obj.m_pCount;

				if( m_pCount!=nullptr )	++m_pCount->weakCount;
			}

			return *this;
		}


		// Copy assignment operator
		WeakPtr& operator=( const SharedPtr<T>& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr		= obj.m_Ptr;
				m_pCount	= obj.m_pCount;

				if( m_pCount!=nullptr )	++m_pCount->weakCount;
			}

			return *this;
		}


		// Move assingment operator
		WeakPtr& operator=( WeakPtr&& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr			= obj.m_Ptr;
				m_pCount		= obj.m_pCount;

				obj.m_Ptr		= nullptr;
				obj.m_pCount	= nullptr;
			}

			return *this;
		}


		bool operator==( const WeakPtr& obj ) const
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


		bool Expired() const
		{
			if( m_pCount != nullptr && m_pCount->sharedCount > 0 ) return false;

			return true;
		}


		void Swap( WeakPtr& obj )
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
			m_pCount	= nullptr;
			m_Ptr		= nullptr;
		}



	private:

		T*				m_Ptr;
		ReferenceCount*	m_pCount;



		void Release()
		{
			if( m_pCount == nullptr )	return;

			if( --m_pCount->weakCount <= 0 && m_pCount->sharedCount <= 0 )
				SafeDelete( m_pCount );

		}


	};




	template< class T >
	class WeakPtr<T[]>
	{
	public:

		friend class SharedPtr<T[]>;


		// Default constructor
		WeakPtr()
			: m_Ptr( nullptr )
			, m_pCount( nullptr )
		{
		}


		// Constructor
		WeakPtr( const SharedPtr<T[]>& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
		{
			++m_pCount->weakCount;
		}


		// Copy constructor
		WeakPtr( const WeakPtr& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
		{
			++m_pCount->weakCount;
		}


		// Move constructor
		WeakPtr( WeakPtr&& obj )
			: m_Ptr( obj.m_Ptr )
			, m_pCount( obj.m_pCount )
		{
			obj.m_Ptr		= nullptr;
			obj.m_pCount	= nullptr;
		}


		// Destructor
		~WeakPtr()
		{
			Release();
		}


		// Copy assingment operator
		WeakPtr& operator=( const WeakPtr& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr		= obj.m_Ptr;
				m_pCount	= obj.m_pCount;

				if( m_pCount!=nullptr )	++m_pCount->weakCount;
			}

			return *this;
		}


		// Copy assignment operator
		WeakPtr& operator=( const SharedPtr<T[]>& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr		= obj.m_Ptr;
				m_pCount	= obj.m_pCount;

				if( m_pCount!=nullptr )	++m_pCount->weakCount;
			}

			return *this;
		}


		// Move assingment operator
		WeakPtr& operator=( WeakPtr&& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				Release();

				m_Ptr			= obj.m_Ptr;
				m_pCount		= obj.m_pCount;

				obj.m_Ptr		= nullptr;
				obj.m_pCount	= nullptr;
			}

			return *this;
		}



		bool operator==( const WeakPtr& obj ) const
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


		// Disable direct access to row pointer
		//T* Get() const
		//{
		//	return m_Ptr;
		//}


		int UseCount() const
		{
			if( m_pCount != nullptr )
				return m_pCount->sharedCount;
			else
				return 0;
		}


		bool Expired() const
		{
			if( m_pCount != nullptr && m_pCount->sharedCount > 0 ) return false;

			return true;
		}


		void Swap( WeakPtr& obj )
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
			m_pCount	= nullptr;
			m_Ptr		= nullptr;
		}



	private:

		T*				m_Ptr;
		ReferenceCount*	m_pCount;



		void Release()
		{
			if( m_pCount == nullptr )	return;

			if( --m_pCount->weakCount <= 0 && m_pCount->sharedCount <= 0 )
			{
				m_Ptr	= nullptr;
				SafeDelete( m_pCount );
			}

		}


	};



}// end of namespace


#endif // !WEAK_PTR_H
