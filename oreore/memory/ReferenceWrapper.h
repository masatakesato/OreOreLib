#ifndef REFERENCE_WRAPPER_H
#define	REFERENCE_WRAPPER_H


namespace OreOreLib
{
	template < typename T >
	class ReferenceWrapper
	{
	public:

		// Default constructor
		ReferenceWrapper()
			: m_Ptr( nullptr )
		{
		}


		// Constructor
		ReferenceWrapper( T& ref )
			: m_Ptr( &ref )
		{
		}


		// Copy constructor
		ReferenceWrapper( const ReferenceWrapper& obj )
			: m_Ptr( obj.m_Ptr )
		{
		}


		// Move constructor
		ReferenceWrapper( ReferenceWrapper&& obj )
			: m_Ptr( &obj.m_Ptr )
		{
			obj.m_Ptr = nullptr;
		}


		// Destructor
		~ReferenceWrapper()
		{
			m_Ptr = nullptr;
		}


		// Copy assingment operator
		ReferenceWrapper& operator=( const ReferenceWrapper& obj )
		{
			if( m_Ptr != obj.m_Ptr )
				m_Ptr = obj.m_Ptr;

			return *this;
		}


		// Move assingment operator
		ReferenceWrapper& operator=( ReferenceWrapper&& obj )
		{
			if( m_Ptr != obj.m_Ptr )
			{
				m_Ptr = obj.m_Ptr;
				obj.m_Ptr = nullptr;
			}

			return *this;
		}


		bool operator==( const ReferenceWrapper& obj ) const
		{
			return m_Ptr == obj.m_Ptr;
		}


		operator T&() const
		{
			return *m_Ptr;
		}


		T& Ref() const
		{
			return *m_Ptr;
		}


		T* Ptr() const
		{
			return m_Ptr;
		}



	private:

		T*	m_Ptr;
	};


}// end of namespace OreOreLib


#endif // !REFERENCE_WRAPPER_H
