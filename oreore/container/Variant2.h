#ifndef VARIANT2_H
#define	VARIANT2_H

#include	"../common/Utility.h"

#include	<typeinfo>

namespace OreOreLib
{

	class Variant2;


	namespace detail
	{

		//##################################################################################//
		//																					//
		//				Support classes(IValManager/ValManager) implementation				//
		//																					//
		//##################################################################################//

		class IValManager
		{
		public:

			const std::type_info& TypeInfo;

			IValManager( const std::type_info& info ) : TypeInfo( info ){}

			virtual ~IValManager() {}
			virtual void SetObjectPointer( void* pobj ) = 0;
			virtual void* CreateObject() const = 0;
			virtual void ReleaseObject() = 0;
			virtual IValManager* Clone() const = 0;
			virtual size_t Size() const = 0;

		};



		template< class T >
		class ValManager : public IValManager
		{
			// Constructor
			ValManager( T* pobj )
				: IValManager( typeid(T) )
				, m_refValue( pobj )
			{
				//tcout << _T( "ValManager constructor...\n" );
			}


			// Copy constructor
			ValManager( const ValManager &obj )
				: IValManager( obj.TypeInfo )
				, m_refValue( obj.m_refValue )
			{
				//tcout << _T( "ValManager copy constructoor...\n" );
			}


			// Destructor
			virtual ~ValManager()
			{
				//tcout << _T( "ValManager destructor...\n" );
				SafeDelete( m_refValue );
			}


			virtual ValManager* Clone() const
			{
				//tcout << _T( "ValManager create instance...\n" );
				return new ValManager( nullptr );
			}


			virtual void* CreateObject() const
			{
				return new T( *m_refValue );
			}


			virtual void ReleaseObject()
			{
				SafeDelete( m_refValue );
			}


			virtual void SetObjectPointer( void* pobj )
			{
				m_refValue = (T*)pobj;
			}


			virtual size_t Size() const
			{
				return sizeof( T );
			}


			T*	m_refValue;



			friend class Variant2;

		};


	}// end of namespace detail




	//##################################################################################//
	//																					//
	//							Variant class implementation							//
	//																					//
	//##################################################################################//

	class Variant2
	{
	public:

		// Default constructor
		Variant2()
			: m_pValue( nullptr )
			, m_pManager( nullptr )
		{
			//tcout << _T( "Variant2 default constructor...\n" );
		}


		// Constructor
		template < typename T >
		Variant2( /*const*/ T& val )
			: m_pValue( new T( val ) )
			, m_pManager( new detail::ValManager<T>( (T*)m_pValue ) )
		{
			//tcout << _T( "Variant2 constructor...\n" );
		}


		// Constructor
		template < typename T >
		Variant2( T&& val )
			: m_pValue( new T( (T&&)val ) )
			, m_pManager( new detail::ValManager<T>( (T*)m_pValue ) )
		{
			//tcout << _T( "Variant2 constructor...\n" );
		}


		// Constructor
		template < typename T >
		Variant2( T* val )
			: m_pValue( val )
			, m_pManager( nullptr )
		{
			//tcout << _T( "Variant2 constructor...\n" );
		}


		// Copy constructor
		Variant2( const Variant2& obj )
		{
			//tcout << _T( "Variant2 copy constructor...\n" );

			m_pValue	= obj.m_pManager->CreateObject();

			m_pManager = obj.m_pManager->Clone();
			m_pManager->SetObjectPointer( m_pValue );
		}


		// Move constructor
		Variant2( Variant2&& obj )
			: m_pValue( obj.m_pValue )
			, m_pManager( obj.m_pManager )
		{
			//tcout << _T( "Variant2 move constructor...\n" );

			obj.m_pValue	= nullptr;
			obj.m_pManager	= nullptr;
		}


		// Destructor
		~Variant2()
		{
			//tcout << _T( "Variant2 destructor...\n" );
			SafeDelete( m_pManager );// m_pManager destructs m_pValue pointer
		}


		//======================================== Assignment Operators ========================================//

		// Copy assignment operator for Variant2
		inline Variant2& operator =( const Variant2& obj )
		{
			if( this != &obj )
			{
				//tcout << _T( "Variant2 copy assignment operator...\n" );

				SafeDelete( m_pManager );

				m_pValue	= obj.m_pManager->CreateObject();
				m_pManager	= obj.m_pManager->Clone();
				m_pManager->SetObjectPointer( m_pValue );
			}

			return *this;
		}


		// Move asignment operator for Variant2
		inline Variant2& operator =( Variant2&& obj )
		{
			if( this != &obj )
			{
				//tcout << _T( "Variant2 move assignment operator...\n" );
				SafeDelete( m_pManager );

				m_pValue	= obj.m_pValue;
				m_pManager	= obj.m_pManager;

				obj.m_pValue	= nullptr;
				obj.m_pManager	= nullptr;
			}

			return *this;
		}


		// Copy assignment operator for other types
		template< typename T >
		inline Variant2& operator =( T& obj )
		{

			if( m_pValue != &obj )
			{
				if( !m_pManager )
				{
					//tcout << _T( "   Variant2 is null. allocating...\n" );
					m_pValue	= new T( obj );
					m_pManager	= new detail::ValManager<T>( (T*)m_pValue );
				}
				else if( typeid( T ) != m_pManager->TypeInfo )
				{
					//tcout << _T( "   Variant2 type changed. reallocating...\n" );
					SafeDelete( m_pManager );
					m_pValue	= new T( obj );
					m_pManager	= new detail::ValManager<T>( (T*)m_pValue );
				}
				else
				{
					//tcout << _T( "   Already allocated same type. moving data...\n" );
					m_pManager->ReleaseObject();
					m_pValue = new T( obj );
					m_pManager->SetObjectPointer( m_pValue );
				}
			}

			return *this;
		}


		template< typename T >
		inline Variant2& operator =( T&& obj )
		{
			if( m_pValue != &obj )
			{	
				if( !m_pManager )
				{
					//tcout << _T( "   Variant2 is null. allocating...\n" );
					m_pValue	= new T( (T&&)obj );
					m_pManager	= new detail::ValManager<T>( (T*)m_pValue );
				}
				else if( typeid( T ) != m_pManager->TypeInfo )
				{
					//tcout << _T( "   Variant2 type changed. reallocating...\n" );
					SafeDelete( m_pManager );
					m_pValue	= new T( (T&&)obj );
					m_pManager	= new detail::ValManager<T>( (T*)m_pValue );
				}
				else
				{
					//tcout << _T( "   Already allocated same type. moving data...\n" );
					*(T*)m_pValue = (T&&)obj;
					m_pManager->SetObjectPointer( m_pValue );
				}
			}

			return *this;
		}


		// Copy assignment operator for other types
		template< typename T >
		inline Variant2& operator =( T* obj )
		{
			if( m_pValue != obj )
			{
				SafeDelete( m_pManager );
				m_pValue	= obj;
				m_pManager	= nullptr;
			}

			return *this;
		}


		//======================================== Cast Operators ========================================//

		template < typename T >
		inline constexpr operator T&() const
		{
			ASSERT( typeid(T) == ( m_pManager ? m_pManager->TypeInfo : typeid(T) ) );
			return *(T*)m_pValue;
		}


		template < typename T >
		inline constexpr operator T*&()
		{
			//tcout << typeid(T*).hash_code() << tendl;
			//tcout <<  m_pManager->TypeInfo.hash_code() << tendl;
			ASSERT( typeid(T*&) == ( m_pManager ? m_pManager->TypeInfo : typeid(T*&) ) );
			return (T*&)m_pValue;//*(T*&)m_pValue;
		}

		
		//template < typename T >
		//inline constexpr operator T*() const
		//{
		//	//tcout << typeid(T*).hash_code() << tendl;
		//	//tcout <<  m_pManager->TypeInfo.hash_code() << tendl;
		//	ASSERT( typeid(T*) == ( m_pManager ? m_pManager->TypeInfo : typeid(T*) ) );
		//	return *(T**)m_pValue;
		//}



		//====================================== Indirection Operators ===================================//

		// non-const object
		template < typename T >
		inline T& operator *()
		{
			//tcout << "T& operator*();\n";
			return  *(T*)m_pValue;
		}


		// const object
		template < typename T >
		inline const T& operator *() const
		{
			//tcout << "const T& operator*() const;\n";
			return *(T*)m_pValue;
		}


		//======================================== Comparison Operators ========================================//
		
		template < typename T >
		inline bool operator ==( const T& rhs ) const
		{
			return *(T*)m_pValue == rhs;
		}


		inline bool operator ==( const Variant2& rhs ) const
		{
			return size_t(m_pValue) == size_t(rhs.m_pValue);
		}


		template < typename T >
		inline bool operator !=( const T& rhs ) const
		{
			return *(T*)m_pValue != rhs;
		}


		inline bool operator !=( const Variant2& rhs ) const
		{
			return size_t(m_pValue) != size_t(rhs.m_pValue);
		}


		template < typename T >
		inline bool operator <( const T& rhs ) const
		{
			return *(T*)m_pValue < rhs;
		}


		inline bool operator <( const Variant2& rhs ) const
		{
			return size_t(m_pValue) < size_t(rhs.m_pValue);
		}


		template < typename T >
		inline bool operator >( const T& rhs ) const
		{
			return *(T*)m_pValue > rhs;
		}


		inline bool operator >( const Variant2& rhs ) const
		{
			return size_t(m_pValue) > size_t(rhs.m_pValue);
		}


		template < typename T >
		inline bool operator <=( const T& rhs ) const
		{
			return *(T*)m_pValue <= rhs;
		}


		inline bool operator <=( const Variant2& rhs ) const
		{
			return size_t(m_pValue) <= size_t(rhs.m_pValue);
		}


		template < typename T >
		inline bool operator >=( const T& rhs ) const
		{
			return *(T*)m_pValue >= rhs;
		}


		inline bool operator >=( const Variant2& rhs ) const
		{
			return size_t(m_pValue) >= size_t(rhs.m_pValue);
		}



	private:

		void*					m_pValue;
		detail::IValManager*	m_pManager;





		//################# Deprecated. Variant2& operator=( T& obj ) can handle SharedPtr ##############//
		//template< typename T >
		//inline Variant2& operator=( SharedPtr<T>& obj )
		//{
		//	if( m_pValue != &obj )
		//	{	
		//		if( !m_pManager )
		//		{
		//			//tcout << _T( "   Variant2 is null. allocating...\n" );
		//			m_pValue	= new SharedPtr<T>( std::move( obj ) );
		//			m_pManager	= new detail::ValManager<SharedPtr<T>>( (SharedPtr<T>*)m_pValue );
		//		}
		//		else if( typeid( SharedPtr<T> ) != m_pManager->TypeInfo )
		//		{
		//			//tcout << _T( "   Variant2 type changed. reallocating...\n" );
		//			SafeDelete( m_pManager );
		//			m_pValue	= new SharedPtr<T>( std::move( obj ) );
		//			m_pManager	= new detail::ValManager<SharedPtr<T>>( (SharedPtr<T>*)m_pValue );
		//		}
		//		else
		//		{
		//			//tcout << _T( "   Already allocated same type. moving data...\n" );
		//			m_pManager->ReleaseObject();
		//			m_pValue = new SharedPtr<T>( obj );//*(SharedPtr<T>*)m_pValue = (SharedPtr<T>&&)( obj );// std::move( obj );
		//			m_pManager->SetObjectPointer( m_pValue );//m_pManager	= new detail::ValManager<SharedPtr<T>>( (SharedPtr<T>*)m_pValue );//
		//		}
		//	}

		//	return *this;
		//}


		//################# Deprecated. Variant2& operator=( T&& obj ) can handle SharedPtr #################//
		//template< typename T >
		//inline Variant2& operator=( SharedPtr<T>&& obj )
		//{
		//	if( m_pValue != &obj )
		//	{	
		//		if( !m_pManager )
		//		{
		//			//tcout << _T( "   Variant2 is null. allocating...\n" );
		//			m_pValue	= new SharedPtr<T>( std::move( obj ) );
		//			m_pManager	= new detail::ValManager<SharedPtr<T>>( (SharedPtr<T>*)m_pValue );
		//		}
		//		else if( typeid( SharedPtr<T> ) != m_pManager->TypeInfo )
		//		{
		//			//tcout << _T( "   Variant2 type changed. reallocating...\n" );
		//			SafeDelete( m_pManager );
		//			m_pValue	= new SharedPtr<T>( std::move( obj ) );
		//			m_pManager	= new detail::ValManager<SharedPtr<T>>( (SharedPtr<T>*)m_pValue );
		//		}
		//		else
		//		{
		//			//tcout << _T( "   Already allocated same type. moving data...\n" );
		//			*(SharedPtr<T>*)m_pValue = (SharedPtr<T>&&)obj;
		//			m_pManager->SetObjectPointer( m_pValue );
		//		}
		//	}

		//	return *this;
		//}


		//################# Deprecated. Variant2& operator=( T&& obj ) can handle WeakPtr #################//
		//template< typename T >
		//inline Variant2& operator=( WeakPtr<T>&& obj )
		//{
		//	if( m_pValue != &obj )
		//	{	
		//		if( !m_pManager )
		//		{
		//			//tcout << _T( "   Variant2 is null. allocating...\n" );
		//			m_pValue	= new WeakPtr<T>( obj );
		//			m_pManager	= new detail::ValManager<WeakPtr<T>>( (WeakPtr<T>*)m_pValue );
		//		}
		//		else if( typeid( WeakPtr<T> ) != m_pManager->TypeInfo )
		//		{
		//			//tcout << _T( "   Variant2 type changed. reallocating...\n" );
		//			SafeDelete( m_pManager );
		//			m_pValue	= new WeakPtr<T>( obj );
		//			m_pManager	= new detail::ValManager<WeakPtr<T>>( (WeakPtr<T>*)m_pValue );
		//		}
		//		else
		//		{
		//			//tcout << _T( "   Already allocated same type. moving data...\n" );
		//			*(WeakPtr<T>*)m_pValue = (WeakPtr<T>&)obj;
		//			m_pManager->SetObjectPointer( m_pValue );
		//		}
		//	}

		//	return *this;
		//}


	};


}// end of namespace



#endif // !VARIANT2_H