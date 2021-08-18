#ifndef VARIANT2_H
#define	VARIANT2_H

#include	"../common/Utility.h"



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

			virtual ~IValManager() {}
			virtual void SetObjectPointer( void* pobj ) = 0;
			virtual void* CreateObject() const = 0;
			virtual void ReleaseObject() = 0;
			virtual IValManager* Clone() const = 0;
			virtual const std::type_info& GetType() const = 0;// override
			virtual size_t Size() const = 0;

		};



		template< class T >
		class ValManager : public IValManager
		{
			// Constructor
			ValManager( T* pobj )
				: m_refValue( pobj )
			{
				//tcout << _T( "ValManager constructor...\n" );
			}


			// Copy constructor
			ValManager( const ValManager &obj )
				: IValManager( obj )
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


			virtual const std::type_info& GetType() const
			{
				return typeid( T );
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
		template<typename T>
		Variant2( const T& val )
			: m_pValue( new T( val ) )
			, m_pManager( new detail::ValManager<T>( (T*)m_pValue ) )
		{
			//tcout << _T( "Variant2 constructor...\n" );
		}


		// Constructor
		template<typename T>
		Variant2( T&& val )
			: m_pValue( new T( (T&&)val ) )
			, m_pManager( new detail::ValManager<T>( (T*)m_pValue ) )
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


		// 型変換キャスト演算子
		template<typename T>
		inline constexpr operator T&() const
		{
			return *(T*)m_pValue;
		}


		template<typename T>
		inline constexpr operator T*() const
		{
			return (T*)m_pValue;
		}


		// Copy assignment operator for Variant2
		inline Variant2& operator=( const Variant2& obj )
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
		inline Variant2& operator=( Variant2&& obj )
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
		inline Variant2& operator=( T& obj )
		{

			if( m_pValue != &obj )
			{
				if( !m_pManager )
				{
					//tcout << _T( "   Variant2 is null. allocating...\n" );
					m_pValue	= new T( obj );
					m_pManager	= new detail::ValManager<T>( (T*)m_pValue );
				}
				else if( typeid( T ) != m_pManager->GetType() )
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
		inline Variant2& operator=( T&& obj )
		{
			if( m_pValue != &obj )
			{	
				if( !m_pManager )
				{
					//tcout << _T( "   Variant2 is null. allocating...\n" );
					m_pValue	= new T( (T&&)obj );
					m_pManager	= new detail::ValManager<T>( (T*)m_pValue );
				}
				else if( typeid( T ) != m_pManager->GetType() )
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


		// Indirection operator
		template< typename T >
		T& operator*()
		{
			//tcout << "T& operator*();\n";
			return *(T*)m_pValue;
		}


		// Indirection operator( const object )
		template< typename T >
		const T& operator*() const
		{
			//tcout << "const T& operator*() const;\n";
			return *(T*)m_pValue;
		}






// Deprecated. Variant2& operator=( T& obj ) can handle SharedPtr

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
		//		else if( typeid( SharedPtr<T> ) != m_pManager->GetType() )
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


// Deprecated. Variant2& operator=( T&& obj ) can handle SharedPtr
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
		//		else if( typeid( SharedPtr<T> ) != m_pManager->GetType() )
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



// Deprecated. Variant2& operator=( T&& obj ) can handle WeakPtr

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
		//		else if( typeid( WeakPtr<T> ) != m_pManager->GetType() )
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



	private:

		void*					m_pValue;
		detail::IValManager*	m_pManager;

	};


}



#endif // !VARIANT2_H