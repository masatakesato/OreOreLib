#ifndef ARRAY_VIEW_H
#define	ARRAY_VIEW_H

#include	"../common/TString.h"
//#include	"Array.h"
//#include	"StaticArray.h"



namespace OreOreLib
{

	// https://www.codeproject.com/Articles/848746/ArrayView-StringView

	template< typename T >
	class ArrayView
	{
		using Ptr = const T*;

	public:

		ArrayView()
			: m_refData( nullptr )
			, m_Length( 0 )
		{
		}


		ArrayView( const Ptr pdata, int length )
			: m_refData( pdata )
			, m_Length( length )
		{
		}


		~ArrayView()
		{
			Release();
		}


		// copy constructor
		ArrayView( const ArrayView& obj )
			: m_refData(obj.pdata)
			, m_Length(obj.m_Length)
		{

		}


		void Init( const Ptr pdata, int length )
		{
			m_refData	= pdata;
			m_Length	= length;
		}


		void Release()
		{
			m_refData = nullptr;
			m_Length = 0;
		}


		void Clear()
		{
			memset( m_refData, 0, sizeof(T) * m_Length );
		}


		// Subscription operator for read only.( called if MemoryView is )
		inline const T& operator[]( std::size_t n ) const&
		{
			return m_refData[n];
		}


		// Subscription operator for read-write.( called if MemoryView if non-const )
		inline T& operator[]( std::size_t n ) &
		{
			return const_cast<T&>(m_refData[n]);
		}


		// Subscription operator. ( called by following cases: "T& a = MemoryView(data,10)[n]", "auto&& a = MemoryView(data,20)[n]" )
		inline T operator[]( std::size_t n ) const&&
		{
			return std::move(m_refData[n]);
		}


		int Length() const
		{
			return m_Length;
		}


		// begin / end overload for "range-based for loop"
		inline T* begin()
		{
			return m_refData;
		}

		inline const T* begin() const
		{
			return m_refData;
		}

		inline T* end()
		{
			return begin() + m_Length;
		}

		inline const T* end() const
		{
			return begin() + m_Length;
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T("[ ") << m_Length << _T(" ]:\n" );

			for( int i=0; i<m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << m_refData[i] << tendl;

			tcout << tendl;
		}



	private:

		Ptr	m_refData;
		int	m_Length;

	};


}// end of namespace


#endif // !ARRAY_VIEW_H
