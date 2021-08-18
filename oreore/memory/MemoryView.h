#ifndef MEMORY_VIEW_H
#define	MEMORY_VIEW_H

#include	"../common/TString.h"

#include	"Memory.h"




namespace OreOreLib
{

	// https://www.codeproject.com/Articles/848746/ArrayView-StringView

	template< typename T >
	class MemoryView : public Memory<T>
	{
		using Ptr = const T*;

	public:

		MemoryView()
			: Memory()
		{
		}


		MemoryView( const T* const pdata, int length )
		{
			Init( pdata, length );
		}


		~MemoryView()
		{
			Release();
		}


		// copy constructor
		MemoryView( const MemoryView& obj )
		{
			this->m_pData		= obj.m_pData;
			this->m_Length		= obj.m_Length;
			this->m_AllocSize	= 0;
		}


		void Init( const T* const pdata, int length )
		{
			this->m_pData		= (T*)pdata;
			this->m_Length		= length;
			this->m_AllocSize	= 0;
		}


		void Release()
		{
			this->m_pData = nullptr;
			this->m_Length = 0;
		}



		//// Subscription operator for read only.( called if MemoryView is )
		//inline const T& operator[]( std::size_t n ) const&
		//{
		//	return m_refData[n];
		//}


		//// Subscription operator for read-write.( called if MemoryView if non-const )
		//inline T& operator[]( std::size_t n ) &
		//{
		//	return const_cast<T&>(m_refData[n]);
		//}


		//// Subscription operator. ( called by following cases: "T& a = MemoryView(data,10)[n]", "auto&& a = MemoryView(data,20)[n]" )
		//inline T operator[]( std::size_t n ) const&&
		//{
		//	return std::move(m_refData[n]);
		//}



		void Display() const
		{
			tcout << typeid(*this).name() << _T("[ ") << this->m_Length << _T(" ]:\n" );

			for( int i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}
		


	private:

		bool Extend( int ) = delete;//using Memory<T>::Extend;
		bool Resize( int ) = delete;//using Memory<T>::Resize;
		bool Shrink( int ) = delete;//using Memory<T>::Shrink;


	};


}// end of namespace


#endif // !ARRAY_VIEW_H
