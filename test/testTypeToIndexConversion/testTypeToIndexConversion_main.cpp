#include	<oreore/common/Utility.h>
#include	<oreore/memory/DebugNew.h>




template < typename T >
struct TypeID
{
	static const uint32 value;
};

template < typename T >
const uint32 TypeID<T>::value = detail::SeqID::Generate();


namespace detail
{
	class SeqID
	{
	private:

		static uint32 Generate()
		{
			static uint32 counter = 0;
			return counter++;
		}

		template < typename T >
		friend struct TypeID;
	};

}





int main()
{
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	tcout << TypeID<int>::value << tendl;
	tcout << TypeID<float>::value << tendl;
	tcout << TypeID<double>::value << tendl;
	tcout << TypeID<int>::value << tendl;

	return 0;
}