#ifndef TYPE_LIST_H
#define	TYPE_LIST_H


#include	"../common/TString.h"


// https://www.codetd.com/en/article/6934735  Above C++20
// https://blog.galowicz.de/2016/05/08/compile_time_type_lists/
// https://gist.github.com/DieHertz/8417266



namespace OreOreLib
{
	namespace TypeList
	{

		//######################################################################//
		//																		//
		//						Nested template type lists						//
		//																		//
		//######################################################################//

		struct NullType
		{};


		template <typename T, typename U>
		struct TypeList
		{
			using head = T;
			using tail = U;
		};



		template <typename ... Ts> struct MakeTypeList;


		// Case: Recursion abort, because the list of types ran empty
		template <>
		struct MakeTypeList<>
		{
			using type = NullType;
		};


		// Case: Normal recursion. Consume one type per call
		template < typename T, typename ... REST >
		struct MakeTypeList< T, REST ... >
		{
			using type = TypeList< T, typename MakeTypeList<REST...>::type >;
		};


		template < typename ... Ts >
		using MakeTypeList_t = typename MakeTypeList<Ts...>::type;




		//######################################################################//
		//																		//
		//							Length< TypeList >							//
		//																		//
		//######################################################################//

		template < typename TList > struct Length;


		template <>
		struct Length< NullType >
		{
			static constexpr int value = 0;//enum{ value = 0 };
		};


		template < typename Head, typename Tail >
		struct Length< TypeList<Head, Tail> >
		{
			static constexpr int value = Length<Tail>::value + 1;//enum{ value = Length<Tail>::value + 1 };
		};




		//######################################################################//
		//																		//
		//						IndexOf< TypeList, type >						//
		//																		//
		//######################################################################//

		template < typename TList, typename T > struct IndexOf;


		template < typename T >
		struct IndexOf< NullType, T >
		{
			static constexpr int value = -1;//enum{ value = -1 };
		};


		template < typename T, typename Tail >
		struct IndexOf< TypeList<T, Tail>, T >
		{
			static constexpr int value = 0;//enum{ value = 0 };
		};


		template < typename Head, typename Tail, typename T >
		struct IndexOf< TypeList<Head, Tail>, T >
		{
			using Result = IndexOf<Tail, T>;
			static constexpr int value = Result::value==-1 ? -1 : Result::value + 1;//enum{ value = Result::value==-1 ? -1 : Result::value + 1 };
		};




		//######################################################################//
		//																		//
		//						TypeAt< TypeList, index >						//
		//																		//
		//######################################################################//

		// https://stackoverflow.com/questions/2150892/how-to-build-this-c-typelist-into-a-variant

		template < typename TList, int i > struct TypeAt;


		template < typename Head, typename Tail >
		struct TypeAt< TypeList<Head, Tail>, 0 >
		{
			using type = Head;
		};


		template < typename Head, typename Tail, int i >
		struct TypeAt< TypeList<Head, Tail>, i >
		{
			using type = typename TypeAt< Tail, i-1 >::type;
		};


		template < int i >
		struct TypeAt< NullType, i >
		{
			using type = NullType;
		};



		//######################################################################//
		//																		//
		//						Contains< TypeList, type >						//
		//																		//
		//######################################################################//


		template < typename TList, typename T > struct Contains;


		template < typename TList, typename T > struct Contains
		{
			static constexpr bool value = 
				std::is_same_v< typename TList::head, T >
				|| Contains< typename TList::tail, T >::value;

		};


		template < typename T >
		struct Contains< NullType, T >
		{
			static constexpr bool value = false;
		};




		//######################################################################//
		//																		//
		//							Print< TypeList >							//
		//																		//
		//######################################################################//

		template < typename TList >
		struct Print
		{
			static void print()
			{
				tcout << typeid( typename TList::head ).name() << _T( " " );
				Print< typename TList::tail >::print();
			}
		};


		template <>
		struct Print< NullType >
		{
			static void print()// Do nothing
			{
				tcout << _T( "\n" );
			}
		};





		//######################################################################//
		//																		//
		//								Example Usage							//
		//																		//
		//######################################################################//

		/**

			// Create typelist
			using my_types = MakeTypeList_t< char, short, int, float, double >;

			// Print types
			tcout << _T("my_types...\n");
			Print< my_types >::print();

			// Get Length of my_types
			tcout << "Length = " << Length< my_types >::value << tendl;

			// Get index using specified type
			tcout << _T("IndexOf<char> = ") << IndexOf< my_types, char >::value << tendl;
			tcout << _T("IndexOf<short> = ") << IndexOf< my_types, short >::value << tendl;
			tcout << _T("IndexOf<int> = ") << IndexOf< my_types, int >::value << tendl;
			tcout << _T("IndexOf<float> = ") << IndexOf< my_types, float >::value << tendl;
			tcout << _T("IndexOf<double> = ") << IndexOf< my_types, double >::value << tendl;
			tcout << _T("IndexOf<unsigned int> = ") << IndexOf< my_types, unsigned int >::value << tendl;// querying unregistered type.(returns -1)

			// Get type using specified index
			tcout << _T("TypeAt<0> = ") << typeid( TypeAt< my_types, 0 >::type ).name() << tendl;
			tcout << _T("TypeAt<1> = ") << typeid( TypeAt< my_types, 1 >::type ).name() << tendl;
			tcout << _T("TypeAt<2> = ") << typeid( TypeAt< my_types, 2 >::type ).name() << tendl;
			tcout << _T("TypeAt<3> = ") << typeid( TypeAt< my_types, 3 >::type ).name() << tendl;
			tcout << _T("TypeAt<4> = ") << typeid( TypeAt< my_types, 4 >::type ).name() << tendl;

			// Check if type exists
			tcout << _T("Contains<char> = ") << Contains< my_types, char >::value << tendl;
			tcout << _T("Contains<short> = ") << Contains< my_types, short >::value << tendl;
			tcout << _T("Contains<int> = ") << Contains< my_types, int >::value << tendl;
			tcout << _T("Contains<unsigned int> = ") << Contains< my_types, unsigned int >::value << tendl;


		**/




	}// end of TypeList namespace

}// end of OreOreLib namespace




#endif // !TYPE_LIST_H
