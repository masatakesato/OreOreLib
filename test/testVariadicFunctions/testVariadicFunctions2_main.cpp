//#include <stdarg.h>
//#include <iostream>
//#include <string>
//
//
//
//
////##################################################################################//
////																					//
////								Function implementation								//
////																					//
////##################################################################################//
//
//void func_impl( int a, char c, const std::string& stt )
//{
//	std::cout << "func_impl( int a, char c, const std::string& stt )..." << std::endl;
//
//	std::cout << a << std::endl;
//	std::cout << c << std::endl;
//	std::cout << stt << std::endl;
//
//}
//
//
//
//void func_impl( int a, int b, int c )
//{
//	std::cout << "func_impl( int a, int b, int c )..." << std::endl;
//	std::cout << a + b + c << std::endl;
//}
//
//
//
////##################################################################################//
////																					//
////								Variadic function									//
////																					//
////##################################################################################//
//
//template <class... Args>
//void f( Args... args )
//{
//  func_impl( args... );
//}
//
//
//
//
//
//// CRTP(Curiously Recurring Template Pattern)
//template< typename Derived >
//class IEvaluator
//{
//public:
//
//	template< class... Args >
//	void Evaluate( Args... args )
//	{
//		std::cout << "IEvaluator::Execute()..." << std::endl;
//		static_cast<Derived*>(this)->Evaluate( args... );
//	}
//
//};
//
//
//
//
//
//class MyEvaluator : public IEvaluator<MyEvaluator>
//{
//public:
//
//	void Evaluate( int a, char c, const std::string& stt )
//	{
//		std::cout << "Evaluator::MyFunc()..." << std::endl;
//
//		std::cout << a << std::endl;
//		std::cout << c << std::endl;
//		std::cout << stt << std::endl;
//	}
//
//};
//
//
////https://stackoverflow.com/questions/36797770/get-function-parameters-count
//// 関数の引数の数を取得する
//template < typename FUNC, typename... Args >
//constexpr size_t GetArgumentCount( FUNC( *f )( Args... ) )
//{
//	return sizeof ... (Args);
//}
//
//
//
//void p(){ }
//void p( int ){}
//
//
//int main()
//{
//	std::string	str1( "Hello" ), str2( "world" );
//
//	f( 1, 'a', str1 );
//
//	f( 3, 4, 5 );
//
//
//	IEvaluator<MyEvaluator> *eval = new MyEvaluator();
//
//	eval->Evaluate( 1, 'a', str1 );
//
//	void (* func)() = p;//[](){};
//
//
//	std::cout << GetArgumentCount( func ) << std::endl;
//
////	MyEvaluator eval;
//
////	eval.Evaluate( 1, 'a', str1 );
//
//
//
//}



// https://stackoverflow.com/questions/62811132/build-variadic-tuple-from-array-to-return
// TODO: 通常の配列からは作れない?

#include	<iostream>
#include	<tuple>
#include	<array>
#include	<vector>


template < typename T, size_t N, size_t... IDXs >
auto _array_to_tuple( const std::array<T, N>& a, std::index_sequence<IDXs...> )
{
    return std::make_tuple( a[IDXs]... );//std::tuple{ a[IDXs]... };
}


template < typename T, size_t N >
auto array_to_tuple( const std::array<T, N>& a )
{
    return _array_to_tuple( a, std::make_index_sequence<N>{} );
}


template<class... Args>
auto arrays_to_tuple(Args&&... args)
{
    return std::tuple_cat(array_to_tuple(args)...);
}




//####################################### vector to tuple ? ######################################//

//template < typename T,  size_t... IDXs, size_t N >
//constexpr auto _vec_to_tuple( const std::vector<T>& vec, std::index_sequence<IDXs...> )
//{
//    return std::make_tuple( vec[IDXs]... );//std::tuple{ a[IDXs]... };
//}
//
//
//template < typename T, size_t N >
//constexpr auto vec_to_tuple( const std::vector<T>& vec )
//{
//    return _vec_to_tuple( vec, std::make_index_sequence<N>{} );
//}


//template< class... Args >
//auto vecs_to_tuple( Args&&... args )
//{
//    return std::tuple_cat( vec_to_tuple(args)... );
//}

//################################################################################################//




//int main()
//{
//    //std::array<int, 1> i{1};
//    //std::array<float, 2> f{0.1f, 0.2f};
//    //std::array<short, 3> s{4,5,6};
//
//    //std::tuple<int> t1 = arrays_to_tuple(i);
//    //std::tuple<int, float, float> t2 = arrays_to_tuple(i, f);
//    //std::tuple<int, float, float, short, short, short> t3 =  arrays_to_tuple(i, f, s);
//
//
//    std::vector<int> i{1};
//    std::vector<float> f{0.1f, 0.2f};
//    std::vector<short> s{4,5,6};
//
////   std::tuple<int> t1 = vecs_to_tuple(i);
////   std::tuple<int, float, float> t2 = vecs_to_tuple(i, f);
////   std::tuple<int, float, float, short, short, short> t3 =  vecs_to_tuple(i, f, s);
//
//
//	/*std::tuple<int>*/auto t1 = vec_to_tuple<int, 1>(i);
//
//	return 0;
//}








#include<functional>

template < typename T, std::size_t... I, std::size_t N >
constexpr auto f( const T (&arr)[N], std::index_sequence<I...> )
{
    return std::make_tuple( arr[I]... );
}


template< typename T, std::size_t N >
constexpr auto f( const T (&arr)[N] )
{
    return f( arr, std::make_index_sequence<N>{} );
}


template< class... Args >
constexpr auto vecs_to_tuple( Args&&... args )
{
    return std::tuple_cat( f(args)... );
}



int main()
{
    float float_array[]	= { 0.0f, 1.0f, 2.0f };
	int   int_array[]	= { 3, 4, 5 };
	int* dynamic_array = new int[2];

	dynamic_array[0] = 9999;
	dynamic_array[1] = -367;

    auto tup = f( float_array );

	// タプルの要素はこれで書き換え可能 -> 動的ループは無理だけど、テンプレートメタプログラミング使った再帰処理はいける
	constexpr int i=0;
	auto& val = std::get<i>(tup);// 再帰処理に置き換えてパラメータ設定が必要
	val = 999.555f;


	//auto tups = vecs_to_tuple( dynamic_array );
    //static_assert(std::get<0>(tup) == 0, "!");
    //static_assert(std::get<1>(tup) == 1, "!");
    //static_assert(std::get<2>(tup) == 2, "!");
}


// TODO: 