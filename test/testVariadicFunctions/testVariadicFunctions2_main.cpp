#include <stdarg.h>
#include <iostream>
#include <string>




//##################################################################################//
//																					//
//								Function implementation								//
//																					//
//##################################################################################//

void func_impl( int a, char c, const std::string& stt )
{
	std::cout << "func_impl( int a, char c, const std::string& stt )..." << std::endl;

	std::cout << a << std::endl;
	std::cout << c << std::endl;
	std::cout << stt << std::endl;

}



void func_impl( int a, int b, int c )
{
	std::cout << "func_impl( int a, int b, int c )..." << std::endl;
	std::cout << a + b + c << std::endl;
}



//##################################################################################//
//																					//
//								Variadic function									//
//																					//
//##################################################################################//

template <class... Args>
void f( Args... args )
{
  func_impl( args... );
}





// CRTP(Curiously Recurring Template Pattern)
template< typename Derived >
class IEvaluator
{
public:

	template< class... Args >
	void Evaluate( Args... args )
	{
		std::cout << "IEvaluator::Execute()..." << std::endl;
		static_cast<Derived*>(this)->Evaluate( args... );
	}

};





class MyEvaluator : public IEvaluator<MyEvaluator>
{
public:

	void Evaluate( int a, char c, const std::string& stt )
	{
		std::cout << "Evaluator::MyFunc()..." << std::endl;

		std::cout << a << std::endl;
		std::cout << c << std::endl;
		std::cout << stt << std::endl;
	}

};


//https://stackoverflow.com/questions/36797770/get-function-parameters-count
// 関数の引数の数を取得する
template < typename FUNC, typename... Args >
constexpr size_t GetArgumentCount( FUNC( *f )( Args... ) )
{
	return sizeof ... (Args);
}



void p(){ }
void p( int ){}


int main()
{
	std::string	str1( "Hello" ), str2( "world" );

	f( 1, 'a', str1 );

	f( 3, 4, 5 );


	IEvaluator<MyEvaluator> *eval = new MyEvaluator();

	eval->Evaluate( 1, 'a', str1 );

	void (* func)() = p;//[](){};


	std::cout << GetArgumentCount( func ) << std::endl;

//	MyEvaluator eval;

//	eval.Evaluate( 1, 'a', str1 );



}
