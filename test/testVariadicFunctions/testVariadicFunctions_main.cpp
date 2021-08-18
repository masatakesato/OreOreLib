#include <stdarg.h>
#include <iostream>
#include <string>





void g( int a, char c, const std::string& stt )
{
	std::cout << "Func g()..." << std::endl;

	std::cout << a << std::endl;
	std::cout << c << std::endl;
	std::cout << stt << std::endl;

}



template <class... Args>
void f( Args... args )
{
  g( args... );
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




int main()
{
	std::string	str1( "Hello" ), str2( "world" );

	//f( 1, 'a', str1 );


	MyEvaluator eval;


	eval.Evaluate( 1, 'a', str1 );



}
