#include	<thread>
#include	<future>


// https://yohhoy.hatenadiary.jp/entry/20120131/p1


void func( std::promise<double> p, double x )
{
	try
	{
		double ret = x*2.0;
		p.set_value(ret);
	}
	catch( ... )
	{
		p.set_exception( std::current_exception() );
	}

}





int main()
{
	std::promise<double> p;
	std::future<double> f = p.get_future();


	double x = 3.14159;
	std::thread th( func, std::move(p), x );



	try
	{
		double result = f.get();  // (3a) promiseに設定された値を取得（別スレッドでの処理完了を待機）
	}
	catch (...)
	{
		// (3b) promiseに設定された例外が再throwされる
	}


	th.join();

	return 0;
}