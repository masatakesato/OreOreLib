#include <crtdbg.h>
#include	<iostream>

#ifdef _DEBUG
#undef	_DEBUG
#include	<Python.h>
#include	<../Lib/site-packages/numpy/core/include/numpy/arrayobject.h>
#define	_DEBUG
#else
#include	<Python.h>
#include	<../Lib/site-packages/numpy/core/include/numpy/arrayobject.h>
#endif // _DEBUG




// https://gist.github.com/KobayashiRui/e2598d6cbee95897561f9a4df41d8033
// https://tomosoft.jp/design/?p=8818 <- これ参考にしながらPyObjectの参照カウンタ調整方法を勉強する
// https://stackoverflow.com/questions/51236885/python-capi-reference-count-details <- 参照のオーナーシップの話


// 変数のオーナーシップが移動するかどうか関数ごとに確認が必要
// タプルに値セットする場合: 変数のオーナーシップがタプルへ移動
// リストに値セットする場合: オーナーシップ残されたまま
// numpy配列は?


int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	PyObject *pName, *pModule, *pResult, *pFunc;

	Py_Initialize();
	

	// Set system paths
	PyObject *sys = PyImport_ImportModule("sys");
	PyObject *path = PyObject_GetAttrString( sys, "path" );
	PyList_Append( path, PyUnicode_DecodeFSDefault(".") );// current directory( *.vsproj directory)
	PyList_Append( path, PyUnicode_DecodeFSDefault("../../../test/testCallPython") );// source file directory


	pName = PyUnicode_DecodeFSDefault( "py_test1" );
	pModule = PyImport_Import( pName );
	Py_DECREF( pName );

	if( pModule != NULL )
	{

		// call func1
		{
			int func_output;// Variable to store output from python

			// call func
			pFunc = PyObject_GetAttrString( pModule, "func" );

			pResult = PyObject_CallObject( pFunc, NULL );// pResult refcount incremented
			PyArg_Parse( pResult, "i", &func_output );
			std::cout << func_output << std::endl;
			Py_XDECREF( pResult );// Decrement pResult refcount
			Py_XDECREF( pFunc );// Decrement pFunc reference counter
		}

// https://stackoverflow.com/questions/24185199/reference-counting-for-c-function


		// call func2
		//while(1)
		{
			// Get func2 refernce object
			auto pFunc2 = PyObject_GetAttrString( pModule, "func2" );// func2オブジェクトの参照カウンタがインクリメントされる

			// Setup in/out params
			double func2_input = 555.5;
			double func2_output;

			// Setup tuple argument(s)
			auto pArgs = PyTuple_New(1);
			auto pFloatValue = PyFloat_FromDouble( func2_input );
			PyTuple_SetItem( pArgs, 0, pFloatValue );// pFloatValueの所有権を奪う. pArgs削除と同時にpFloatValueもメモリ上から消える

			// Execute func2 and get result
			pResult = PyObject_CallObject( pFunc2, pArgs );
			PyArg_Parse( pResult, "d", &func2_output );
			std::cout << func2_output << std::endl;
			Py_XDECREF( pArgs );

			// Decrement reference counter(s) for unused python GC
			//Py_XDECREF( pFloatValue );// pFloatの所有権はpArgsタプルに移ってるので参照カウンタデクリメントは不要
			Py_XDECREF( pResult );// Decrement pResult refcount
			Py_XDECREF( pFunc2 );// Decrement pFunc2 reference counter
		}


		// call func3 https://stackoverflow.com/questions/30388170/sending-a-c-array-to-python-and-back-extending-c-with-numpy
		{
			// Get func3 refernce object
			auto pFunc3 = PyObject_GetAttrString( pModule, "func3" );
			//pFunc3 = PyObject_GetAttrString( pModule, "func3" );// 変数代入なしで参照受け取るのはマズい. 参照カウンタだけインクリメントされる

			// Setup in/out params
			npy_intp dims[1] = { 4 };
			float Array [] = { 1.2f, 3.4f, 5.6f, 7.8f };
			float *ptr = Array;

			// Setup tuple argument(s)
			auto pArgsArr = PyTuple_New(1);
			import_array();
			auto py_array = PyArray_SimpleNewFromData( 1, dims, NPY_FLOAT, ptr );
			PyTuple_SetItem( pArgsArr,0, py_array );// move py_array ownership to pArgsArr

			// Execute func3 and get result
			pResult = PyObject_CallObject( pFunc3, pArgsArr );
			Py_XDECREF( pArgsArr );

			// Decrement reference counter(s) for unused python GC
			Py_XDECREF( pResult );// Decrement pResult refcount
			Py_XDECREF( pFunc3 );//std::cout << pFunc3->ob_refcnt << std::endl;
		}



	}

	Py_Finalize();


	return 0;
}