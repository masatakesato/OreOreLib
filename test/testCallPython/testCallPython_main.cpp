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



int main()
{
	PyObject *pName, *pModule, *pTmp, *pFunc;
	char *sTmp;
	int func_output;// Variable to store output from python
	double func2_input = 555.5;
	double func2_output;

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
		// call func
		pFunc = PyObject_GetAttrString( pModule, "func" );

		pTmp = PyObject_CallObject( pFunc, NULL );
		PyArg_Parse( pTmp, "i", &func_output );
		std::cout << func_output << std::endl;


		// call func2
		auto pFunc2 = PyObject_GetAttrString( pModule, "func2" );
		auto pArgs = PyTuple_New(1);
		auto pFloatValue = PyFloat_FromDouble( func2_input );
		PyTuple_SetItem( pArgs, 0, pFloatValue );

		pTmp = PyObject_CallObject( pFunc2, pArgs );
		PyArg_Parse( pTmp, "d", &func2_output );
		std::cout << func2_output << std::endl;


		// call func3 https://stackoverflow.com/questions/30388170/sending-a-c-array-to-python-and-back-extending-c-with-numpy
		auto pArgsArr = PyTuple_New(1);

		npy_intp dims[1] = { 4 };
		float Array [] = {1.2, 3.4, 5.6, 7.8};
		float *ptr = Array;
		import_array();
		auto py_array = PyArray_SimpleNewFromData( 1, dims, NPY_FLOAT, ptr );

	}

	Py_Finalize();


	return 0;
}