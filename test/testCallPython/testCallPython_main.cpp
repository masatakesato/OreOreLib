#include	<iostream>

#ifdef _DEBUG
#undef	_DEBUG
#include	<Python.h>
#define	_DEBUG
#else
#include	<Python.h>
#endif // _DEBUG


// https://gist.github.com/KobayashiRui/e2598d6cbee95897561f9a4df41d8033



int main()
{
	PyObject *pName, *pModule, *pTmp, *pFunc;
	char *sTmp;
	int data;// Variable to store output from python

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
		pFunc = PyObject_GetAttrString( pModule, "func" );

		pTmp = PyObject_CallObject( pFunc, NULL );

		PyArg_Parse( pTmp, "i", &data );
		std::cout << data << std::endl;
	}
	Py_Finalize();


	return 0;
}