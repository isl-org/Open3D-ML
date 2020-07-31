#include <Python.h>
#include <numpy/arrayobject.h>
#include "neighbors/neighbors.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module provides two methods to compute radius neighbors from pointclouds or batch of pointclouds";

static char batch_query_docstring[] = "Method to get radius neighbors in a batch of stacked pointclouds";


// Declare the functions
// *********************

static PyObject *batch_neighbors(PyObject *self, PyObject *args, PyObject *keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "batch_query", (PyCFunction)batch_neighbors, METH_VARARGS | METH_KEYWORDS, batch_query_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "radius_neighbors",		// m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_radius_neighbors(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Definition of the batch_subsample method
// **********************************

static PyObject* batch_neighbors(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* queries_obj = NULL;
	PyObject* supports_obj = NULL;
	PyObject* q_batches_obj = NULL;
	PyObject* s_batches_obj = NULL;

	// Keywords containers
	static char* kwlist[] = { "queries", "supports", "q_batches", "s_batches", "radius", NULL };
	float radius = 0.1;

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|$f", kwlist, &queries_obj, &supports_obj, &q_batches_obj, &s_batches_obj, &radius))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}


	// Interpret the input objects as numpy arrays.
	PyObject* queries_array = PyArray_FROM_OTF(queries_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* supports_array = PyArray_FROM_OTF(supports_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* q_batches_array = PyArray_FROM_OTF(q_batches_obj, NPY_INT, NPY_IN_ARRAY);
	PyObject* s_batches_array = PyArray_FROM_OTF(s_batches_obj, NPY_INT, NPY_IN_ARRAY);

	// Verify data was load correctly.
	if (queries_array == NULL)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting query points to numpy arrays of type float32");
		return NULL;
	}
	if (supports_array == NULL)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting support points to numpy arrays of type float32");
		return NULL;
	}
	if (q_batches_array == NULL)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting query batches to numpy arrays of type int32");
		return NULL;
	}
	if (s_batches_array == NULL)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting support batches to numpy arrays of type int32");
		return NULL;
	}

	// Check that the input array respect the dims
	if ((int)PyArray_NDIM(queries_array) != 2 || (int)PyArray_DIM(queries_array, 1) != 3)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : query.shape is not (N, 3)");
		return NULL;
	}
	if ((int)PyArray_NDIM(supports_array) != 2 || (int)PyArray_DIM(supports_array, 1) != 3)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : support.shape is not (N, 3)");
		return NULL;
	}
	if ((int)PyArray_NDIM(q_batches_array) > 1)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : queries_batches.shape is not (B,) ");
		return NULL;
	}
	if ((int)PyArray_NDIM(s_batches_array) > 1)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : supports_batches.shape is not (B,) ");
		return NULL;
	}
	if ((int)PyArray_DIM(q_batches_array, 0) != (int)PyArray_DIM(s_batches_array, 0))
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong number of batch elements: different for queries and supports ");
		return NULL;
	}

	// Number of points
	int Nq = (int)PyArray_DIM(queries_array, 0);
	int Ns= (int)PyArray_DIM(supports_array, 0);

	// Number of batches
	int Nb = (int)PyArray_DIM(q_batches_array, 0);

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> queries;
	vector<PointXYZ> supports;
	vector<int> q_batches;
	vector<int> s_batches;
	queries = vector<PointXYZ>((PointXYZ*)PyArray_DATA(queries_array), (PointXYZ*)PyArray_DATA(queries_array) + Nq);
	supports = vector<PointXYZ>((PointXYZ*)PyArray_DATA(supports_array), (PointXYZ*)PyArray_DATA(supports_array) + Ns);
	q_batches = vector<int>((int*)PyArray_DATA(q_batches_array), (int*)PyArray_DATA(q_batches_array) + Nb);
	s_batches = vector<int>((int*)PyArray_DATA(s_batches_array), (int*)PyArray_DATA(s_batches_array) + Nb);

	// Create result containers
	vector<int> neighbors_indices;

	// Compute results
	//batch_ordered_neighbors(queries, supports, q_batches, s_batches, neighbors_indices, radius);
	batch_nanoflann_neighbors(queries, supports, q_batches, s_batches, neighbors_indices, radius);

	// Check result
	if (neighbors_indices.size() < 1)
	{
		PyErr_SetString(PyExc_RuntimeError, "Error");
		return NULL;
	}

	// Manage outputs
	// **************

	// Maximal number of neighbors
	int max_neighbors = neighbors_indices.size() / Nq;

	// Dimension of output containers
	npy_intp* neighbors_dims = new npy_intp[2];
	neighbors_dims[0] = Nq;
	neighbors_dims[1] = max_neighbors;

	// Create output array
	PyObject* res_obj = PyArray_SimpleNew(2, neighbors_dims, NPY_INT);
	PyObject* ret = NULL;

	// Fill output array with values
	size_t size_in_bytes = Nq * max_neighbors * sizeof(int);
	memcpy(PyArray_DATA(res_obj), neighbors_indices.data(), size_in_bytes);

	// Merge results
	ret = Py_BuildValue("N", res_obj);

	// Clean up
	// ********

	Py_XDECREF(queries_array);
	Py_XDECREF(supports_array);
	Py_XDECREF(q_batches_array);
	Py_XDECREF(s_batches_array);

	return ret;
}
