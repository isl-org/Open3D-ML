#include <Python.h>
#include <numpy/arrayobject.h>
#include "grid_subsampling/grid_subsampling.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module provides an interface for the subsampling of a pointcloud";

static char compute_docstring[] = "function subsampling a pointcloud";


// Declare the functions
// *********************

static PyObject *grid_subsampling_compute(PyObject *self, PyObject *args, PyObject *keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "compute", (PyCFunction)grid_subsampling_compute, METH_VARARGS | METH_KEYWORDS, compute_docstring },
    {NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "grid_subsampling",     // m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_grid_subsampling(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Actual wrapper
// **************

static PyObject *grid_subsampling_compute(PyObject *self, PyObject *args, PyObject *keywds)
{

    // Manage inputs
    // *************

	// Args containers
	PyObject *points_obj = NULL;
	PyObject *features_obj = NULL;
	PyObject *classes_obj = NULL;

	// Keywords containers
	static char *kwlist[] = {"points", "features", "classes", "sampleDl", "method", "verbose", NULL };
	float sampleDl = 0.1;
	const char *method_buffer = "barycenters";
	int verbose = 0;

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|$OOfsi", kwlist, &points_obj, &features_obj, &classes_obj, &sampleDl, &method_buffer, &verbose))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Get the method argument
	string method(method_buffer);

	// Interpret method
	if (method.compare("barycenters") && method.compare("voxelcenters"))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing method. Valid method names are \"barycenters\" and \"voxelcenters\" ");
		return NULL;
	}

	// Check if using features or classes
	bool use_feature = true, use_classes = true;
	if (features_obj == NULL)
		use_feature = false;
	if (classes_obj == NULL)
		use_classes = false;

    // Interpret the input objects as numpy arrays.
	PyObject *points_array = PyArray_FROM_OTF(points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject *features_array = NULL;
	PyObject *classes_array = NULL;
	if (use_feature)
		features_array = PyArray_FROM_OTF(features_obj, NPY_FLOAT, NPY_IN_ARRAY);
	if (use_classes)
		classes_array = PyArray_FROM_OTF(classes_obj, NPY_INT, NPY_IN_ARRAY);

	// Verify data was load correctly.
	if (points_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(classes_array);
		Py_XDECREF(features_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting input points to numpy arrays of type float32");
		return NULL;
	}
	if (use_feature && features_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(classes_array);
		Py_XDECREF(features_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting input features to numpy arrays of type float32");
		return NULL;
	}
	if (use_classes && classes_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(classes_array);
		Py_XDECREF(features_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting input classes to numpy arrays of type int32");
		return NULL;
	}

    // Check that the input array respect the dims
	if ((int)PyArray_NDIM(points_array) != 2 || (int)PyArray_DIM(points_array, 1) != 3)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(classes_array);
		Py_XDECREF(features_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : points.shape is not (N, 3)");
		return NULL;
	}
	if (use_feature && ((int)PyArray_NDIM(features_array) != 2))
	{
		Py_XDECREF(points_array);
		Py_XDECREF(classes_array);
		Py_XDECREF(features_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : features.shape is not (N, d)");
		return NULL;
	}

	if (use_classes && (int)PyArray_NDIM(classes_array) > 2)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(classes_array);
		Py_XDECREF(features_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : classes.shape is not (N,) or (N, d)");
		return NULL;
	}

	// Number of points
	int N = (int)PyArray_DIM(points_array, 0);

	// Dimension of the features
	int fdim = 0;
	if (use_feature)
	    fdim = (int)PyArray_DIM(features_array, 1);

	//Dimension of labels
	int ldim = 1;
	if (use_classes && (int)PyArray_NDIM(classes_array) == 2)
	    ldim = (int)PyArray_DIM(classes_array, 1);

	// Check that the input array respect the number of points
	if (use_feature && (int)PyArray_DIM(features_array, 0) != N)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(classes_array);
		Py_XDECREF(features_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : features.shape is not (N, d)");
		return NULL;
	}
	if (use_classes && (int)PyArray_DIM(classes_array, 0) != N)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(classes_array);
		Py_XDECREF(features_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : classes.shape is not (N,) or (N, d)");
		return NULL;
	}


    // Call the C++ function
    // *********************

    // Create pyramid
    if (verbose > 0)
        cout << "Computing cloud pyramid with support points: " << endl;


    // Convert PyArray to Cloud C++ class
	vector<PointXYZ> original_points;
	vector<float> original_features;
	vector<int> original_classes;
	original_points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(points_array), (PointXYZ*)PyArray_DATA(points_array) + N);
	if (use_feature)
		original_features = vector<float>((float*)PyArray_DATA(features_array), (float*)PyArray_DATA(features_array) + N*fdim);
	if (use_classes)
		original_classes = vector<int>((int*)PyArray_DATA(classes_array), (int*)PyArray_DATA(classes_array) + N*ldim);

    // Subsample
	vector<PointXYZ> subsampled_points;
	vector<float> subsampled_features;
	vector<int> subsampled_classes;
	grid_subsampling(original_points,
                     subsampled_points,
                     original_features,
                     subsampled_features,
                     original_classes,
                     subsampled_classes,
                     sampleDl,
                     verbose);

    // Check result
	if (subsampled_points.size() < 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error");
        return NULL;
    }

    // Manage outputs
    // **************

	// Dimension of input containers
	npy_intp* point_dims = new npy_intp[2];
	point_dims[0] = subsampled_points.size();
	point_dims[1] = 3;
	npy_intp* feature_dims = new npy_intp[2];
	feature_dims[0] = subsampled_points.size();
	feature_dims[1] = fdim;
	npy_intp* classes_dims = new npy_intp[2];
	classes_dims[0] = subsampled_points.size();
	classes_dims[1] = ldim;

    // Create output array
	PyObject *res_points_obj = PyArray_SimpleNew(2, point_dims, NPY_FLOAT);
	PyObject *res_features_obj = NULL;
	PyObject *res_classes_obj = NULL;
	PyObject *ret = NULL;

	// Fill output array with values
	size_t size_in_bytes = subsampled_points.size() * 3 * sizeof(float);
	memcpy(PyArray_DATA(res_points_obj), subsampled_points.data(), size_in_bytes);
	if (use_feature)
	{
	    size_in_bytes = subsampled_points.size() * fdim * sizeof(float);
		res_features_obj = PyArray_SimpleNew(2, feature_dims, NPY_FLOAT);
		memcpy(PyArray_DATA(res_features_obj), subsampled_features.data(), size_in_bytes);
	}
	if (use_classes)
	{
		size_in_bytes = subsampled_points.size() * ldim * sizeof(int);
		res_classes_obj = PyArray_SimpleNew(2, classes_dims, NPY_INT);
		memcpy(PyArray_DATA(res_classes_obj), subsampled_classes.data(), size_in_bytes);
	}


	// Merge results
	if (use_feature && use_classes)
		ret = Py_BuildValue("NNN", res_points_obj, res_features_obj, res_classes_obj);
	else if (use_feature)
		ret = Py_BuildValue("NN", res_points_obj, res_features_obj);
	else if (use_classes)
		ret = Py_BuildValue("NN", res_points_obj, res_classes_obj);
	else
		ret = Py_BuildValue("N", res_points_obj);

    // Clean up
    // ********

	Py_DECREF(points_array);
	Py_XDECREF(features_array);
	Py_XDECREF(classes_array);

	return ret;
}