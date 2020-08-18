#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "grid_subsampling/grid_subsampling.h"

using namespace tensorflow;

REGISTER_OP("GridSubsampling")
    .Input("points: float")
    .Input("dl: float")
    .Output("sub_points: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        c->set_output(0, input);
        return Status::OK();
    });





class GridSubsamplingOp : public OpKernel {
    public:
    explicit GridSubsamplingOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

        // Grab the input tensors
        const Tensor& points_tensor = context->input(0);
        const Tensor& dl_tensor = context->input(1);

        // check shapes of input and weights
        const TensorShape& points_shape = points_tensor.shape();

        // check input are [N x 3] matrices
        DCHECK_EQ(points_shape.dims(), 2);
        DCHECK_EQ(points_shape.dim_size(1), 3);

        // Dimensions
        int N = (int)points_shape.dim_size(0);

        // get the data as std vector of points
        float sampleDl = dl_tensor.flat<float>().data()[0];
        vector<PointXYZ> original_points = vector<PointXYZ>((PointXYZ*)points_tensor.flat<float>().data(),
                                                            (PointXYZ*)points_tensor.flat<float>().data() + N);

        // Unsupported label and features
        vector<float> original_features;
        vector<int> original_classes;

        // Create result containers
        vector<PointXYZ> subsampled_points;
        vector<float> subsampled_features;
        vector<int> subsampled_classes;

        // Compute results
        grid_subsampling(original_points,
                         subsampled_points,
                         original_features,
                         subsampled_features,
                         original_classes,
                         subsampled_classes,
                         sampleDl);

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(subsampled_points.size());
        output_shape.AddDim(3);

        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_tensor = output->matrix<float>();

        // Fill output tensor
        for (int i = 0; i < output->shape().dim_size(0); i++)
        {
            output_tensor(i, 0) = subsampled_points[i].x;
            output_tensor(i, 1) = subsampled_points[i].y;
            output_tensor(i, 2) = subsampled_points[i].z;
        }
    }
};


REGISTER_KERNEL_BUILDER(Name("GridSubsampling").Device(DEVICE_CPU), GridSubsamplingOp);