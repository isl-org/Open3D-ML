

#include <cstdlib>
void cpp_knn(const float* points, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, long* indices);

void cpp_knn_omp(const float* points, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, long* indices);


void cpp_knn_batch(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim,
			const float* queries, const size_t nqueries,
			const size_t K, long* batch_indices);

void cpp_knn_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, 
				const float* queries, const size_t nqueries,
				const size_t K, long* batch_indices);

void cpp_knn_batch_distance_pick(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, 
				float* queries, const size_t nqueries,
				const size_t K, long* batch_indices);

void cpp_knn_batch_distance_pick_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, 
				float* batch_queries, const size_t nqueries,
				const size_t K, long* batch_indices);