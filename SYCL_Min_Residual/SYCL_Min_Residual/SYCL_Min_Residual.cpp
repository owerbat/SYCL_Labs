#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <CL/sycl.hpp>
#include <tbb/tick_count.h>
#include <mkl_blas_sycl.hpp>

using FPType = float;

const auto read_mode = cl::sycl::access::mode::read;
const auto write_mode = cl::sycl::access::mode::write;

void fill_matrix(std::vector<FPType>& matrix, const size_t size);
void solve_mkl(std::vector<FPType>& a, std::vector<FPType>& b, std::vector<FPType>& x, const size_t max_iter);
void solve_internal(std::vector<FPType>& a, std::vector<FPType>& b, std::vector<FPType>& x, const size_t max_iter);
void check(std::vector<FPType>& a, std::vector<FPType>& b, std::vector<FPType>& x);

int main() {
	constexpr size_t size = 1 << 10, max_iter = 10000;
	std::vector<FPType> a(size * size), b(size), x(size, 0.0);

	fill_matrix(a, size);
	for (size_t i = 0; i < size; ++i) b[i] = 1;

	solve_mkl(a, b, x, max_iter);
	check(a, b, x);

	return 0;
}

void fill_matrix(std::vector<FPType>& matrix, const size_t size) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<FPType> dis(1.0, 2.0);

	for (size_t j = 0; j < size; j++) {
		FPType sum = 0;
		for (size_t i = 0; i < size; ++i) {
			matrix[j * size + i] = dis(gen);
			sum += matrix[j * size + i];
		}
		matrix[j * size + j] = sum * dis(gen);
	}
}

void solve_mkl(std::vector<FPType>& a, std::vector<FPType>& b, std::vector<FPType>& x, const size_t max_iter) {
	const size_t size = x.size();
	size_t iter = 0;
	std::vector<FPType> residuals(size), ar(size);
	FPType norm = 0, tau1 = 0, tau2 = 1;

	try {
		cl::sycl::queue queue(cl::sycl::cpu_selector{}, cl::sycl::async_handler{});

		cl::sycl::buffer<FPType, 1> a_buffer(a.data(), cl::sycl::range<1>(size * size));
		cl::sycl::buffer<FPType, 1> b_buffer(b.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<FPType, 1> x_buffer(x.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<FPType, 1> r_buffer(residuals.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<FPType, 1> ar_buffer(ar.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<FPType, 1> tau1_buffer(&tau1, cl::sycl::range<1>(1));
		cl::sycl::buffer<FPType, 1> tau2_buffer(&tau2, cl::sycl::range<1>(1));
		cl::sycl::buffer<FPType, 1> norm_buffer(&norm, cl::sycl::range<1>(1));

		std::cout << "device: " << queue.get_device().get_info<cl::sycl::info::device::name>() << std::endl;

		auto t_start_mkl = tbb::tick_count::now();
		for (iter = 0; iter < max_iter; ++iter) {
			mkl::blas::gemv(queue, mkl::transpose::T, size, size, 1, a_buffer, size, x_buffer, 1, 0, r_buffer, 1);
			mkl::blas::axpy(queue, size, -1, b_buffer, 1, r_buffer, 1);

			mkl::blas::dot(queue, size, r_buffer, 1, r_buffer, 1, norm_buffer);
			if (norm < 1e-5) break;

			mkl::blas::gemv(queue, mkl::transpose::T, size, size, 1, a_buffer, size, r_buffer, 1, 0, ar_buffer, 1);
			mkl::blas::dot(queue, size, ar_buffer, 1, r_buffer, 1, tau1_buffer);
			mkl::blas::dot(queue, size, ar_buffer, 1, ar_buffer, 1, tau2_buffer);
			FPType tau = tau1 / tau2;

			mkl::blas::axpy(queue, size, -tau, r_buffer, 1, x_buffer, 1);
		}
		auto t_finish_mkl = tbb::tick_count::now();

		std::cout << "mkl time: " << (t_finish_mkl - t_start_mkl).seconds() << std::endl;
		std::cout << "mkl iterations: " << iter + 1 << std::endl;
		std::cout << "mkl norm: " << norm << std::endl;
	}
	catch (std::exception e) {
		std::cout << "Exception: " << e.what() << std::endl;
	}
}

void solve_internal(std::vector<FPType>& a, std::vector<FPType>& b, std::vector<FPType>& x, const size_t max_iter) {

}

void check(std::vector<FPType>& a, std::vector<FPType>& b, std::vector<FPType>& x) {
	const size_t size = x.size();

	for (size_t i = 0; i < size; ++i) {
		FPType residual = 0.0;
		for (size_t j = 0; j < size; ++j) {
			residual += a[j * size + i] * x[j];
		}
		if (std::abs(residual - b[i]) > 0.001) {
			std::cout << "Error: difference is too large\n";
			return;
		}
	}
	std::cout << "Successful check\n";
}
