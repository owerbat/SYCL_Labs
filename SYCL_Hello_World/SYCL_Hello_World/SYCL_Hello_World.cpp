#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <CL/sycl.hpp>
#include <tbb/tick_count.h>

using FPType = float;

const auto read_mode  = cl::sycl::access::mode::read;
const auto write_mode = cl::sycl::access::mode::write;

class add;

bool check(std::vector<FPType> & vec, FPType value, FPType tol = 0.001);

int main() {
	constexpr size_t size = 1 << 20;
	constexpr size_t group_size = 256;
	const std::vector<FPType> a(size, 1.0), b(size, 2.0);
	std::vector<FPType> c(size, 0.0);

	try {
		cl::sycl::queue queue(cl::sycl::cpu_selector{}, cl::sycl::async_handler{});

		cl::sycl::buffer<FPType, 1> a_buffer(a.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<FPType, 1> b_buffer(b.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<FPType, 1> c_buffer(c.data(), cl::sycl::range<1>(size));

		auto devices = cl::sycl::device::get_devices();
		for (auto& device : devices) {
			std::cout << device.get_info<cl::sycl::info::device::name>() << " \n";
		}
		std::cout << std::endl;

		std::cout << "device: " << queue.get_device().get_info<cl::sycl::info::device::name>() << std::endl;

		auto t_start = tbb::tick_count::now();

		queue.submit([&](cl::sycl::handler& cgh) {
			auto a_in  = a_buffer.get_access<read_mode >(cgh);
			auto b_in  = b_buffer.get_access<read_mode >(cgh);
			auto c_out = c_buffer.get_access<write_mode>(cgh);

			cgh.parallel_for<add>(cl::sycl::nd_range<1>(cl::sycl::range<1>(size), cl::sycl::range<1>(group_size)),
				[=](cl::sycl::nd_item<1> item) {
					auto i = item.get_global_id();
					c_out[i] = a_in[i] + b_in[i];
				}
			);
		});

		queue.wait_and_throw();

		auto t_finish = tbb::tick_count::now();

		std::cout << "time: " << (t_finish - t_start).seconds() << std::endl;
		std::for_each(c.begin(), c.begin() + 10, [&c](const size_t i) { std::cout << c[i] << " "; });
		std::cout << std::endl;
		std::cout << "number of groups: " << size / group_size + !!(size % group_size) << std::endl;

		check(c, 3.0);
	}
	catch (std::exception e) {
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}

bool check(std::vector<FPType>& vec, FPType value, FPType tol) {
	for (auto item : vec) {
		if (std::abs(item - value) > tol) {
			std::cout << "Calculation error: " << item << " != " << value << std::endl;
			return false;
		}
	}
	std::cout << "Successful check\n";
	return true;
}
