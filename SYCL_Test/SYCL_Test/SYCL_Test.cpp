#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <CL/sycl.hpp>

using FPType = float;

const auto read_mode  = cl::sycl::access::mode::read;
const auto write_mode = cl::sycl::access::mode::write;

class add;

int main() {
	constexpr size_t size = 1024;
	const std::vector<FPType> a(size, 1.0), b(size, 2.0);
	std::vector<FPType> c(size, 0.0);

	auto devices = cl::sycl::device::get_devices();
	for (auto & device : devices) {
		std::cout << device.get_info<cl::sycl::info::device::name>() << " \n";
	}
	std::cout << std::endl;

	try {
		cl::sycl::queue queue(cl::sycl::gpu_selector{}, cl::sycl::async_handler{});

		cl::sycl::buffer<FPType, 1> a_buffer(a.data(), cl::sycl::range<1>(a.size()));
		cl::sycl::buffer<FPType, 1> b_buffer(b.data(), cl::sycl::range<1>(a.size()));
		cl::sycl::buffer<FPType, 1> c_buffer(c.data(), cl::sycl::range<1>(a.size()));

		queue.submit([&](cl::sycl::handler & cgh) {
			auto a_in  = a_buffer.get_access<read_mode>(cgh);
			auto b_in  = b_buffer.get_access<read_mode>(cgh);
			auto c_out = c_buffer.get_access<write_mode>(cgh);

			cgh.parallel_for<add>(cl::sycl::range<1>(size), [=] (cl::sycl::id<1> id) {
				c_out[id] = a_in[id] + b_in[id];
			});
		});

		queue.wait_and_throw();

		std::cout << "result: ";
		std::for_each(c.begin(), c.begin() + 10, [&c](const size_t i) { std::cout << c[i] << " "; });
		std::cout << std::endl;
	}
	catch (...) {
		std::cout << "Exception!" << std::endl;
	}

	return 0;
}
