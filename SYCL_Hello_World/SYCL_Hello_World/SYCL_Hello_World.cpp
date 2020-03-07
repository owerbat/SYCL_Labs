#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <CL/sycl.hpp>

using FPType = float;

const auto read_mode  = cl::sycl::access::mode::read;
const auto write_mode = cl::sycl::access::mode::write;

class add;

//class DeviceSelector : public cl::sycl::cpu_selector {
//public:
//	int operator()(const cl::sycl::device& device) const override {
//		const std::string name   = device.get_info<cl::sycl::info::device::name>();
//		const std::string vendor = device.get_info<cl::sycl::info::device::vendor>();
//
//		std::cout << "Device Name: " << name << std::endl;
//
//		return device.is_gpu() && (name.find("My Device Name") != std::string::npos);
//	}
//};

int main() {
	constexpr size_t size = 1024, group_size = 256;
	size_t group_number = 0;
	const std::vector<FPType> a(size, 1.0), b(size, 2.0);
	std::vector<FPType> c(size, 0.0);

	try {
		cl::sycl::queue queue(cl::sycl::cpu_selector{});

		cl::sycl::buffer<FPType, 1> a_buffer(a.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<FPType, 1> b_buffer(b.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<FPType, 1> c_buffer(c.data(), cl::sycl::range<1>(size));
		cl::sycl::buffer<size_t, 1> gn_buffer(&group_number, cl::sycl::range<1>(1));

		std::cout << "device: " << queue.get_device().get_info<cl::sycl::info::device::name>() << std::endl;

		queue.submit([&](cl::sycl::handler& cgh) {
			auto a_in  = a_buffer.get_access<read_mode >(cgh);
			auto b_in  = b_buffer.get_access<read_mode >(cgh);
			auto c_out = c_buffer.get_access<write_mode>(cgh);
			auto gn_out = gn_buffer.get_access<write_mode>(cgh);

			//cgh.parallel_for<add>(cl::sycl::nd_range<1>(cl::sycl::range<1>(size), cl::sycl::range<1>(group_size)), [=] (cl::sycl::nd_item<1> nd_item) {
			//		const size_t id = nd_item.get_global_linear_id();
			//		c_out[id] = a_in[id] + b_in[id];
			//		++(gn_out[0]);
			//		//std::cout << "group_id: " << nd_item.get_group_linear_id() << std::endl;
			//	});
			//});

			cgh.parallel_for_work_group<add>(cl::sycl::range<1>(size), [=] (cl::sycl::group<1> group) {
				cl::sycl::parallel_for_work_item(group, [=](cl::sycl::h_item<1> id) {
						c_out[id] = a_in[id] + b_in[id];
					});
					++(gn_out[0]);
					//std::cout << "group_id: " << nd_item.get_group_linear_id() << std::endl;
				});
			});

		queue.wait_and_throw();

		std::cout << "group_number = " << group_number << "\nresult: ";
		std::for_each(c.begin(), c.begin() + 10, [&c](const size_t i) { std::cout << c[i] << " "; });
		std::cout << std::endl;
	}
	catch (...) {
		std::cout << "Exception!" << std::endl;
	}

	return 0;
}
