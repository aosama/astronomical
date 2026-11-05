// Process and system memory readings for the Neural Engine footprint
// measurement (issue #161).
//
// `phys_footprint` is the number Activity Monitor and jetsam charge to a
// process, and the reading #376 measured its 4-6x Neural Engine multiplier
// with, so the comparisons here stay on the same scale. The system wired
// count is read as a cross-check, because memory the kernel wires on the
// process's behalf is not always charged to the process.

#include <mach/mach.h>
#include <stdint.h>
#include <sys/sysctl.h>

uint64_t astronomical_process_physical_footprint_bytes(void) {
  task_vm_info_data_t vm_info;
  mach_msg_type_number_t info_count = TASK_VM_INFO_COUNT;
  kern_return_t status =
      task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &info_count);
  return status == KERN_SUCCESS ? vm_info.phys_footprint : 0;
}

uint64_t astronomical_physical_memory_bytes(void) {
  uint64_t physical_memory_bytes = 0;
  size_t value_size = sizeof(physical_memory_bytes);
  int status = sysctlbyname("hw.memsize", &physical_memory_bytes, &value_size, NULL, 0);
  return status == 0 ? physical_memory_bytes : 0;
}

uint64_t astronomical_system_wired_bytes(void) {
  vm_statistics64_data_t vm_statistics;
  mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
  mach_port_t host_port = mach_host_self();
  kern_return_t status =
      host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_statistics, &info_count);
  mach_port_deallocate(mach_task_self(), host_port);
  return status == KERN_SUCCESS ? (uint64_t)vm_statistics.wire_count * vm_kernel_page_size : 0;
}
