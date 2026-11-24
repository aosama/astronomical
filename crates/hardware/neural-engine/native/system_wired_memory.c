// The system's wired memory, which is where the Neural Engine driver keeps a running
// program's weights. They appear neither in MLX's accounting nor in the process's
// physical footprint, so this is the only public reading that sees them.

#include <mach/mach.h>
#include <stdint.h>

uint64_t astronomical_system_wired_bytes(void) {
  vm_statistics64_data_t vm_statistics;
  mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
  mach_port_t host_port = mach_host_self();
  kern_return_t status =
      host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_statistics, &info_count);
  mach_port_deallocate(mach_task_self(), host_port);
  return status == KERN_SUCCESS ? (uint64_t)vm_statistics.wire_count * vm_kernel_page_size : 0;
}
