// The process and system memory readings on Linux, for the same bench as
// process_memory.c. Linux has no phys_footprint; the resident set from
// /proc/self/statm is the closest per-process charge, and MemAvailable is the
// kernel's own estimate of what a new allocation could get without swapping.
// Every reader returns zero when its source cannot be read, as on macOS.

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

uint64_t astronomical_process_physical_footprint_bytes(void) {
  FILE *statm_file = fopen("/proc/self/statm", "r");
  if (statm_file == NULL) {
    return 0;
  }
  unsigned long long total_pages = 0;
  unsigned long long resident_pages = 0;
  int field_count = fscanf(statm_file, "%llu %llu", &total_pages, &resident_pages);
  fclose(statm_file);
  long page_size_bytes = sysconf(_SC_PAGESIZE);
  if (field_count != 2 || page_size_bytes <= 0) {
    return 0;
  }
  return (uint64_t)resident_pages * (uint64_t)page_size_bytes;
}

uint64_t astronomical_physical_memory_bytes(void) {
  long physical_page_count = sysconf(_SC_PHYS_PAGES);
  long page_size_bytes = sysconf(_SC_PAGESIZE);
  if (physical_page_count <= 0 || page_size_bytes <= 0) {
    return 0;
  }
  return (uint64_t)physical_page_count * (uint64_t)page_size_bytes;
}

uint64_t astronomical_system_reclaimable_bytes(void) {
  FILE *meminfo_file = fopen("/proc/meminfo", "r");
  if (meminfo_file == NULL) {
    return 0;
  }
  char meminfo_line[256];
  unsigned long long available_kibibytes = 0;
  int has_available_line = 0;
  while (fgets(meminfo_line, sizeof(meminfo_line), meminfo_file) != NULL) {
    if (sscanf(meminfo_line, "MemAvailable: %llu kB", &available_kibibytes) == 1) {
      has_available_line = 1;
      break;
    }
  }
  fclose(meminfo_file);
  return has_available_line ? (uint64_t)available_kibibytes * 1024 : 0;
}
