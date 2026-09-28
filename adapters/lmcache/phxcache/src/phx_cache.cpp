#include "phx_cache.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>

namespace {

inline size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

}  // namespace

// ---------------------------------------------------------------------------
// PhxCache
// ---------------------------------------------------------------------------

PhxCache::PhxCache(int device_id)
    : device_id_(device_id), initialized_(false) {
    /* Public Phoenix device ids are CUDA/HIP/NPU ordinals. */
    dev_ = device_id;

    int ret = phxfs_open(dev_);
    if (ret < 0) {
        throw std::runtime_error("PhxCache: phxfs_open(" +
                                 std::to_string(dev_) + ") failed with " +
                                 std::to_string(ret));
    }
    initialized_ = true;
}

PhxCache::~PhxCache() {
    if (initialized_) {
        try {
            close();
        } catch (...) {
        }
    }
}

uintptr_t PhxCache::regmem(uintptr_t dev_addr, size_t size) {
    size_t page_size = phxfs_get_page_size();
    size_t aligned_size = align_up(size, page_size);

    void *target_addr = nullptr;
    int ret = phxfs_regmem(dev_, reinterpret_cast<const void *>(dev_addr),
                           aligned_size, &target_addr);
    if (ret < 0) {
        throw std::runtime_error(
            "PhxCache::regmem: phxfs_regmem failed with " +
            std::to_string(ret) + ", size=" + std::to_string(size) +
            ", aligned_size=" + std::to_string(aligned_size));
    }

    auto mapped = reinterpret_cast<uintptr_t>(target_addr);
    reg_map_[dev_addr] = {aligned_size, mapped};
    return mapped;
}

void PhxCache::deregmem(uintptr_t dev_addr, size_t size) {
    auto it = reg_map_.find(dev_addr);
    if (it == reg_map_.end()) {
        return;  // Not registered, nothing to do
    }

    size_t aligned_size = it->second.first;
    int ret = phxfs_deregmem(dev_, reinterpret_cast<const void *>(dev_addr),
                             aligned_size);
    if (ret < 0) {
        fprintf(stderr,
                "PhxCache::deregmem: phxfs_deregmem failed with %d\n", ret);
    }
    reg_map_.erase(it);
}

void PhxCache::close() {
    if (!initialized_) {
        return;
    }

    // Deregister any remaining registered memory
    for (const auto &[dev_addr, info] : reg_map_) {
        int ret = phxfs_deregmem(dev_, reinterpret_cast<const void *>(dev_addr),
                                 info.first);
        if (ret < 0) {
            fprintf(stderr,
                    "PhxCache::close: phxfs_deregmem failed with %d\n", ret);
        }
    }
    reg_map_.clear();

    int ret = phxfs_close(dev_);
    if (ret < 0) {
        fprintf(stderr, "PhxCache::close: phxfs_close failed with %d\n", ret);
    }
    initialized_ = false;
}

uint64_t PhxCache::page_size() const {
    return phxfs_get_page_size();
}

// ---------------------------------------------------------------------------
// PhxCache — batch I/O
// ---------------------------------------------------------------------------

std::vector<ssize_t> PhxCache::read_batch(
        uintptr_t buf_base,
        const std::vector<std::tuple<int, off_t, size_t, off_t>> &reqs) {

    if (reqs.empty()) {
        return {};
    }

    std::vector<phxfs_io_req_t> io_reqs(reqs.size());
    for (size_t i = 0; i < reqs.size(); i++) {
        io_reqs[i].fd         = std::get<0>(reqs[i]);
        io_reqs[i].device_id  = dev_;               // GPU buffer (registered)
        io_reqs[i].buf        = reinterpret_cast<void *>(buf_base);
        io_reqs[i].buf_offset = std::get<1>(reqs[i]);
        io_reqs[i].nbytes     = std::get<2>(reqs[i]);
        io_reqs[i].f_offset   = std::get<3>(reqs[i]);
        io_reqs[i].result     = 0;
    }

    int failed = phxfs_read_batch(io_reqs.data(),
                                  static_cast<int>(io_reqs.size()));
    if (failed < 0) {
        throw std::runtime_error(
            "PhxCache::read_batch: phxfs_read_batch failed with " +
            std::to_string(failed));
    }

    std::vector<ssize_t> results(reqs.size());
    for (size_t i = 0; i < reqs.size(); i++) {
        results[i] = io_reqs[i].result;
    }
    return results;
}

std::vector<ssize_t> PhxCache::write_batch(
        const std::vector<std::tuple<int, uintptr_t, off_t, size_t, off_t>> &reqs) {

    if (reqs.empty()) {
        return {};
    }

    std::vector<phxfs_io_req_t> io_reqs(reqs.size());
    for (size_t i = 0; i < reqs.size(); i++) {
        io_reqs[i].fd         = std::get<0>(reqs[i]);
        io_reqs[i].device_id  = -1;                 // CPU buffer (no regmem)
        io_reqs[i].buf        = reinterpret_cast<void *>(std::get<1>(reqs[i]));
        io_reqs[i].buf_offset = std::get<2>(reqs[i]);
        io_reqs[i].nbytes     = std::get<3>(reqs[i]);
        io_reqs[i].f_offset   = std::get<4>(reqs[i]);
        io_reqs[i].result     = 0;
    }

    int failed = phxfs_write_batch(io_reqs.data(),
                                   static_cast<int>(io_reqs.size()));
    if (failed < 0) {
        throw std::runtime_error(
            "PhxCache::write_batch: phxfs_write_batch failed with " +
            std::to_string(failed));
    }

    std::vector<ssize_t> results(reqs.size());
    for (size_t i = 0; i < reqs.size(); i++) {
        results[i] = io_reqs[i].result;
    }
    return results;
}

// ---------------------------------------------------------------------------
// PhxFile
// ---------------------------------------------------------------------------

PhxFile::PhxFile(const PhxCache &cache, const std::string &path, int flags)
    : owns_fd_(false) {
    // When O_CREAT is in flags, open() requires a mode argument; omitting it
    // causes open() to read a garbage value from the stack as the file mode,
    // which can create files with permission 0 (EACCES on subsequent opens).
    int fd = (flags & O_CREAT)
                 ? ::open(path.c_str(), flags, 0644)
                 : ::open(path.c_str(), flags);
    if (fd < 0) {
        throw std::system_error(errno, std::system_category(),
                                "PhxFile: open(" + path + ")");
    }
    fd_ = fd;
    dev_ = cache.device_id();
    owns_fd_ = true;
}

PhxFile::~PhxFile() {
    if (owns_fd_) {
        try {
            close();
        } catch (...) {
        }
    }
}

ssize_t PhxFile::read(uintptr_t buf, off_t buf_offset, ssize_t nbyte,
                      off_t f_offset) {
    ssize_t result = phxfs_read(
        fd_, dev_,
        reinterpret_cast<void *>(buf),
        buf_offset,
        nbyte,
        f_offset);
    if (result < 0) {
        throw std::runtime_error(
            "PhxFile::read: phxfs_read failed with " +
            std::to_string(result) + ", buf_offset=" +
            std::to_string(buf_offset) + ", nbyte=" +
            std::to_string(nbyte) + ", f_offset=" +
            std::to_string(f_offset));
    }
    return result;
}

ssize_t PhxFile::write(uintptr_t buf, off_t buf_offset, ssize_t nbyte,
                       off_t f_offset) {
    ssize_t result = phxfs_write(
        fd_, dev_,
        reinterpret_cast<void *>(buf),
        buf_offset,
        nbyte,
        f_offset);
    if (result < 0) {
        throw std::runtime_error(
            "PhxFile::write: phxfs_write failed with " +
            std::to_string(result));
    }
    return result;
}

void PhxFile::close() {
    if (owns_fd_) {
        ::close(fd_);
        owns_fd_ = false;
    }
}
