import resource

MEMORY_LIMIT = 512 * 2**20  # bytes
CPU_LIMIT = 5  # seconds CPU time
PROC_LIMIT = 1  # num processes
OPEN_FILES_LIMIT = 32  # num file descriptors
WRITE_FILE_LIMIT = 10 * 2**20  # bytes
CORE_DUMP_LIMIT = 0  # core dump
RECURSION_LIMIT = 1000  # recursion limit


def apply_limits():
    resource.setrlimit(
        resource.RLIMIT_AS,
        (MEMORY_LIMIT, MEMORY_LIMIT),
    )

    resource.setrlimit(
        resource.RLIMIT_CPU,
        (CPU_LIMIT, CPU_LIMIT),
    )

    resource.setrlimit(
        resource.RLIMIT_NPROC,
        (PROC_LIMIT, PROC_LIMIT),
    )

    resource.setrlimit(
        resource.RLIMIT_NOFILE,
        (OPEN_FILES_LIMIT, OPEN_FILES_LIMIT),
    )

    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (WRITE_FILE_LIMIT, WRITE_FILE_LIMIT),
    )

    resource.setrlimit(
        resource.RLIMIT_CORE,
        (CORE_DUMP_LIMIT, CORE_DUMP_LIMIT),
    )
