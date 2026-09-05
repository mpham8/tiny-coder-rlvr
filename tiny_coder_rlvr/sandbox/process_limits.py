import resource

from tiny_coder_rlvr import settings


def apply_limits():
    memory_limit = int(settings.memory_limit_bytes)
    cpu_limit = int(settings.cpu_limit_seconds)
    proc_limit = int(settings.proc_limit)
    open_files_limit = int(settings.open_files_limit)
    write_file_limit = int(settings.write_file_limit_bytes)
    core_dump_limit = int(settings.core_dump_limit)

    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    resource.setrlimit(resource.RLIMIT_NPROC, (proc_limit, proc_limit))
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_files_limit, open_files_limit))
    resource.setrlimit(resource.RLIMIT_FSIZE, (write_file_limit, write_file_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (core_dump_limit, core_dump_limit))


def __getattr__(name: str):
    mapping = {
        "MEMORY_LIMIT": "memory_limit_bytes",
        "CPU_LIMIT": "cpu_limit_seconds",
        "PROC_LIMIT": "proc_limit",
        "OPEN_FILES_LIMIT": "open_files_limit",
        "WRITE_FILE_LIMIT": "write_file_limit_bytes",
        "CORE_DUMP_LIMIT": "core_dump_limit",
        "RECURSION_LIMIT": "recursion_limit",
    }
    if name in mapping:
        return int(settings.get(mapping[name]))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
