import pathlib
import os
import time

try:
    import yappi
except ImportError:
    yappi = None

PACKAGE_PATH = str(pathlib.Path(__file__).parent)
START_TIME = None


def _path_to_module(path):
    if PACKAGE_PATH not in path:
        return ""
    stem = pathlib.Path(path).stem
    path = "pyant" + os.sep + path.replace(PACKAGE_PATH, "").strip(os.sep)
    module = path.split(os.sep)
    module[-1] = stem
    return ".".join(module)


def check_yappi(func):
    def checked_func(*args, **kwargs):
        if yappi is None:
            raise ImportError("'yappi' not installed, please install to profile")
        return func(*args, **kwargs)

    return checked_func


@check_yappi
def profile():
    global START_TIME
    START_TIME = time.time()
    yappi.set_clock_type("cpu")
    yappi.start()


@check_yappi
def get_profile(modules=None):
    if modules is None:
        modules = ["pyant"]
    stats = yappi.get_func_stats(
        filter_callback=lambda x: any(
            list(_path_to_module(x.module).startswith(mod) for mod in modules)
        ),
    )
    stats = stats.sort("ttot", "desc")

    total = time.time() - START_TIME
    return stats, total


def print_profile(stats, total=None):
    header = [
        "Name",
        "Module",
        "Calls",
        "Total [s]",
        "Function [s]",
        "Average [s]",
    ]
    column_sizes = [len(title) for title in header]
    formats = [""] * 3 + ["1.4e"] * 3

    if total is not None:
        header += ["Total [%]"]
        column_sizes += [len(header[-1])]
        formats += ["2.3f"]

    if total is None:
        total = 1

    for ind in range(3, len(header)):
        if column_sizes[ind] < 6:
            column_sizes[ind]

    datas = [
        (
            fn.name,
            _path_to_module(fn.module),
            f"{fn.ncall}",
            fn.ttot,
            fn.tsub,
            fn.tavg,
            fn.ttot / total * 100,
        )
        for fn in stats
    ]
    for data in datas:
        for ind in range(3):
            if column_sizes[ind] < len(data[ind]):
                column_sizes[ind] = len(data[ind])

    _str = " | ".join([f"{title:^{size}}" for title, size in zip(header, column_sizes)])
    print(_str)
    print("-" * len(_str))

    for data in datas:
        _str = " | ".join(
            [
                f"{x:{fmt}}".ljust(size)
                for x, size, fmt in zip(data[: len(header)], column_sizes, formats)
            ]
        )
        print(_str)


@check_yappi
def profile_stop(clear=True):
    yappi.stop()
    if clear:
        global START_TIME
        START_TIME = None
        yappi.clear_stats()
