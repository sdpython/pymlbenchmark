"""
@file
@brief Implements a benchmark about performance.
"""
from io import StringIO
from pstats import SortKey, Stats
from cProfile import Profile as c_Profile
from profile import Profile as py_Profile
from pyinstrument import Profiler


class ProfilerCall:
    """
    Runs a profiler on a specific call.
    It can use either :epkg:`pyinstrument`,
    either :epkg:`cProfile`. The first module
    takes a snapshot every *interval* and stores
    the call stack. That explains why not all
    called functions appear in the summary
    but the overhead is smaller.
    """

    def __init__(self, fct_match=None, name="ProfilerCall",
                 repeat=1000, interval=0.0001, module="pyinstrument"):
        """
        @param      fct_match       function which tells if the profiler
                                    should be run on a set of parameters,
                                    signature is ``match(**kwargs) -> boolean``
        @param      repeat          number of times to repeat the function
                                    to profile
        @param      name            name of the profile
        @param      interval        see `interval <https://github.com/joerick/
                                    pyinstrument/blob/master/pyinstrument/profiler.py#L22>`_
        @param      module          ``'pyinstrument'`` by default, ``'cProfile'`` or
                                    ``'profile'`` works too
        """
        self.interval = interval
        if callable(fct_match):
            self.fct_match = fct_match
        elif isinstance(fct_match, dict):
            def match(**kwargs):
                for k, v in kwargs.items():
                    if v != fct_match.get(k, v):
                        return False
                return True
            self.fct_match = match
        else:
            self.fct_match = fct_match
        self.profiled = []
        self.kwargs = []
        self.name = name
        self.repeat = repeat
        self.module = module

    def match(self, **kwargs):
        """
        Tells if the profiler should be run on this
        set of parameters.

        @param      kwargs          dictionary of parameters
        @return                     boolean
        """
        return self.fct_match is None or self.fct_match(**kwargs)

    def profile(self, fct, **kwargs):
        """
        Profiles function *fct*, calls it
        *repeat* times.

        @param      fct     function to profile (no argument)
        @param      kwargs  stores additional information
                            about the profiling
        """
        if self.module in ('pyinstrument', 'cProfile'):
            if self.module == 'pyinstrument':
                profiler = Profiler(interval=self.interval)
                start = profiler.start
                stop = profiler.stop
            else:
                profiler = c_Profile()
                start = profiler.enable
                stop = profiler.disable
            start()
            for _ in range(self.repeat):
                fct()
            stop()
        elif self.module == "profile":
            profiler = py_Profile()

            def lf():
                for _ in range(self.repeat):
                    fct()
            profiler.runcall(lf)
        else:
            raise ValueError(  # pragma: no cover
                "Unknown profiler '{}'.".format(self.module))
        self.profiled.append(profiler)
        self.kwargs.append(kwargs)

    def __len__(self):
        """
        Returns the number of stored profiles.
        """
        return len(self.profiled)

    def __iter__(self):
        """
        Iterates on stored profiled.
        Returns a couple ``(profile, configuration)``.
        """
        for a, b in zip(self.profiled, self.kwargs):
            yield a, b

    def to_txt(self, filename):
        """
        Saves all profiles into one file.

        @param      filename        filename where to save the profiles,
                                    can be a stream
        """
        if len(self) == 0:
            raise ValueError(  # pragma: no cover
                "No profile was done.")
        if isinstance(filename, str):
            with open(filename, "w") as f:
                self.to_txt(f)
            return

        f = filename
        f.write(self.name + "\n")
        for i, (prof, kw) in enumerate(self):
            f.write("------------------------------------------------------\n")
            f.write("profile %d\n" % i)
            if kw:
                for a, b in sorted(kw.items()):
                    f.write("%s=%s\n" % (a, str(b).replace('\n', '\\n')))
                f.write("--\n")
            if hasattr(prof, 'output_text'):
                f.write(prof.output_text(unicode=False, color=False))
            else:
                s = StringIO()
                sortby = SortKey.CUMULATIVE
                ps = Stats(prof, stream=s).sort_stats(sortby)
                ps.print_stats()
                f.write(s.getvalue())
            f.write("\n")
