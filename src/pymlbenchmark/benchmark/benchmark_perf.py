"""
@file
@brief Implements a benchmark about performance.
"""
import os
import pickle
from time import perf_counter as time_perf
import numpy
from .bench_helper import enumerate_options


class BenchPerfTest:
    """
    Defines a bench perf test.
    See example :ref:`l-bench-slk-poly`.

    .. faqref::
        :title: Conventions for N, dim

        In all the package, *N* refers to the number of observations,
        *dim* the dimension or the number of features.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def data(self, **opts):
        """
        Generates one testing dataset.

        @return                 dataset, usually a list of arrays
                                such as *X, y*
        """
        raise NotImplementedError()

    def fcts(self, **opts):
        """
        Returns the function call to test,
        it produces a dictionary ``{name: fct}``
        where *name* is the name of the function
        and *fct* the function to benchmark
        """
        raise NotImplementedError()

    def validate(self, results, **kwargs):
        """
        Runs validations after the test was done
        to make sure it was valid.

        @param      results     results to validate, list of tuple
                                ``(parameters, results)``
        @param      kwargs      additional information in case
                                errors must traced

        The function raised an exception or not.
        """
        pass

    def dump_error(self, msg, **kwargs):
        """
        Dumps everything which is needed to investigate an error.
        Everything is pickled in the current folder or *dump_folder*
        is attribute *dump_folder* was defined. This folder is created
        if it does not exist.

        @param      msg     message
        @param      kwargs  needed data to investigate
        @return             filename
        """
        dump_folder = getattr(self, "dump_folder", '.')
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)
        pattern = os.path.join(
            dump_folder, "BENCH-ERROR-{0}-%d.pkl".format(
                self.__class__.__name__))
        err = 0
        name = pattern % err
        while os.path.exists(name):
            err += 1
            name = pattern % err
        with open(name, "wb") as f:
            pickle.dump({'msg': msg, 'data': kwargs}, f)


class BenchPerf:
    """
    Factorizes code to compare two implementations.
    See example :ref:`l-bench-slk-poly`.
    """

    def __init__(self, pbefore, pafter, btest, filter_test=None,
                 profilers=None):
        """
        @param      pbefore     parameters before calling *fct*,
                                dictionary ``{name: [list of values]}``,
                                these parameters are sent to the instance
                                of @see cl BenchPerfTest to test
        @param      pafter      parameters after calling *fct*,
                                dictionary ``{name: [list of values]}``,
                                these parameters are sent to method
                                :meth:`BenchPerfTest.fcts
                                <pymlbenchmark.benchmark.benchmark_perf.BenchPerfTest.fcts>`
        @param      btest       instance of @see cl BenchPerfTest
        @param      filter_test function which tells if a configuration
                                must be tested or not, None to test them
                                all
        @param      profilers   list of profilers to run

        Every parameter specifies a function is called through
        a method. The user can only overwrite it.
        """
        self.pbefore = pbefore
        self.pafter = pafter
        self.btest = btest
        self.filter_test = filter_test
        self.profilers = profilers

    def fct_filter_test(self, **conf):
        """
        Tells if the test by *conf* is valid or not.

        @param      conf        dictionary ``{name: value}``
        @return                 boolean
        """
        if self.filter_test is None:
            return True
        return self.filter_test(**conf)

    def enumerate_tests(self, options):
        """
        Enumerates all possible options.

        @param      options     dictionary ``{name: list of values}``
        @return                 list of dictionary ``{name: value}``

        The function applies the method *fct_filter_test*.
        """
        for row in enumerate_options(options, self.fct_filter_test):
            yield row

    def enumerate_run_benchs(self, repeat=10, verbose=False,
                             stop_if_error=True, validate=True,
                             number=1):
        """
        Runs the benchmark.

        @param      repeat          number of repeatition of the same call
                                    with different datasets
        @param      verbose         if True, use :epkg:`tqdm`
        @param      stop_if_error   by default, it stops when method *validate*
                                    fails, if False, the function stores the exception
        @param      validate        compare the outputs against the baseline
        @param      number          number of times to call the same function,
                                    the method then measure this number calls
        @return                     yields dictionaries with all the metrics
        """
        all_opts = self.pbefore.copy()
        all_opts.update(self.pafter)
        all_tests = list(self.enumerate_tests(all_opts))

        if verbose:
            from tqdm import tqdm
            loop = iter(tqdm(range(len(all_tests))))
        else:
            loop = iter(all_tests)

        for a_opt in self.enumerate_tests(self.pbefore):
            if not self.fct_filter_test(**a_opt):
                continue

            inst = self.btest(**a_opt)

            for b_opt in self.enumerate_tests(self.pafter):
                obs = b_opt.copy()
                obs.update(a_opt)
                if not self.fct_filter_test(**obs):
                    continue

                fcts = inst.fcts(**obs)
                if not isinstance(fcts, list):
                    raise TypeError(
                        "Method fcts must return a list of dictionaries (name, fct) not {}".format(fcts))

                data = [inst.data(**obs) for r in range(repeat)]
                if not isinstance(data, (list, tuple)):
                    raise ValueError(
                        "Method *data* must return a list or a tuple.")
                obs["repeat"] = len(data)
                obs["number"] = number
                results = []
                stores = []

                for fct in fcts:
                    if not isinstance(fct, dict) or 'fct' not in fct:
                        raise ValueError(
                            "Method fcts must return a list of dictionaries with keys "
                            "('name', 'fct') not {}".format(fct))
                    f = fct['fct']
                    del fct['fct']
                    times = []
                    fct.update(obs)

                    if isinstance(f, tuple):
                        if len(f) != 2:
                            raise RuntimeError(
                                "If *f* is a tuple, it must return two function f1, f2.")
                        f1, f2 = f
                        dt = data[0]
                        dt2 = f1(*dt)
                        self.profile(fct, lambda: f2(*dt2))
                        for idt, dt in enumerate(data):
                            dt2 = f1(*dt)
                            if number == 1:
                                st = time_perf()
                                r = f2(*dt2)
                                d = time_perf() - st
                            else:
                                st = time_perf()
                                for _ in range(number):
                                    r = f2(*dt2)
                                d = time_perf() - st
                            times.append(d)
                            results.append((idt, fct, r))
                    else:
                        dt = data[0]
                        self.profile(fct, lambda: f(*dt))
                        for idt, dt in enumerate(data):
                            if number == 1:
                                st = time_perf()
                                r = f(*dt)
                                d = time_perf() - st
                            else:
                                st = time_perf()
                                for _ in range(number):
                                    r = f(*dt)
                                d = time_perf() - st
                            times.append(d)
                            results.append((idt, fct, r))
                    times.sort()
                    fct['min'] = times[0]
                    fct['max'] = times[-1]
                    if len(times) > 5:
                        fct['min3'] = times[3]
                        fct['max3'] = times[-3]
                    times = numpy.array(times)
                    fct['mean'] = times.mean()
                    std = times.std()
                    if len(times) >= 4:
                        fct['lower'] = max(
                            fct['min'], fct['mean'] - std * 1.96)
                        fct['upper'] = min(
                            fct['max'], fct['mean'] + std * 1.96)
                    else:
                        fct['lower'] = fct['min']
                        fct['upper'] = fct['max']
                    fct['count'] = len(times)
                    fct['median'] = numpy.median(times)
                    stores.append(fct)

                if validate:
                    if stop_if_error:
                        up = inst.validate(results, data=data)
                    else:
                        try:
                            up = inst.validate(results, data=data)
                        except Exception as e:  # pylint: disable=W0703
                            msg = str(e).replace("\n", " ").replace(",", " ")
                            up = {'error': msg}
                    if up is not None:
                        for fct in stores:
                            fct.update(up)
                for fct in stores:
                    yield fct
                next(loop)  # pylint: disable=R1708

    def profile(self, kwargs, fct):
        """
        Checks if a profiler applies on this set
        of parameters, then profiles function *fct*.

        @param      kwargs      dictionary of parameters
        @param      fct         function to measure
        """
        if self.profilers:
            for prof in self.profilers:
                if prof.match(**kwargs):
                    prof.profile(fct, **kwargs)
