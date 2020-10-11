from multiprocessing import Process
import multiprocessing as mp
# from billiard.context import Process
import os
import pickle
import socket
import sys
import threading
import time
import traceback
from uuid import uuid4

import Pyro4

from autoflow.utils.logging_ import get_logger
from autoflow.utils.sys_ import get_trance_back_msg


class Worker(object):
    """
    The worker is responsible for evaluating a single configuration on a single budget at a time.
    Communication to the individual workers goes via the nameserver, management of the worker-pool and job
    scheduling is done by the Dispatcher and jobs are determined by the Master. In distributed systems, each
    cluster-node runs a Worker-instance. To implement your own worker, overwrite the `__init__`- and the `compute`-method.
    The first allows to perform inital computations, e.g. loading the dataset, when the worker is started, while the
    latter is repeatedly called during the optimization and evaluates a given configuration yielding the associated loss.
    """

    def __init__(
            self,
            run_id,
            nameserver=None,
            nameserver_port=None,
            host=None,
            worker_id=None,
            timeout=None,
            debug=False
    ):
        """

        Parameters
        ----------
        run_id: anything with a __str__ method
            unique id to identify individual HpBandSter run
        nameserver: str
            hostname or IP of the nameserver
        nameserver_port: int
            port of the nameserver
        logger: logging.logger instance
            logger used for debugging output
        host: str
            hostname for this worker process
        worker_id: anything with a __str__method
            if multiple workers are started in the same process, you MUST provide a unique id for each one of them using the `id` argument.
        timeout: int or float or None
            specifies the timeout a worker will wait for a new after finishing a computation before shutting down.
            Towards the end of a long run with multiple workers, this helps to shutdown idling workers. We recommend
            a timeout that is roughly half the time it would take for the second largest budget to finish.
            The default (None) means that the worker will wait indefinitely and never shutdown on its own.
        """
        self.debug = debug
        self.run_id = run_id
        self.host = host
        self.nameserver = nameserver
        self.nameserver_port = nameserver_port
        self.worker_id = "ambo.run_%s.worker.%s.%i" % (self.run_id, socket.gethostname(), os.getpid())
        self.manifest_id = uuid4().hex[-8:]
        self.timeout = timeout
        self.timer = None
        worker_id = str(worker_id)
        if not worker_id is None:
            self.worker_id += f".{worker_id}"
            self.manifest_id = str(worker_id)

        self.thread = None

        self.logger = get_logger(f"Worker[{self.manifest_id}]")  # 分布式环境下的命名问题

        self.busy = False
        self.thread_cond = threading.Condition(threading.Lock())

    def load_nameserver_credentials(self, working_directory, num_tries=60, interval=1):
        """
        loads the nameserver credentials in cases where master and workers share a filesystem

        Parameters
        ----------
            working_directory: str
                the working directory for the HPB run (see master)
            num_tries: int
                number of attempts to find the file (default 60)
            interval: float
                waiting period between the attempts
        """
        fn = os.path.join(working_directory, 'HPB_run_%s_pyro.pkl' % self.run_id)

        for i in range(num_tries):
            try:
                with open(fn, 'rb') as fh:
                    self.nameserver, self.nameserver_port = pickle.load(fh)
                return
            except FileNotFoundError:
                self.logger.warning('config file %s not found (trail %i/%i)' % (fn, i + 1, num_tries))
                time.sleep(interval)
            except:
                raise
        raise RuntimeError("Could not find the nameserver information, aborting!")

    def run(self, background=False, concurrent_type="process"):
        """
        Method to start the worker.

        Parameters
        ----------
            background: bool
                If set to False (Default). the worker is executed in the current thread.
                If True, a new daemon thread is created that runs the worker. This is
                useful in a single worker scenario/when the compute function only simulates
                work.
        """
        if background:
            if concurrent_type == "process":
                # ctx = mp.get_context('spawn')
                self.process = mp.Process(
                    target=self._run,
                    name='worker %s process' % self.worker_id
                )
                self.process.daemon = True
                self.process.start()
            elif concurrent_type == "thread":
                # maybe in a same thread
                self.worker_id += f"_{threading.get_ident()}"
                self.thread = threading.Thread(target=self._run, name='worker %s thread' % self.worker_id)
                self.thread.daemon = True
                self.thread.start()
        else:
            self._run()

    def _run(self):
        # initial ping to the dispatcher to register the worker

        try:
            with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
                self.logger.debug('WORKER: Connected to nameserver %s' % (str(ns)))
                dispatchers = ns.list(prefix="ambo.run_%s.dispatcher" % self.run_id)
        except Pyro4.errors.NamingError:
            if self.thread is None:
                raise RuntimeError(
                    'No nameserver found. Make sure the nameserver is running at that the host (%s) and port (%s) are correct' % (
                        self.nameserver, self.nameserver_port))
            else:
                self.logger.error(
                    'No nameserver found. Make sure the nameserver is running at that the host (%s) and port (%s) are correct' % (
                        self.nameserver, self.nameserver_port))
                exit(1)
        except:
            raise

        for dn, uri in dispatchers.items():
            try:
                self.logger.debug('WORKER: found dispatcher %s' % dn)
                with Pyro4.Proxy(uri) as dispatcher_proxy:
                    dispatcher_proxy.trigger_discover_worker()
            except Pyro4.errors.CommunicationError:
                self.logger.debug('WORKER: Dispatcher did not respond. Waiting for one to initiate contact.')
                pass
            except:
                raise

        if len(dispatchers) == 0:
            self.logger.debug('WORKER: No dispatcher found. Waiting for one to initiate contact.')

        self.logger.info(f"WORKER(worker_id='{self.worker_id}'): start listening for jobs")

        self.pyro_daemon = Pyro4.core.Daemon(host=self.host)

        with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
            uri = self.pyro_daemon.register(self, self.worker_id)
            ns.register(self.worker_id, uri)

        self.pyro_daemon.requestLoop()

        with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
            ns.remove(self.worker_id)

    def compute(self, config_id, config, config_info, budget, working_directory):
        """ The function you have to overload implementing your computation.

        Parameters
        ----------
        config_id: tuple
            a triplet of ints that uniquely identifies a configuration. the convention is
            id = (iteration, budget index, running index) with the following meaning:
            - iteration: the iteration of the optimization algorithms. E.g, for Hyperband that is one round of Successive Halving
            - budget index: the budget (of the current iteration) for which this configuration was sampled by the optimizer. This is only nonzero if the majority of the runs fail and Hyperband resamples to fill empty slots, or you use a more 'advanced' optimizer.
            - running index: this is simply an int >= 0 that sort the configs into the order they where sampled, i.e. (x,x,0) was sampled before (x,x,1).
        config: dict
            the actual configuration to be evaluated.
        budget: float
            the budget for the evaluation
        working_directory: str
            a name of a directory that is unique to this configuration. Use this to store intermediate results on lower budgets that can be reused later for a larger budget (for iterative algorithms, for example).
        Returns
        -------
        dict:
            needs to return a dictionary with two mandatory entries:
                - 'loss': a numerical value that is MINIMIZED
                - 'info': This can be pretty much any build in python type, e.g. a dict with lists as value. Due to Pyro4 handling the remote function calls, 3rd party types like numpy arrays are not supported!
        """

        raise NotImplementedError(
            "Subclass ambo.distributed.worker and overwrite the compute method in your worker script")

    @Pyro4.expose
    @Pyro4.oneway
    def start_computation(self, callback, config_id, *args, **kwargs):

        with self.thread_cond:
            while self.busy:
                self.thread_cond.wait()
            self.busy = True
        if not self.timeout is None and not self.timer is None:
            self.timer.cancel()
        self.logger.info('WORKER: start processing job %s' % str(config_id))
        self.logger.debug('WORKER: args: %s' % (str(args)))
        self.logger.debug('WORKER: kwargs: %s' % (str(kwargs)))
        try:
            result = {'result': self.compute(*args, config_id=config_id, **kwargs),
                      'exception': None}
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(kwargs)
            failed_info = get_trance_back_msg()
            if self.debug:
                self.logger.error("re-raise exception")
                raise sys.exc_info()[1]
            result = {'result': None,
                      'exception': failed_info}
        finally:
            self.logger.debug('WORKER: done with job %s, trying to register it.' % str(config_id))
            with self.thread_cond:
                self.busy = False
                callback.register_result(config_id, result)
                self.thread_cond.notify()
        self.logger.info('WORKER: registered result for job %s with dispatcher' % str(config_id))
        if not self.timeout is None:
            self.timer = threading.Timer(self.timeout, self.shutdown)
            self.timer.daemon = True
            self.timer.start()
        return (result)

    @Pyro4.expose
    def is_busy(self):
        return (self.busy)

    @Pyro4.expose
    @Pyro4.oneway
    def shutdown(self):
        self.logger.debug('WORKER: shutting down now!')
        self.pyro_daemon.shutdown()
        if not self.thread is None:
            self.thread.join()
