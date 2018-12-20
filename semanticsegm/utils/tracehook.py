import tensorflow as tf
from tensorflow.python.client import timeline
import os
import signal
try:
    import horovod.tensorflow as hvd
    hvd_rank = hvd.rank()
except:
    hvd_rank = 0

class TraceHook(tf.train.SessionRunHook):
    def __init__(self, steps_to_trace=None, cache_traces=False, trace_dir=None):
        self.prev_step = -1
        self.phase_count = 0

        if isinstance(steps_to_trace,str):
            if ':' in steps_to_trace:
                fields = steps_to_trace.split(':')
                start = int(fields[0]) if(fields[0] != '') else 0
                stop = int(fields[1]) if(fields[1] != '') else 1000000
                step = int(fields[2]) if len(fields) > 2 and fields[2] != '' else 1
                self.steps_to_trace = range(start, stop, step)
            else:
                self.steps_to_trace = [ int(s) for s in steps_to_trace.split(',') ]
        else:
            self.steps_to_trace = steps_to_trace
            
        if cache_traces:
            self.trace_cache = {}
            
            # install a signal handler to write traces on ^C
            def sigint_handler(signum, frame):
                print('Received SIGINT - writing cached traces to disk...')
                self.write_traces()
                # call any other handlers that were registered and then exit
                if(callable(self.prev_sigint_handler)):
                    self.prev_sigint_handler(signum, frame)
                exit(1)
                
            self.prev_sigint_handler = signal.signal(signal.SIGINT,
                                                     sigint_handler)
        else:
            self.trace_cache = None
        self.trace_dir = trace_dir or '.'
            
    def before_run(self, run_context):
        gstep = tf.train.get_global_step()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        return tf.train.SessionRunArgs(fetches=gstep, options=options)

    def after_run(self, run_context, run_values):
        step = run_values.results
        if step == self.prev_step:
            self.phase_count += 1
        else:
            self.phase_count = 0
            self.prev_step = step
        if (self.steps_to_trace is None) or (step in self.steps_to_trace):
            #print 'on step {}, {}'.format(step, self.phase_count)
            fetched_timeline = timeline.Timeline(run_values.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            filename = os.path.join(self.trace_dir,
                                    'timeline_{}_{}_{}.json'.format(step,
                                                                    self.phase_count,
                                                                    hvd_rank))
            if self.trace_cache is not None:
                self.trace_cache[filename] = chrome_trace
            else:
                with open(filename, 'w') as f:
                    f.write(chrome_trace)

    def write_traces(self):
        if self.trace_cache:
            while self.trace_cache:
                filename, trace = self.trace_cache.popitem()
                with open(filename, 'w') as f:
                    f.write(trace)
                
                
