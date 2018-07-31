#!/usr/bin/env python
from datetime import datetime
import socket
import os
print("start python: {}, {}, {}".format(socket.gethostname(), os.environ.get('PMIX_RANK'), datetime.now()))

# suppress warnings from earlier versions of h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import multiprocessing
import sys
import glob
import random
import time
import re
import signal
import traceback
try:
    import numpy as np
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False    

class JobTimeoutException(Exception):
    def __init__(self, jobstack=None):
        super(JobTimeoutException, self).__init__()
        if jobstack is None:
            self.jobstack = traceback.format_stack()
        else:
            self.jobstack = jobstack

def sigalrm_handler(signum, frame):
    raise JobTimeoutException()

class SimpleFileCopier(object):
    def copy(self, srcfile, dstfile, expected_bytes):
        with open(srcfile, 'rb') as f_in:
            data = f_in.read(expected_bytes)
        assert len(data) == expected_bytes
        with open(dstfile, 'wb') as f_out:
            f_out.write(data)
        return expected_bytes

class ClimateFileCopier(object):
    def __init__(self, dtype, channels):
        self.dtype = dtype
        self.channels = channels
        
    def copy(self, srcfile, dstfile, expected_bytes):
        # read data and labels from input hdf5 file
        with h5py.File(srcfile, 'r', driver='core',
                       backing_store=False, libver='latest') as f:
            data = f['climate']['data'][...]
            labels = f['climate']['labels'][...]
        # if a channel list is available, use just those
        if self.channels:
            data = data[self.channels,:,:]
        # convert data to desired type, reduce labels to int8
        data = data.astype(self.dtype)
        labels = labels.astype(np.int8)

        # now write a new file
        with h5py.File(dstfile, 'w', driver='core', libver='latest') as f:
            c = f.create_group('climate')
            c.create_dataset('data', data=data)
            c.create_dataset('labels', data=labels)
            if self.channels:
                c.create_dataset('channels',
                                 data=np.array(self.channels, dtype=np.int32))

        # return the size of the resulting file
        return os.stat(dstfile).st_size

def do_copy(f, target_dir, timeout, cvt):
    if timeout:
        signal.signal(signal.SIGALRM, sigalrm_handler)
        signal.alarm(timeout)

    return cvt.copy(f['srcname'],
                    os.path.join(target_dir, f['dstname']),
                    f['srcsize'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('srcpaths', type=str, nargs='+',
                        help='path to source files')
    parser.add_argument('--targets', type=str, nargs='+', required=True,
                        help='paths to target directory')
    parser.add_argument('--mkdir', action='store_true',
                        help='create target directory if it does not exist')
    parser.add_argument('--inorder', action='store_true',
                        help='do not shuffle input files')
    parser.add_argument('--seed', type=int,
                        help='random seed for shuffles')
    parser.add_argument('--trim', type=int,
                        help='trim input file list to specified size')
    parser.add_argument('--counts', type=int, nargs='+', required=True,
                        help='number of files desired on each rank per source/target pair')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of file copy worker threads')
    parser.add_argument('--timeout', type=int,
                        help='maximum time (in seconds) to wait for file copy')
    parser.add_argument('--retries', type=int,
                        help='maxmimum number of file copy retries allowed')
    parser.add_argument('--quiet', action='store_true',
                        help='suppress all normal output')
    parser.add_argument('--cvt', type=str,
                        help='data conversion to be performed')
    args = parser.parse_args()
    
    # multiprocessing pool must be created before MPI is initialized so
    #  that MPI doesn't get confused by forks and subprocesses
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(processes = args.workers)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # do mpi stuff inside of main so that subprocesses don't get confused
    try:
        from mpi4py import rc
        rc.recv_mprobe = False

        print("import MPI: {}, {}, {}".format(socket.gethostname(), os.environ.get('PMIX_RANK'), datetime.now()))
        from mpi4py import MPI
        print("post MPI: {}, {}, {}".format(socket.gethostname(), os.environ.get('PMIX_RANK'), datetime.now()))
        have_mpi = True
        comm_rank = MPI.COMM_WORLD.Get_rank()
        comm_size = MPI.COMM_WORLD.Get_size()
    except ImportError:
        have_mpi = False
        comm_rank = 0
        comm_size = 1

    if args.seed:
        # make sure each rank has a different seed
        random.seed(args.seed + comm_rank)

    #some sanity checking srcpaths
    assert len(args.srcpaths) == len(args.targets), "make sure you specify as many sources as destination paths" if comm_rank==0 else None
    assert len(args.srcpaths) == len(args.counts), "make sure you specify a file count for every source path" if comm_rank==0 else None

    #create zipped list:
    ziplist = zip(args.srcpaths, args.targets, args.counts)

    if comm_rank == 0:
        print("Staging in total {dircount} directories".format(dircount=len(ziplist)))

    for stageid,dirtup in enumerate(ziplist):

        #extract
        source = dirtup[0]
        target = dirtup[1]
        count = dirtup[2]

        # step 1: construct the initial list of files, get their sizes and assign
        #  which rank will do the copy
        if comm_rank == 0:
            t_scan_start = time.time()

            print("Staging {cnt} files in {src} to {dst}".format(cnt=count, src=source, dst=target))

            files = []
            if os.path.isdir(source):
                files.extend([ {'srcname': os.path.join(source, f)} for f in os.listdir(source) ])
            else:
                files.extend([ {'srcname': f} for f in glob.glob(source) ])

            if len(files) < count:
                print('Not enough files found! Needed {}, got {}!'.format(count,len(files)))
                exit(1)
            
            # shuffle list
            if not args.inorder:
                random.shuffle(files)

            # trim list using smaller of input trim or count*ranks
            maxfiles = count * comm_size
            if args.trim:
                maxfiles = min(maxfiles, args.trim)
            files = files[0:maxfiles]

            # now go through and get sizes/etc.
            totalbytes = 0
            for idx, f in enumerate(files):
                f['idx'] = idx
                f['owner'] = idx % comm_size
                bytes = os.stat(f['srcname']).st_size
                f['srcsize'] = bytes
                totalbytes += bytes
                f['dstname'] = os.path.split(f['srcname'])[1]

            if have_mpi:
                MPI.COMM_WORLD.bcast(files, root=0)

            t_scan_end = time.time()
            if not args.quiet:
                print('Scan complete ({:.2f} s): {} files, {:.1f} MB total'.format(t_scan_end - t_scan_start,
                                                                                   len(files),
                                                                                   totalbytes / 1048576.))
                sys.stdout.flush()
        else:
            # get file list from first rank
            files = MPI.COMM_WORLD.bcast(None, root=0)

        # create the target directory if needed
        target_dir = re.sub('%', str(comm_rank), target)
    
        if not os.path.isdir(target_dir):
            if args.mkdir:
                os.makedirs(target_dir)
            else:
                print('ERROR: target directory {} does not exist - use --mkdir to create it'.format(target_dir))
                exit(1)

        # copy our chunk of the files, using subprocesses with timeouts
        t_copy_start = time.time()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        #pool = multiprocessing.Pool(processes = args.workers)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # create the converter we will use
        if args.cvt is None or args.cvt == 'raw':
            cvt = SimpleFileCopier()
        elif args.cvt.startswith('climate:'):
            if not have_h5py:
                print('ERROR: climate file conversion not possible without h5py!')
                exit(1)
            params = args.cvt.split(':')
            dtype = np.float16 if params[1] == 'fp16' else np.float32
            if len(params) > 2:
                channels = [ int(x) for x in params[2].split(',') ]
            else:
                channels = None
            cvt = ClimateFileCopier(dtype=dtype, channels=channels)
        else:
            print('ERROR: unknown conversion type: {}'.format(args.cvt))
            exit(1)

        # request a copy for each file we own
        owned_files = [ f for f in files if (f['owner'] == comm_rank) ]
        pending = [ (f['idx'],
                     pool.apply_async(do_copy,
                                      (f,
                                       target_dir,
                                       args.timeout,
                                       cvt))) for f in owned_files ]
        dstsizes = {}
        retry_count = 0
        while pending:
            still_pending = []
            for p in pending:
                try:
                    dsize = p[1].get(0) # do not wait
                    dstsizes[p[0]] = dsize
                    continue
                except multiprocessing.TimeoutError:
                    still_pending.append(p)
                    continue
                except JobTimeoutException:
                    print('WARNING: timeout for read of {} on rank {} - retrying...'.format(f['srcname'], comm_rank))
                except Exception as e:
                    print('WARNING: error on read of {} on rank {}: {}'.format(f['srcname'],
                                                                               comm_rank,
                                                                               traceback.format_exc(e)))

                if args.retries and retry_count >= args.retries:
                    print('ERROR: too many file copy retries on rank {}'.format(comm_rank))
                    exit(1)

                retry_count += 1
                f = files[p[0]]
                assert(f['idx'] == p[0])
                still_pending.append((f['idx'],
                                      pool.apply_async(do_copy,
                                                       (f,
                                                        target_dir,
                                                        args.timeout,
                                                        cvt))))
            pending = still_pending

        if have_mpi:
            # every rank gets a list of all the dstsizes from each other rank
            all_dstsizes = MPI.COMM_WORLD.allgather(dstsizes)
            
        t_copy_end = time.time()
                
        if comm_rank == 0 and not args.quiet:
            elapsed = t_copy_end - t_copy_start
            eff_bw = 1e-9 * totalbytes / elapsed 
            print('Copy complete ({:.2f} s): {:.2f} GB/s'.format(elapsed, eff_bw))
            sys.stdout.flush()

        # step 3: distribute copies of data via mpi, if needed
        if len(files) < (count * comm_size):
            t_dist_start = time.time()

            # choose the files we want a copy of
            needed_count = count - len(owned_files)
            assert needed_count >= 0
            if needed_count > 0:
                remote_idxs = [ f['idx'] for f in files if f['owner'] != comm_rank ]
                random.shuffle(remote_idxs)
                needed = set(remote_idxs[0:needed_count])
            else:
                needed = None

            # mpi4py's irecv requires an explicit buffer for large transfers - make it a
            #   little larger than the largest file we expect to receive to account for
            #   pickling overhead
            max_recv_size = max(all_dstsizes[files[idx]['owner']][idx] for idx in needed)
            recv_buf = bytearray(max_recv_size + 4096)
            recv_bytes = 0

            # maximize parallelism by having ranks send data along a "ring" whose
            #  step size grows from 1 to size-1
            for step in xrange(1, comm_size):
                send_rank = (comm_rank + step) % comm_size
                recv_rank = (comm_rank + comm_size - step) % comm_size

                # use irecv/send/wait pattern for all communication to avoid hangs
                to_recv = [ i for i in needed if files[i]['owner'] == recv_rank ]
                req = MPI.COMM_WORLD.irecv(recv_buf, source=send_rank)
                MPI.COMM_WORLD.send(to_recv, dest=recv_rank)
                to_send = req.wait()

                while to_recv or to_send:
                    # post irecv first, if we have anything left to receive
                    if to_recv:
                        req = MPI.COMM_WORLD.irecv(recv_buf, source=recv_rank)

                    # now send data if needed
                    if to_send:
                        send_idx = to_send.pop(0)
                        send_size = dstsizes[send_idx]
                        with open(os.path.join(target_dir, files[send_idx]['dstname']), 'rb') as f:
                            send_data = f.read(send_size)
                            assert len(send_data) == send_size
                        MPI.COMM_WORLD.send(send_data, dest=send_rank)

                    # and then process receive data if needed
                    if to_recv:
                        recv_idx = to_recv.pop(0)
                        recv_size = all_dstsizes[recv_rank][recv_idx]
                        recv_data = req.wait()
                        assert(len(recv_data) == recv_size)
                        with open(os.path.join(target_dir, files[recv_idx]['dstname']), 'wb') as f:
                            f.write(recv_data)
                        recv_bytes += recv_size

            # we need a collective operation to make sure everybody is done, and so let's add up the
            #  total bytes transferred (we can compute the number of files directly)
            total_dist_files = comm_size * count - len(files)
            total_dist_size = MPI.COMM_WORLD.reduce(recv_bytes / 1048576., root=0)
        
            t_dist_end = time.time()

            if comm_rank == 0 and not args.quiet:
                elapsed = t_dist_end - t_dist_start
                eff_bw = 1e-9 * 1048576. * total_dist_size / elapsed
                print('Dist complete ({:.2f} s): {} files, {:.1f} MB, {:.2f} GB/s'.format(elapsed,
                                                                                          total_dist_files,
                                                                                          total_dist_size,
                                                                                          eff_bw))

if __name__ == '__main__':
    main()
