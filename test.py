#!/usr/bin/python3


'''
Runs basic tests on invocations of the command line interface.
'''


import argparse
parser = argparse.ArgumentParser(description='Runs basic tests on invocations of the command line interface.')
parser.add_argument('image_t0', help='path to the GeoTIFF image at t0')
parser.add_argument('image_t1', help='path to the GeoTIFF image at t1')
parser.add_argument('labels', help='path to the CSV labels')
test_args = parser.parse_args()

import subprocess
import itertools


if __name__ == '__main__':
    fixed_args = ('python3', 'run.py', test_args.image_t0, test_args.image_t1, test_args.labels)
    bad_args = ('--bad',)
    init_args = ('--save test --overwrite',)
    product_args = ((None, '--baseline'), (None, '--no-test'), (None, '--save test --overwrite'), (None, '--load test'))

    def call(args):
        print('Testing', args)
        if subprocess.call(' '.join(fixed_args + args), shell=True) != 0: raise Exception('Process should not have failed')

    # Check that failure can be detected properly
    succeeded = False
    try:
        call(bad_args)
        succeeded = True
    except: pass
    if succeeded: raise Exception('Process should have failed')

    # Check that normal invocations do not fail
    call(init_args)
    for args in itertools.product(*product_args):
        args = tuple(filter(lambda x: x != None, args))
        call(args)

    print('Done')
