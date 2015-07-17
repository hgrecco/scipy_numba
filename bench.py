"""A tiny benchmarking framework

The tests are stored in a YAML file.

"""

from __future__ import division, unicode_literals, print_function, absolute_import

import fnmatch
import os
import collections
from timeit import Timer

import yaml

try:
    basestring
except NameError:
    basestring = str


_RESERVED = ('stmt', 'setup', 'number', 'repeat', 'common', 'base')

Result = collections.namedtuple('Result', 'name time base')

_CACHE = {}


_UNITS = ((1e-9, 'ns'), (1e-6, 'us'), (1e-3, 'ms'))

def humanize_time(dt_seconds):
    for value, unit in _UNITS:
        if dt_seconds < 500 * value:
            return '%.2f%s' % (dt_seconds / value, unit)

    return '%.2fs' % dt_seconds


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=collections.OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def timeit(stmt='pass', setup='pass', number=0, repeat=3):
    """Timer function with the same behaviour as running `python -m timeit `
    in the command line.

    :return: best elapsed time in seconds or NaN if the command failed.
    :rtype: float
    """

    if not setup:
        setup = 'pass'

    if not stmt:
        stmt = 'pass'

    key = (stmt, setup, number, repeat)

    if key in _CACHE:
        return _CACHE[key]

    t = Timer(stmt, setup)

    if not number:
        # determine number so that 0.2 <= total time < 2.0
        for i in range(1, 10):
            number = 10**i

            try:
                x = t.timeit(number)
            except:
                print(t.print_exc())
                return float('NaN')

            if x >= 0.2:
                break

    try:
        r = t.repeat(repeat, number)
    except:
        print(t.print_exc())
        return float('NaN')

    result = min(r) / number

    _CACHE[key] = result

    return result


def based_timeit(stmt='pass', setup='pass', number=0, repeat=3, base=''):
    if base:
        base_dt = timeit(base, setup, number, repeat)
    else:
        base_dt = None

    dt = timeit(stmt, setup, number, repeat)

    return dt, base_dt


def time_task(name, content, parent_task):
    task = build_task(content, parent_task)

    if 'stmt' in content or 'setup' in content:
        yield Result(name, *based_timeit(**task))

    for sub_name, sub_content in content.items():

        if sub_name in _RESERVED:
            continue

        for result in time_task(name + (sub_name, ), sub_content, task):
            yield result


def build_task(task, parent_task):
    nt = {}
    nt['setup'] = (parent_task.get('setup', '') + '\n' +
                   task.get('common', {}).get('setup', '') + '\n' +
                   task.get('setup', '')).strip('\n')

    nt['stmt'] = (parent_task.get('stmt', '') + '\n' +
                  task.get('common', {}).get('stmt', '') + '\n' +
                  task.get('stmt', '')).strip('\n')

    nt['base'] = (parent_task.get('base', '') + '\n' +
                  task.get('common', {}).get('base', '') + '\n' +
                  task.get('base', '')).strip('\n')

    nt['number'] = task.get('number', task.get('number', 0))

    nt['repeat'] = task.get('repeat', task.get('repeat', 3))

    return nt


def time_file(filename, number=None, repeat=None):
    """Open a yaml benchmark file and time each statement.

    Yields a tuple with filename, task name, time in seconds.
    """
    with open(filename, 'r') as fp:
        content = ordered_load(fp)

    for result in time_task((), content, {}):
        yield result


def recursive_glob(rootdir='.', pattern='*'):
    """Return a list of files matching the pattern.
    """
    return [os.path.join(folder, filename)
            for folder, _, filenames in os.walk(rootdir)
            for filename in filenames
            if fnmatch.fnmatch(filename, pattern)]


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description='Bench.')
    parser.add_argument('filenames', nargs='*',
                         help='Filenames to be benched.\n'
                              'If none is provided, a discovery will be tried.')
    parser.add_argument('-s', '--start-directory', default='.', dest='start',
                        help="Directory to start discovery ('.' default)")
    parser.add_argument('-p', '--pattern', default='bench*.yaml',
                        help="Pattern to match tests ('bench*.yaml' default)")

    args = parser.parse_args()

    if not args.filenames:
        args.filenames = recursive_glob(args.start, args.pattern)

    print()

    cwd = os.getcwd()

    for filename in args.filenames:
        folder, filename = os.path.split(filename)
        if folder:
            os.chdir(folder)
        print(filename)
        print('-' * len(filename))
        print()
        for result in time_file(filename):
            if result.base is None:
                print('%-40s\t\t%s' % (' > '.join(result.name), humanize_time(result.time)))
            else:
                print('%-40s\t\t%.2fx (base %s)' % (' > '.join(result.name), result.time / result.base, humanize_time(result.base)))
        print()
        if folder:
            os.chdir(cwd)

if __name__ == '__main__':
    main()
