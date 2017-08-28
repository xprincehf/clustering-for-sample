# coding = utf-8

import argparse


DEFAULT_PARSER = {
    'cluster': {
        'args': ['-c', '--cluster'],
        'kwargs': {
            'action': 'store',
            'dest': 'cluster',
            'required': True,
            'help': 'the cluster method',
        }
    },
    'source': {
        'args': ['-s', '--source'],
        'kwargs': {
            'action': 'store',
            'dest': 'source',
            'required': True,
            'help': 'the data source of task',
        }
    },
    'field': {
        'args': ['-f', '--field'],
        'kwargs': {
            'action': 'store',
            'dest': 'field',
            'required': True,
            'help': 'the target field to clustering and filter'
        }
    },
    'target': {
        'args': ['-t', '--target'],
        'kwargs': {
            'action': 'store',
            'dest': 'target',
            'required': True,
            'help': 'output file path',
        }
    },
    'number': {
        'args': ['-n', '--number'],
        'kwargs': {
            'action': 'store',
            'dest': 'number',
            'type': int,
            'required': False,
            'help': 'the number of sample',
        }
    },
}


def args_parse():
    parser = argparse.ArgumentParser()

    for arg_name, arg_value in DEFAULT_PARSER.items():
        parser.add_argument(*arg_value['args'], **arg_value['kwargs'])

    return parser

