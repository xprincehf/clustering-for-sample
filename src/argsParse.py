# coding = utf-8

import argparse


DEFAULT_PARSER = {
    'task': {
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
    'number': {
        'args': ['-n', '--number'],
        'kwargs': {
            'action': 'store',
            'dest': 'number',
            'type': int,
            'required': True,
            'help': 'the number of sample',
        }
    },
    'pattern': {
        'args': ['-p', '--pattern'],
        'kwargs': {
            'action': 'store',
            'dest': 'pattern',
            'required': False,
            'help': 'regular pattern',
            'default': None,
        }
    },
    'mode': {
        'args': ['-m', '--mode'],
        'kwargs': {
            'action': 'store',
            'dest': 'mode',
            'required': False,
            'help': 'exclude or include',
            'default': None,
        }
    }
}


def args_parse():
    parser = argparse.ArgumentParser()

    for arg_name, arg_value in DEFAULT_PARSER.items():
        parser.add_argument(arg_value['args'], arg_value['kwargs'])

    return parser

