#!/usr/bin/env python
import argparse
import logging


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Author Profiler for PAN 2016')
    argparser.add_argument('-l', '--log-level', dest='log_level', type=str, default='INFO',
                           help='Set log level (DEBUG, INFO, ERROR)')

    argparser.add_argument('-i', '--tira_input', dest='tira_input', type=str, required=True,
                           help='Path to the corpus for which the gender and age of the authors have to be predicted')

    argparser.add_argument('-r', '--tira_run', dest='input_run', type=str, required=False,
                           help='Input run')

    argparser.add_argument('-o', '--tira_output', dest='tira_output', type=str, required=True,
                           help='Output directory')

    args = argparser.parse_args()
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOGFMT)
    print(args.tira_input)
    print(args.input_run)
    print(args.tira_output)
