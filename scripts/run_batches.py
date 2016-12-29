import sys
import source.runner as runner


def main(arguments):
    """
    Run this script to run all the batches files of configurations in the
    folder at 'mypath': results will be put in the folder at 'resultspath'.
    If the folder exists already, an exception will be raised, to prevent
    overwriting previous results. Otherwise the folder will be created.
    """
    mypath = "/home/lorenzo/Scrivania/Polimi/Thesis/code/onlineSG/games"
    resultspath = "/home/lorenzo/Scrivania/Polimi/Thesis/code/onlineSG/results4"
    r = runner.Runner(mypath, resultspath)
    r.run()


def main2(arguments):
    """
    Run this script to run the batch file of configurations
    in 'conf'': results will be put in the folder at
    'resultspath'. If the folder exists already, nothing will happen.
    Otherwise the folder will be created.
    """
    results = "/home/lorenzo/Scrivania/Polimi/Thesis/code/onlineSG/results3"
    conf = "/home/lorenzo/Scrivania/Polimi/Thesis/code/onlineSG/games/conf.csv"
    b = runner.Batch(conf, results)
    b.parse_batch()
    b.run()


if __name__ == '__main__':
    main2(sys.argv)
