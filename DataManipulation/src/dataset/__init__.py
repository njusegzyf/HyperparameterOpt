from os.path import abspath, realpath, dirname, join

# point to `data` in the project dir
# for example `D:\ProjsIDEA\HyperparameterOpt\DataManipulation\src\dataset/../../../data`
data_dir_base = abspath(dirname(realpath(__file__)) + '/../../../data')
