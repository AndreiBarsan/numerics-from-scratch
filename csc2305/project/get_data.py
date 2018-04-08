#!/usr/bin/env python3
"""Downloads the Bundle Adjustment in the Large datasets relevant for this
Should be run from the root directory of the project. (Alternatively you can
just use the '-o' parameter to specify the output directory.)"""

import argparse
import bz2
import os
from urllib.request import urlretrieve

import shutil

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/"

FILE_NAMES = {
    'trafalgar': [
        "problem-21-11315-pre.txt",
        "problem-39-18060-pre.txt",
        "problem-50-20431-pre.txt",
        "problem-126-40037-pre.txt",
        "problem-138-44033-pre.txt",
        "problem-161-48126-pre.txt",
        "problem-170-49267-pre.txt",
        "problem-174-50489-pre.txt",
        "problem-193-53101-pre.txt",
        "problem-201-54427-pre.txt",
        "problem-206-54562-pre.txt",
        "problem-215-55910-pre.txt",
        "problem-225-57665-pre.txt",
        "problem-257-65132-pre.txt",
    ],
    'dubrovnik': [
        "problem-16-22106-pre.txt",
        "problem-88-64298-pre.txt",
        "problem-135-90642-pre.txt",
        "problem-142-93602-pre.txt",
        "problem-150-95821-pre.txt",
        "problem-161-103832-pre.txt",
        "problem-173-111908-pre.txt",
        "problem-182-116770-pre.txt",
        "problem-202-132796-pre.txt",
        "problem-237-154414-pre.txt",
        "problem-253-163691-pre.txt",
        "problem-262-169354-pre.txt",
        "problem-273-176305-pre.txt",
        "problem-287-182023-pre.txt",
        "problem-308-195089-pre.txt",
        "problem-356-226730-pre.txt",
    ],
    'venice': [
        "problem-52-64053-pre.txt",
        "problem-89-110973-pre.txt",
        "problem-245-198739-pre.txt",
        "problem-427-310384-pre.txt",
        "problem-744-543562-pre.txt",
        "problem-951-708276-pre.txt",
        "problem-1102-780462-pre.txt",
        "problem-1158-802917-pre.txt",
        "problem-1184-816583-pre.txt",
        "problem-1238-843534-pre.txt",
        "problem-1288-866452-pre.txt",
        "problem-1350-894716-pre.txt",
        "problem-1408-912229-pre.txt",
        "problem-1425-916895-pre.txt",
        "problem-1473-930345-pre.txt",
        "problem-1490-935273-pre.txt",

        # These sequences are all sort of the same order of magnitude, so we
        # skip them for the sake of saving some disk space and time.
        # "problem-1521-939551-pre.txt",
        # "problem-1544-942409-pre.txt",
        # "problem-1638-976803-pre.txt",
        # "problem-1666-983911-pre.txt",
        # "problem-1672-986962-pre.txt",
        # "problem-1681-983415-pre.txt",
        # "problem-1682-983268-pre.txt",
        # "problem-1684-983269-pre.txt",
        # "problem-1695-984689-pre.txt",
        # "problem-1696-984816-pre.txt",
        # "problem-1706-985529-pre.txt",
        # "problem-1776-993909-pre.txt",
        # "problem-1778-993923-pre.txt",
    ]
}


def get_file(dataset_name, fname, target_root):
    fpath = os.path.join(target_root, dataset_name, fname)
    if os.path.isfile(fpath):
        print("File [{}] from dataset [{}] already downloaded and extracted.".format(fpath, dataset_name))
        return

    bz_fname = fname + ".bz2"
    bz_fpath = os.path.join(target_root, dataset_name, bz_fname)

    url = os.path.join(BASE_URL, dataset_name, bz_fname)
    if not os.path.isfile(bz_fpath):
        print("Downloading [{}]...".format(url))
        urlretrieve(url, bz_fpath, reporthook=None)

    print("Extracting archive [{}]...".format(bz_fpath))
    with bz2.open(bz_fpath, 'rb') as input_bz, open(fpath, 'wb') as extracted_f:
        shutil.copyfileobj(input_bz, extracted_f)
    print("Extracted archive to [{}].".format(fpath))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--out-dir', '-o', type=str, default='data',
                        help="Directory to download data to.")

    return parser.parse_args(argv)


def main():
    args = parse_args()

    for dataset_name in FILE_NAMES.keys():
        os.makedirs(os.path.join('data/{}'.format(dataset_name)), exist_ok=True)
        print("\n\nGetting dataset [{}]...\n\n".format(dataset_name))
        for fname in FILE_NAMES[dataset_name]:
            get_file(dataset_name, fname, args.out_dir)


if __name__ == '__main__':
    main()
