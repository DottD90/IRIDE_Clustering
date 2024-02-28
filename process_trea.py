#!/usr/bin/env python3
"""
Read all the zip files in a directory extract the content and save it to a new
zipfile without the original zip file directory structure.
"""
# - Python modules
import os
import argparse
import shutil
import zipfile


def main() -> None:
    # - Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Read all the zip files in a directory extract "
                    "the content and save it to a new zipfile "
                    "without the original zip file directory structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # - Input directory
    parser.add_argument('in_dir', type=str,
                        help='Input directory containing the zip files.')

    # - Optional arguments
    # - Output directory
    parser.add_argument("-D", "--out_dir", type=str,
                        default=os.getcwd(),
                        help='Output directory where '
                             'the results will be saved.')
    args = parser.parse_args()

    # - Verify if input directory exists
    if not os.path.exists(args.in_dir):
        raise FileNotFoundError(f"# - Input directory {args.in_dir} "
                                f"does not exist.")
    # - Create the output directory if it does not exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"# - Created output directory {args.out_dir}")

    # - Loop over the files in the input directory
    z_file_list = [os.path.join(args.in_dir, f)
                   for f in os.listdir(args.in_dir) if f.endswith('.zip')]
    for z_file in z_file_list:
        print(f"# - Reading file {z_file}")
        with zipfile.ZipFile(z_file, 'r') as zip_ref:
            # - Extract all files to the current directory
            print(f"# - Extracting files from {z_file}")
            zip_ref.extractall(args.out_dir)

        # - Move the extracted file tho the base directory
        # - This is a temporary fix for TRE-A data.
        temporary_dir\
            = os.path.join(args.out_dir, 'data', 'IRIDE', 'S3-02-SNT-05')
        # - List temporary directory content
        z_file_base = os.path.basename(z_file).replace('.zip', '')
        ftm_list = [os.path.join(temporary_dir, f)
                    for f in os.listdir(temporary_dir)
                    if f.endswith('.xml') or f.endswith('.csv')]
        ftm_list = filter(lambda x: z_file_base in x, ftm_list)

        # - Define directory file name
        file_name = os.path.basename(z_file).replace('.zip', '')
        # - Create a new zipfile
        print(f"# - Creating new zipfile {file_name}.zip")
        with zipfile.ZipFile(os.path.join(args.out_dir, f"{file_name}.zip"),
                             'w') as zip_ref:
            # -  Loop through all files in the list
            for file_name in ftm_list:
                # Add the file to the zip file
                zip_ref.write(file_name, os.path.basename(file_name))

    # - Remove the temporary directory
    shutil.rmtree(os.path.join(args.out_dir, 'data'))


if __name__ == "__main__":
    main()
