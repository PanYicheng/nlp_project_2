import argparse
import os
import numpy as np

parser = argparse.ArgumentParser('Split train data into train, discriminator')
parser.add_argument('data', help='text file to split')
parser.add_argument('out_dir', help='directory to write output to')
parser.add_argument('--disc_train_frac', type=float, default=0.2,
                    help='what fraction of the data to use for discriminator training')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    print('Making directory {}'.format(args.out_dir))
    os.makedirs(args.out_dir)

n_lines = 0
with open(args.data, 'r') as lines:
    for line in lines:
        n_lines += 1
    lines.seek(0)
    disc_train_limit = np.ceil(n_lines * args.disc_train_frac)
    disc_train_done = False
    line_buff = []
    for line in lines:
        line_buff.append(line.strip())
        if not disc_train_done:
            if len(line_buff) == disc_train_limit:
                with open(os.path.join(args.out_dir, 'disc_train.txt'), 'w') as test_file:
                    test_file.write('\n'.join(line_buff))
                line_buff = []
                disc_train_done = True
with open(os.path.join(args.out_dir, 'train.txt'), 'w') as train_file:
    train_file.write('\n'.join(line_buff))

print('Total input data lines: {}\n'
      'Disc data lines       : {}\n'
      'Remaining train lines : {}'
      .format(n_lines, disc_train_limit, n_lines - disc_train_limit))
