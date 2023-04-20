import argparse

parser = argparse.ArgumentParser(description='Weakly-Supervised Online Action Segmentation')
parser.add_argument('--multi-view', type=bool, default=True, help='multi-view/single-view')
parser.add_argument('--split', type=int, default=1,help='Split number (default: 1)')
parser.add_argument('--dataset-path', default='./data/', help='root path to the dataset (default: ')
parser.add_argument('--MVI', default="SV", help='WPI/PI/SV')
parser.add_argument('--WPI-net', default="TC", help='TC/GRU')
parser.add_argument('--n-iterations', default=66000)

