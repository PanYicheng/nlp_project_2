import sys
import pickle
from utils import dictionary

sys.modules['dictionary'] = dictionary

with open('/mnt/ssd/l2w/models/tbooks/vocab.pickle', 'rb') as f:
    dic = pickle.load(f)

del sys.modules['dictionary']

with open('/mnt/ssd/l2w/models/tbooks/vocab-converted.pickle', 'wb') as f:
    pickle.dump(dic, f)
