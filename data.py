__auther__ = 'yizhangzc'

import numpy as np
from sklearn import utils as skutils

class PAMAP2( object ):
    def __init__( self ):
        self._path              = '../'
        self._channel_num       = 36
        self._length            = 200
        self._user_num          = 8
        self._act_num           = 12

    def load_data( self, test_user=0 ):

        train_x = np.empty( [0, self._length, self._channel_num], dtype=np.float )
        train_y = np.empty( [0], dtype=np.int )

        test_x  = np.empty( [0, self._length, self._channel_num], dtype=np.float )
        test_y  = np.empty( [0], dtype=np.int )

        for user_idx in range( self._user_num ):
            if user_idx == test_user:
                test_x  = np.concatenate( (test_x, np.load(self._path+'processed_data/sub{}_features.npy'.format(user_idx)) ), axis=0 )
                test_y  = np.concatenate( (test_y, np.load(self._path+'processed_data/sub{}_labels.npy'.format(user_idx)) ), axis=0 )
            else:
                train_x  = np.concatenate( (train_x, np.load(self._path+'processed_data/sub{}_features.npy'.format(user_idx)) ), axis=0 )
                train_y  = np.concatenate( (train_y, np.load(self._path+'processed_data/sub{}_labels.npy'.format(user_idx)) ), axis=0 )

        train_x, train_y    = skutils.shuffle( train_x, train_y )
        test_x, test_y      = skutils.shuffle( test_x, test_y )

        return train_x, train_y, test_x, test_y