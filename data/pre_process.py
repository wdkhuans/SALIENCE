import os
import shutil

import numpy as np
from absl import app
from scipy import stats

from pandas import Series
from sliding_window import sliding_window

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='PyTorch SALIENCE Pre-Process Implementation')
parser.add_argument('--dataset', type=str, default='pamap', metavar='N',
                    help='opp or pamap')
args = parser.parse_args()

if args.dataset == 'opp':
    args.length = 300
    args.overlap = 30
else:
    args.length = 200
    args.overlap = 100



def preprocess_opportunity( ):
    dataset_path    = 'opp/'
    channel_num     = 113

    file_list = [   ['OpportunityUCIDataset/dataset/S1-Drill.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL1.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL2.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL3.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL4.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL5.dat'] ,
                    ['OpportunityUCIDataset/dataset/S2-Drill.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL1.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL2.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL3.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL4.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL5.dat'] ,
                    ['OpportunityUCIDataset/dataset/S3-Drill.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL1.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL2.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL3.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL4.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL5.dat'] ,
                    ['OpportunityUCIDataset/dataset/S4-Drill.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL1.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL2.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL3.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL4.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL5.dat'] ]

    invalid_feature = np.arange( 46, 50 )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(59, 63)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(72, 76)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(85, 89)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(98, 102)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(134, 244)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(245, 249)] )

    lower_bound = np.array([    3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                                3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                                3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                                3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                                10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                                200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                                10000,  10000,  10000,  10000,  250, ])

    upper_bound = np.array([    -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                                -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                                -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                                -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                                -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                                -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                                -10000, -10000, -10000, -10000, -250, ])

    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    for usr_idx in range( 4 ):
        
        # import pdb; pdb.set_trace()
        print( "process data... user{}".format( usr_idx ) )
        time_windows    = np.empty( [0, args.length, channel_num], dtype=np.float )
        act_labels      = np.empty( [0], dtype=np.int )

        for file_idx in range( 6 ):

            filename = file_list[ usr_idx ][ file_idx ]

            file    = dataset_path + filename
            signal  = np.loadtxt( file )
            signal  = np.delete( signal, invalid_feature, axis = 1 )

            data    = signal[:, 1:114].astype( np.float )
            label   = signal[:, 114].astype( np.int )

            label[ label == 0 ] = -1

            label[ label == 101 ] = 0
            label[ label == 102 ] = 1
            label[ label == 103 ] = 2
            label[ label == 104 ] = 3
            label[ label == 105 ] = 4

            # label[ label == 406516 ] = 0
            # label[ label == 406517 ] = 1
            # label[ label == 404516 ] = 2
            # label[ label == 404517 ] = 3
            # label[ label == 406520 ] = 4
            # label[ label == 404520 ] = 5
            # label[ label == 406505 ] = 6
            # label[ label == 404505 ] = 7
            # label[ label == 406519 ] = 8
            # label[ label == 404519 ] = 9
            # label[ label == 406511 ] = 10
            # label[ label == 404511 ] = 11
            # label[ label == 406508 ] = 12
            # label[ label == 404508 ] = 13
            # label[ label == 408512 ] = 14
            # label[ label == 407521 ] = 15
            # label[ label == 405506 ] = 16

            # fill missing values using Linear Interpolation
            data    = np.array( [Series(i).interpolate(method='linear') for i in data.T] ).T
            data[ np.isnan( data ) ] = 0.

            # normalization
            diff = upper_bound - lower_bound
            data = ( data - lower_bound ) / diff

            data[ data > 1 ] = 1.0
            data[ data < 0 ] = 0.0

            #sliding window
            data    = sliding_window( data, (args.length, channel_num), (args.overlap, 1) )
            label   = sliding_window( label, args.length, args.overlap )
            label   = stats.mode( label, axis=1 )[0][:,0]

            #remove non-interested time windows (label==-1)
            invalid_idx = np.nonzero( label < 0 )[0]
            data        = np.delete( data, invalid_idx, axis=0 )
            label       = np.delete( label, invalid_idx, axis=0 )

            time_windows    = np.concatenate( (time_windows, data), axis=0 )
            act_labels      = np.concatenate( (act_labels, label), axis=0 )

        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), time_windows )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), act_labels )                
        print( "sub{} finished".format( usr_idx) )


def preprocess_pamap2( ):
    dataset_path    = 'pamap/'
    channel_num     = 36

    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    lowerBound = np.array( [
        -18.1809,       -10.455566,     -7.7453649,     -18.321083,
        -10.492066,     -7.6882524,     -3.9656347,     -2.54338,
        -4.6695066,     -40.503183,     -71.010566,     -68.132566,
        -4.0144814,     -2.169227,      -10.6296,       -4.1517899,
        -2.09331,       -10.3659,       -1.0943914,     -1.7640771,
        -0.85873557,    -35.379757,     -67.172728,     -42.7236,
        -1.540593,      -15.741095,     -12.220085,     -1.4369205,
        -15.282295,     -11.544785,     -3.2952385,     -2.2376485,
        -4.7753095,     -92.0835,       -54.226,        -38.719725
    ] )

    upperBound = np.array( [
        8.4245698,      18.942083,      10.753683,      8.45324,        19.126415,
        10.8359,        4.0320432,      2.9766798,      4.4932249,      68.037749,
        41.263183,      28.875164,      4.8361714,      21.236957,      9.54976,
        4.7052142,      21.4364,        9.7664,         1.13992,        1.7279,
        0.84088028,     40.9678,        15.5543,        53.282157,      26.630485,
        24.1561,        6.9268395,      26.4967,        23.778745,      6.78438,
        3.505307,       1.5628585,      6.4969625,      5.87456,        44.9736,
        61.16018    ] )

    file_list = [
        'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
        'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat'  ]

    for usr_idx in range(len(file_list)):

        file    = dataset_path + 'Protocol/' + file_list[usr_idx]
        data    = np.loadtxt( file )

        label   = data[:,1].astype( int )
        label[label == 0]   = -1
        label[label == 1]   = 0         # lying
        label[label == 2]   = 1         # sitting
        label[label == 3]   = 2         # standing
        label[label == 4]   = 3         # walking
        label[label == 5]   = 4         # running
        label[label == 6]   = 5         # cycling
        label[label == 7]   = 6         # nordic walking
        label[label == 12]  = 7         # ascending stairs
        label[label == 13]  = 8         # descending stairs
        label[label == 16]  = 9         # vacuum cleaning
        label[label == 17]  = 10        # ironing
        label[label == 24]  = 11        # rope jumping

        # fill missing values
        valid_idx   = np.concatenate( (np.arange(4, 16), np.arange(21, 33), np.arange(38, 50)), axis = 0 )
        data        = data[ :, valid_idx ]
        data        = np.array( [Series(i).interpolate() for i in data.T] ).T

        # min-max normalization
        diff = upperBound - lowerBound
        data = 2 * (data - lowerBound) / diff - 1

        data[ data > 1 ]    = 1.0
        data[ data < -1 ]   = -1.0

        # sliding window
        data    = sliding_window( data, (args.length, channel_num), (args.overlap, 1) )
        label   = sliding_window( label, args.length, args.overlap )
        label   = stats.mode( label, axis=1 )[0][:,0]

        # remove non-interested time windows (label==-1)
        invalid_idx = np.nonzero( label < 0 )[0]
        data        = np.delete( data, invalid_idx, axis=0 )
        label       = np.delete( label, invalid_idx, axis=0 )

        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), data )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), label )
        print( "sub{} finished".format( usr_idx) )


def main():
    
    if args.dataset == 'opp':
        preprocess_opportunity()
    else:
        preprocess_pamap2()

if __name__ == '__main__':
    main()