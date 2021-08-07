from unaligned_data_loader_opp import UnalignedDataLoader
import data
import numpy as np

def one_hot( y, n_values ):
        return np.eye( n_values )[ np.array( y, dtype = np.int32 ) ]

def return_dataset(test_user, dataset):
    if dataset == 'opp':
        _dataset = data.Opportunity()
        print("Test Dataset is Opportunity")
    else:
        _dataset = data.PAMAP2()
        print("Test Dataset is PAMAP2")
        
    train_x, train_y, test_x, test_y = _dataset.load_data(test_user)
    print('user {} is the test set'.format(test_user))
    train_x     = train_x.astype( np.float32 )
    train_ya    = train_y 
    train_yu    = np.full(train_ya.shape[0], 0) 
    
    test_x      = test_x.astype( np.float32 )
    test_ya     = test_y
    test_yu     = np.full(test_ya.shape[0], 1) 

    adapt_size  = int( test_x.shape[0] * 0.5 )
    adapt_x = test_x[: adapt_size]
    adapt_ya = test_ya[: adapt_size]
    adapt_yu = test_yu[: adapt_size]

    test_x = test_x[adapt_size:]
    test_ya = test_ya[adapt_size:]
    test_yu = test_yu[adapt_size:]

    return train_x, train_ya, train_yu, adapt_x, adapt_ya, adapt_yu, test_x, test_ya, test_yu

def dataset_read(batch_size, test_user=0, dataset = 'opp'): 
    S = {}
    S_test = {}
    T = {}
    T_test = {}

    train_x, train_ya, train_yu, adapt_x, adapt_ya, adapt_yu, test_x, test_ya, test_yu = return_dataset(test_user, dataset)
    
    train_source = train_x
    s_label_train = train_ya
    test_source = train_x
    s_label_test = train_ya
    
    train_target = adapt_x
    t_label_train = adapt_ya
    test_target = test_x
    t_label_test = test_ya


    S['imgs'] = train_source
    S['labels'] = s_label_train
    T['imgs'] = train_target
    T['labels'] = t_label_train

    S_test['imgs'] = test_source
    S_test['labels'] = s_label_test
    T_test['imgs'] = test_target
    T_test['labels'] = t_label_test

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size)
    dataset_test = test_loader.load_data()
    return dataset, dataset_test
