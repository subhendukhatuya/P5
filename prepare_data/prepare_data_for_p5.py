import os
import numpy as np
import pickle

def save(x, write_file):

    with open(write_file, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(file):
    with open(file,'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    return data


def format(folder_src, folder_dest,exogenous_events):

    test = load_data(folder_src + "test.pkl")['test']
    train = load_data(folder_src + "train.pkl")['train']
    # exit()
    train_dict,mn,mx = {},[],[]
    for seq in train:
        # print(seq[0])
        # exit()
        user = seq[0]['user_emb']
        all_valid_events = [x['type_event'] for x in seq if x['type_event'] not in exogenous_events]
        if len(all_valid_events) > 0:
            train_dict[user] = all_valid_events

    test_dict={}
    for seq in test:
        user = seq[0]['user_emb']
        all_valid_test_events = [x['type_event'] for x in seq if x['type_event'] not in exogenous_events ]
        if len(all_valid_test_events) > 0:
            test_dict[user] = all_valid_test_events

    print(len(train_dict.keys()))
    print(len(test_dict.keys()))
    # exit()

    all_users = sorted(train_dict.keys())

    #print(all_users[:10])
    # exit()

    print('all users',len(all_users))

    test_user=[]
    for user in all_users:
        if user not in test_dict:
            test_user += [False]
            test_dict[user] = [0]
        else:
            test_user += [True]
    print(np.where(np.array(test_user)==False)[0].shape[0])#count_nnz(test_user))
    # exit()

    final_train_list, final_test_list = [],[]
    count_test = 0
    for user,flag in zip(all_users, test_user):
        final_train_list += [[user] + train_dict[user]]

        if flag:
            test_seq_length = int(len(test_dict[user])*0.70)
            if test_seq_length > 0:
                #print('user', user)
                count_test = count_test+1
            final_test_list += [[user] + test_dict[user][test_seq_length:]]
        #print(len(train_dict[user]), len(test_dict[user]), flag )
    # exit()

    print('count_test', count_test)



    if not os.path.exists(folder_dest):
        os.mkdir(folder_dest)

    def save_txt(x_list, write_file,exogenous_events):

        with open(write_file+'.txt','w') as fw:
            # i=0
            for seq in x_list:
                print_str=''
                i=0
                for event in seq:
                    if True:
                        i+=1
                        print_str = print_str + str(event) + ' '
                if i < 2:
                    print('len',i)
                if print_str=='':
                    print('error check')




                # print_str = print_str.strip()
                # print_str = ' '.join([str(z) for z in x])
                fw.write(print_str.strip() + '\n')
                # i+=1
                # if i==10:
                    # break

    save_txt(final_train_list, folder_dest + 'train_p5',exogenous_events)
    save_txt(final_test_list, folder_dest + 'test_p5',exogenous_events)
    save(test_user, folder_dest + 'test_user_list.pkl')



folder_src = 'dunnhumby_formatted2/'
folder_dest= 'dunnhumby_test_p5/'
exogenous_events = [x for x in range(44,88)]

# folder_src = 'stackoverflow_formatted2/'
# folder_dest= 'stackoverflow_formatted2_p5/'
# exogenous_events = [3]


# folder_src = 'stackoverflow_formatted_seq_len_500/'
# folder_dest= 'stackoverflow_formatted_seq_len_500_p5/'
# exogenous_events = [3]
#folder_src = 'mooc1_formatted/'
#folder_dest= 'mooc1_formatted_p5/'
#exogenous_events = [0,23]


#folder_src = 'mooc2_formatted/'
#folder_dest= 'mooc2_formatted_p5/'
#exogenous_events = [0,23]


#folder_src = 'moodle_formatted_scaled/'
#folder_dest= 'moodle_formatted_scaled_p5/'
#exogenous_events = [x for x in range(39,54)]



# folder_src = 'synthetic_formatted/'
# folder_dest= 'synthetic_formatted_p5/'
# exogenous_events = [2,3]#x for x in range(39,54)]


format(folder_src, folder_dest,exogenous_events)
