train_seq_file = open('./train_p5.txt', 'r')
test_seq_file = open('./test_p5.txt', 'r')

new_test_file = open('./sequential_data_test.txt', 'w')

train_seq_dict = {}
for line in train_seq_file:
    uid = line.strip().split(' ')[0]
    train_seq_dict[uid] = line.strip().split(' ')[1:]


test_seq_dict = {}
for line in test_seq_file:
    uid = line.strip().split(' ')[0]
    test_seq_dict[uid] = line.strip().split(' ')[1:]

for uid in test_seq_dict.keys():
    train_seq = train_seq_dict[uid]
    test_seq = test_seq_dict[uid]

    train_seq.insert(0, uid)

    for ele in test_seq:
        train_seq.append(ele)
        test_ele = ' '.join([str(elem) for elem in train_seq])
        new_test_file.write(test_ele)
        new_test_file.write('\n')

