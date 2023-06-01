test_file = open('./test_p5.txt', 'r')
new_test_file = open('./negative_samples.txt', 'w')

for line in test_file:
    test_elemnets = line.strip().split(' ')[:2]
    test_ele = ' '.join([str(elem) for elem in test_elemnets])
    new_test_file.write(test_ele)

    new_test_file.write('\n')




