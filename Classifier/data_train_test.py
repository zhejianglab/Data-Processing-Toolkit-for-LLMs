import os, json
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split


def data_train_test(opts):
    data_dir = opts.data_dir
    output_dir = opts.output_dir
    subject_name = opts.subject_name

    files = os.listdir(data_dir)
    files.sort()

    subject_filter = [subject_name+'.json']

    wos_stat = []
    data_with_label = []
    for i, file in enumerate(files):
        if i>=0:
            filein = os.path.join(data_dir, file)
            
            with open(filein, 'r') as fi:
                lines = fi.readlines()
                numlines = len(lines)

                if file in subject_filter:
                    numlines_filter = numlines
                else:
                    numlines_filter = 10000

                if numlines >= numlines_filter:
                    print(f'{i}: {file}, lines= {numlines_filter}')
                    wos_stat.append(f'{file} [lines:{numlines_filter}]\n')

                    for num in range(0,numlines_filter):
                        line = lines[num]
                        line_dict = json.loads(line)
                        abstract = line_dict['Abstract']

                        if abstract != '':
                            if file in subject_filter:
                                labels = ['__label__'+subject_name]
                            else:
                                labels = ['__label__Other_Subjects']

                            tags = " ".join(labels)
                            newline = f"{abstract}" + " " + tags + "\n"
                            data_with_label.append(newline)


    with open('./wos_stat.txt', 'w') as file:
        for data in wos_stat:
            file.write(str(data))


    data_train, data_test = train_test_split(data_with_label, shuffle = True, test_size = 0.2, random_state = 123)
    with open(os.path.join(output_dir, 'data_train.txt'), 'w') as file:
        for data in data_train:
            file.write(str(data))
    with open(os.path.join(output_dir, 'data_test.txt'), 'w') as file:
        for data in data_test:
            file.write(str(data))


if __name__ == '__main__':
    parser = ArgumentParser(description='Prepare train and test dataset for fasttext.')
    parser.add_argument('-dd', '--data_dir', type=str, help='data_dir')
    parser.add_argument('-od', '--output_dir', type=str, help='output_dir')
    parser.add_argument('-n', '--subject_name', type=str, help='subject_name')
    opts = parser.parse_args()

    if not os.path.exists(opts.output_dir):
        os.mkdir(opts.output_dir)

    data_train_test(opts)
