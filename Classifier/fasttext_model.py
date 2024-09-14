from argparse import ArgumentParser
import os, fasttext


def fasttext_model(opts):
    data_train_path = os.path.join(opts.data_train_test_dir,'data_train.txt')
    model = fasttext.train_supervised(data_train_path,
                                      wordNgrams = 2,
                                      dim = 200,
                                      loss = 'hs',
                                      epoch = 50)
    model_path = os.path.join(opts.model_dir, opts.model_name)
    model.save_model(model_path)

    model = fasttext.load_model(model_path)
    data_test_path = os.path.join(opts.data_train_test_dir,'data_test.txt')
    test = model.test(data_test_path, k=-1, threshold=0.5)
    precision = test[1]
    print(f'\n{model_path}\n')
    print(f'precision: {precision}\n')

    with open('./model_train.txt', 'w') as file:
        file.write(f'\n{model_path}\n')
        file.write(f'precision: {precision}\n')


if __name__ == '__main__':
    parser = ArgumentParser(description='Prepare train and test dataset for fasttext.')
    parser.add_argument('-dd', '--data_train_test_dir', type=str, help='data_train_test_dir')
    parser.add_argument('-md', '--model_dir', type=str, help='model_dir')
    parser.add_argument('-mn', '--model_name', type=str, help='model_name')
    opts = parser.parse_args()

    if not os.path.exists(opts.model_dir):
        os.mkdir(opts.model_dir)

    fasttext_model(opts)
