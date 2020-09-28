

file = open("shakespeare_dataset.txt", "r")

train_data = []
test_data = []
all_data = []


def uploadData(size):
    for i in range(size):
        line = file.readline()
        train_data.append(line)
        all_data.append(line)

    # for i in range(size + 1, 2 * size):
    #     line = file.readline()
    #     test_data.append(line)
    #     all_data.append(line)
