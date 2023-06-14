from training import*


def main():
    """

    :return:
    """

    print("Constructing training data...")
    tensor, syllables = construct_data()

    print("Beginning training...")
    model, history = netTrain(tensor=tensor,syllables=syllables, epochs=100, batch_size=32, n_channels=32)
    del tensor, syllables

    print("Constructing testing data...")
    tensor, syllables = construct_data(languages=["Estonian"], test=True, root=testing_loc)

    print("Beginning training...")
    netTest(tensor=tensor, syllables=syllables, model=model, batch_size=32)
    del tensor, syllables

    print("Constructing testing data...")
    tensor, syllables = construct_data(languages=["English"], test=True, root=testing_loc)

    print("Beginning training...")
    netTest(tensor=tensor, syllables=syllables, model=model, batch_size=32)
    del tensor, syllables

    return


main()
