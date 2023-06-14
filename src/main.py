from training import*


def main():
    """

    :return:
    """

    print("Constructing training data...")
    tensor, syllables = construct_data(samples=10000)

    print("Beginning training...")
    model, history = netTrain(tensor=tensor,syllables=syllables, epochs=100 , batch_size=32, n_channels=32)
    del tensor, syllables

    print("Constructing testing data...")
    tensor, syllables = construct_data(languages=["Estonian"], test=True, root=testing_loc)

    print("Beginning testing...")
    netTest(tensor=tensor, syllables=syllables, model=model, batch_size=32, language="Estonian")
    del tensor, syllables

    print("Constructing testing data...")
    tensor, syllables = construct_data(languages=["English"], test=True, root=testing_loc)

    print("Beginning testing...")
    netTest(tensor=tensor, syllables=syllables, model=model, batch_size=32, language="English")
    del tensor, syllables

    return


main()
