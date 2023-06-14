from training import*


def main():
    """

    :return:
    """
    tensor, syllables = construct_data(samples=1250)
    model, history = netTrain(tensor=tensor,syllables=syllables, epochs=100, batch_size=32, n_channels=32)
    del tensor, syllables

    tensor, syllables = construct_data(languages=["Estonian"], test=True, root=testing_loc)
    mae, mape = netTest(tensor=tensor, syllables=syllables, model=model, batch_size=32)
    del tensor, syllables

    print("Estonian MAE:", mae)
    print("Estonian MAPE:", mape)

    tensor, syllables = construct_data(languages=["English"], test=True, root=testing_loc)
    mae, mape = netTest(tensor=tensor, syllables=syllables, model=model, batch_size=32)
    del tensor, syllables

    print("English MAE:", mae)
    print("English MAPE:", mape)

    return


main()
