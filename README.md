# DGM

Attempt to reproduce the results from https://arxiv.org/pdf/1708.07469.pdf
The merton problem is corretly solved, burger's one isnt, feel free to fill an issue
The loss function and training have to be rewrite for each PDE

# sampler
Just a little class to create easily samples on the domain
TODO : find a way to make it more "natural"

# first_net
The implementation of the architecture propose in the paper above, with modified linear layers to have a propre xavier init
Using the regular pytorch layers leads prevent the model to converge properly.
TODO : figure out how to initialise the bias to converge faster


# Not working
The model in burger does not converge, try different hyper parameters

# TODO
remove all the useless testing stuff


