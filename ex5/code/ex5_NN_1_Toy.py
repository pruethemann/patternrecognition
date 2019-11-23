import sys
import torch


def toyNetwork() -> None:
    # TODO: Implement network as given in the exercise sheet.
    # Manual implementation functionality when computing loss, gradients and optimization
    # i.e. do not use torch.optim or any of the torch.nn functionality
    # Torch documentation: https://pytorch.org/docs/stable/index.html

    ### Define weight variables using: torch.tensor([], requires_grad=True)
    weights = [0.5, 0.3, 0.3, 0.1, 0.8, 0.3, 0.5, 0.9, 0.2]
    w = [ torch.tensor([w], requires_grad=True)  for w in weights ]
    print(w)

    b = [torch.tensor([1.0]) , torch.tensor([1.0]) ]

    ### Define data: x, y using torch.tensor
    x = [torch.tensor([1.0]), torch.tensor([1.0])]
    y_true = torch.tensor([1.0])

    # TODO: Define learning rate

    # TODO: Train network until convergence
    # TODO: Define network forward pass connectivity
    # TODO: Get gradients of weights and manually update the network weights

    # Steps:
    # 1 - compute error
    # 2 - do backward propagation, use: error.backward() to do so
    # 3 - update weight variables according to gradient and learning rate
    # 4 - Zero weight gradients with w_.grad_zero_()
    loss_before = 10
    loss = 1
    learning_rate = 0.2
    while abs(loss_before - loss) > 0.001:
        loss_before = loss

        h1 = torch.sigmoid( x[0] * w[0] + x[1] * w[2] + b[0] * w[4]  )
        h2 = torch.sigmoid( x[0] * w[1] + x[1] * w[3] + b[0] * w[5] )
        y = torch.sigmoid( h1 *w[6] + h2 * w[7] + b[1] * w[8]  )

        loss = 0.5 *(y_true - y)**2

        print(f'h1: {h1.item()} h2: {h2.item()} y: {y.item()}')
        print(f'Loss: {loss.item()}')

        loss.backward()

        with torch.no_grad():
            for i in range(len(w)):
                w[i] -= learning_rate * w[i].grad
                w[i].grad.zero_()
                print(f'New w{i+1}: {w[i].item()}')

        print("################### NEXT ROUND ###################")




if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network toy example!")
    toyNetwork()
    print("Done!")
