from scoop import futures

def product(x, y):
    """Return the product of the arguments"""
    return x+y

def sum(x, y):
    """Return the sum of the arguments"""
    return x+y

if __name__ == "__main__":

    a = range(1,101)
    b = range(101, 201)

    results = futures.map(product, a, b)

    total = reduce(sum, results)

    print("Sum of the products equals %d" % total)
