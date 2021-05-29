import csv, itertools, math, random
import numpy as np
import scipy.cluster.hierarchy as scihac
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename) as csvfile:
        pokemon = []

        reader = itertools.islice(csv.DictReader(csvfile), 20)
        
        for row in reader:
            row.pop("Generation")
            row.pop("Legendary")
            row["#"] = int(row["#"])
            row["Total"] = int(row["Total"])
            row["HP"] = int(row["HP"])
            row["Attack"] = int(row["Attack"])
            row["Defense"] = int(row["Defense"])
            row["Sp. Atk"] = int(row["Sp. Atk"])
            row["Sp. Def"] = int(row["Sp. Def"])
            row["Speed"] = int(row["Speed"])

            pokemon.append(row)
        

    return pokemon

def calculate_x_y(stats):

    x = stats["Attack"] + stats["Sp. Atk"] + stats["Speed"]
    y = stats["Defense"] + stats["Sp. Def"] + stats["HP"]

    return (x, y)

def hac(dataset):
    # Remove invalid tuples
    ds = []
    for (x,y) in dataset:
        if math.isfinite(x) and math.isfinite(y):
            ds.append((x, y))
    dataset = ds

    # Number of tuples given
    m = len(dataset)
    # Keep track of how many clusters are made
    clusterCount = 0

    # This is a list of dictionaries. Each row has a point and its current associated
    # cluster number and cluster size.
    origPoints = []
    # Fill in origPoints
    for (x, y) in dataset:
        row = {}

        row["p"] = (x, y)
        row["cluster"] = clusterCount
        row["cSize"] = 1

        clusterCount += 1

        origPoints.append(row)

    # List of dictionaries in ascending order of distance, with tiebreaker implemented.
    # Each row has both points, their original cluster numbers, and their euclidean distance.
    orderedDist = []
    # Fill in orderedDist
    for i in range(m):
        for j in range(m):
            if i<j:
                dict = {}

                for row in origPoints:
                    if row["cluster"] == i:
                        dict["p1"] = row["p"]
                        dict["p1origin"] = row["cluster"]
                    if row["cluster"] == j:
                        dict["p2"] = row["p"]
                        dict["p2origin"] = row["cluster"]
                    
                dict["dist"] = math.dist(dict["p1"], dict["p2"])

                orderedDist.append(dict)
    # Order distances in ascending, tiebroken order
    orderedDist = sorted(orderedDist, key = lambda dict: (dict["dist"], origPoints[dict["p1origin"]]["cluster"], origPoints[dict["p2origin"]]["cluster"]))

    # The matrix to be returned
    Z = np.zeros((m-1, 4))
    # Keep track of which row in Z is being filled in
    Zcount = 0
    # Fill in Z
    while (Zcount < m - 1):
        # The first row in orderedDist has the smallest, tie-broken, distance
        dict = orderedDist[0]

        # Row numbers of the points in origPoints
        p1origin = dict["p1origin"]
        p2origin = dict["p2origin"]

        # Remove any intra-cluster distances that show up
        if origPoints[p1origin]["cluster"] == origPoints[p2origin]["cluster"]:
            orderedDist.remove(dict)
            continue

        #Update Z
        if origPoints[p1origin]["cluster"] < origPoints[p2origin]["cluster"]:
            Z[Zcount][0] = origPoints[p1origin]["cluster"]
            Z[Zcount][1] = origPoints[p2origin]["cluster"]
        else:
            Z[Zcount][0] = origPoints[p2origin]["cluster"]
            Z[Zcount][1] = origPoints[p1origin]["cluster"]
        Z[Zcount][2] = dict["dist"]
        Z[Zcount][3] = origPoints[p1origin]["cSize"] + origPoints[p2origin]["cSize"]

        # Update cluster info in origPoints
        for row in origPoints:
            if (row["cluster"] == Z[Zcount][0]) or (row["cluster"] == Z[Zcount][1]):
                row["cluster"] = clusterCount
                row["cSize"] = Z[Zcount][3]

        # Remove used distance
        orderedDist.remove(dict)
        
        # Ensure smaller cluster is first
        for dict in orderedDist:
            if origPoints[dict["p1origin"]]["cluster"] > origPoints[dict["p2origin"]]["cluster"]:
                t = dict["p1origin"]
                dict["p1origin"] = dict["p2origin"]
                dict["p2origin"] = t

                (t1, t2) = dict["p1"]
                dict["p1"] = dict["p2"]
                dict["p2"] = (t1, t2)

        # Order distances in ascending, tiebroken order
        orderedDist = sorted(orderedDist, key = lambda dict: (dict["dist"], origPoints[dict["p1origin"]]["cluster"], origPoints[dict["p2origin"]]["cluster"]))

        # Update counts
        clusterCount += 1
        Zcount += 1

    # Convert to numpy matrix
    Z = np.asmatrix(Z)
    return Z

def random_x_y(m):
    arr = []

    for i in range(m):
        x = random.randint(1, 359)
        y = random.randint(1, 359)
        arr.append((x,y))

    return arr



def imshow_hac(dataset):
    # Remove invalid tuples
    ds = []
    for (x,y) in dataset:
        if math.isfinite(x) and math.isfinite(y):
            ds.append((x, y))
    dataset = ds

    # Plot the original points
    fig = plt.figure()
    ax = fig.add_subplot()
    for (x, y) in dataset:
        ax = plt.scatter(x, y)
    plt.pause(0.1)

    # Number of tuples given
    m = len(dataset)
    # Keep track of how many clusters are made
    clusterCount = 0

    # This is a list of dictionaries. Each row has a point and its current associated
    # cluster number and cluster size.
    origPoints = []
    # Fill in origPoints
    for (x, y) in dataset:
        row = {}

        row["p"] = (x, y)
        row["cluster"] = clusterCount
        row["cSize"] = 1

        clusterCount += 1

        origPoints.append(row)

    # List of dictionaries in ascending order of distance, with tiebreaker implemented.
    # Each row has both points, their original cluster numbers, and their euclidean distance.
    orderedDist = []
    # Fill in orderedDist
    for i in range(m):
        for j in range(m):
            if i<j:
                dict = {}

                for row in origPoints:
                    if row["cluster"] == i:
                        dict["p1"] = row["p"]
                        dict["p1origin"] = row["cluster"]
                    if row["cluster"] == j:
                        dict["p2"] = row["p"]
                        dict["p2origin"] = row["cluster"]
                    
                dict["dist"] = math.dist(dict["p1"], dict["p2"])

                orderedDist.append(dict)
    # Order distances in ascending, tiebroken order
    orderedDist = sorted(orderedDist, key = lambda dict: (dict["dist"], origPoints[dict["p1origin"]]["cluster"], origPoints[dict["p2origin"]]["cluster"]))

    # The matrix to be returned
    Z = np.zeros((m-1, 4))
    # Keep track of which row in Z is being filled in
    Zcount = 0
    # Fill in Z
    while (Zcount < m - 1):
        # The first row in orderedDist has the smallest, tie-broken, distance
        dict = orderedDist[0]

        # Row numbers of the points in origPoints
        p1origin = dict["p1origin"]
        p2origin = dict["p2origin"]

        # Remove any intra-cluster distances that show up
        if origPoints[p1origin]["cluster"] == origPoints[p2origin]["cluster"]:
            orderedDist.remove(dict)
            continue

        #Update Z
        if origPoints[p1origin]["cluster"] < origPoints[p2origin]["cluster"]:
            Z[Zcount][0] = origPoints[p1origin]["cluster"]
            Z[Zcount][1] = origPoints[p2origin]["cluster"]
        else:
            Z[Zcount][0] = origPoints[p2origin]["cluster"]
            Z[Zcount][1] = origPoints[p1origin]["cluster"]
        Z[Zcount][2] = dict["dist"]
        Z[Zcount][3] = origPoints[p1origin]["cSize"] + origPoints[p2origin]["cSize"]

        # Update cluster info in origPoints
        for row in origPoints:
            if (row["cluster"] == Z[Zcount][0]) or (row["cluster"] == Z[Zcount][1]):
                row["cluster"] = clusterCount
                row["cSize"] = Z[Zcount][3]

        # Plot each cluster
        (x1, y1) = origPoints[p1origin]["p"]
        (x2, y2) = origPoints[p2origin]["p"]
        ax = plt.plot((x1, x2), (y1, y2))
        plt.pause(0.1)

        # Remove used distance
        orderedDist.remove(dict)
        
        # Ensure smaller cluster is first
        for dict in orderedDist:
            if origPoints[dict["p1origin"]]["cluster"] > origPoints[dict["p2origin"]]["cluster"]:
                t = dict["p1origin"]
                dict["p1origin"] = dict["p2origin"]
                dict["p2origin"] = t

                (t1, t2) = dict["p1"]
                dict["p1"] = dict["p2"]
                dict["p2"] = (t1, t2)

        # Order distances in ascending, tiebroken order
        orderedDist = sorted(orderedDist, key = lambda dict: (dict["dist"], origPoints[dict["p1origin"]]["cluster"], origPoints[dict["p2origin"]]["cluster"]))

        # Update counts
        clusterCount += 1
        Zcount += 1

    # Display our clustering process
    plt.show()

    # Convert to numpy matrix
    Z = np.asmatrix(Z)
    return Z

def main():

    pokemon = load_data("Pokemon.csv")
    
    # arr = []
    # for row in pokemon:
    #     (x, y) = calculate_x_y(row)
    #     arr.append((x,y))

    arr = random_x_y(50)

    # print("points:\n")
    # print(arr)
    print("\n\n my hac:\n")
    print(imshow_hac(arr))
    #imshow_hac(arr)

    #arr = np.array(arr)
    print("\n\nscihac:\n")
    print(scihac.linkage(arr))

if __name__=="__main__": 
    main()