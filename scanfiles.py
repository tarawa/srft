if __name__ == "__main__":
    import os
    import re
    import csv

    path = os.getcwd() + "/build/"
    # fileformat = r"^slurm-[0-9]{8}\.out$"
    rowformat = r"^Simulation Time = ([0-9]+\.[0-9]+) seconds for arr of size ([0-9]+) using transform ([a-z]{3,4}) with seed ([0-9]+) and d ([0-9]+) and rank ([0-9]+)\.$"

    transform_type = ["fwht", "dft", "idft"]
    parallel_type = ["serial", "openmp", "gpu"]

    for tt in transform_type:
        for pt in parallel_type:
            with open("data_" + tt + "-" + pt + ".csv", "w", newline="") as csvfile:
                datawriter = csv.writer(csvfile, delimiter = ",")
                datawriter.writerow(["Array_Size", "Rank", "Time", "Subsample_Size"])
                filepath = "build/" + tt + "-" + pt + ".out"
                try:
                    f = open(filepath, "r")
                    data = f.read()
                    for line in data.splitlines():
                        if re.match(rowformat, line):
                            result = re.search(rowformat, line)
                            dwr = [result.group(2), result.group(6), result.group(1), result.group(5)]
                            datawriter.writerow(dwr)
                    f.close()
                except:
                    print(filepath + " does not exist. Continuing...")
