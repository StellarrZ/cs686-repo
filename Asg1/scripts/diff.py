import os


path = "/home/z7sheng/CS686/cs686-repo/Asg1/cmp_out/"
num = "01"
bloName = "blo_" + num + ".txt"
advName = "adv_" + num + ".txt"

with open(os.path.join(path, bloName)) as fh:
    bloVal = list(map(lambda x: int(x), fh.readlines()))

with open(os.path.join(path, advName)) as fh:
    advVal = list(map(lambda x: int(x), fh.readlines()))

print("jams\t blo_h\t adv_h\t diff\n")
for k in range(len(bloVal)):
    b, a = str(bloVal[k]), str(advVal[k])
    print("Jam-" + str(k + 1) + "\t", b + "\t", a + "\t", bloVal[k] - advVal[k])