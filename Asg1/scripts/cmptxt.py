from filecmp import cmp

fa = "/home/z7sheng/CS686/cs686-repo/Asg1/heu_out/Jam-all-10.txt"
fb = "/home/z7sheng/CS686/cs686-repo/Asg1/adv_out/Jam-all-80.txt"

# fa = "/home/z7sheng/CS686/cs686-repo/Asg1/cmp_out/blo_00.txt"
# fb = "/home/z7sheng/CS686/cs686-repo/Asg1/cmp_out/adv_00.txt"

ret = cmp(fa, fb)
print(ret)