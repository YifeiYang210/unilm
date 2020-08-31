def line2column():
    with open("pred1.txt", "r", encoding="utf-8") as f, open(
            "pred2.txt", "w", encoding="utf-8"
    ) as f2:
        for line in f:
            labels = line.split()
            for label in labels:
                f2.write(label + "\n")
            f2.write("\n")
    print("line2column success")


def write_labels():
    with open("pred2.txt", "r", encoding="iso8859-1") as f, open(
            "test.txt", "r", encoding="iso8859-1"
    ) as f2, open("pred3.txt", "w", encoding="iso8859-1") as f3:
        labels = f.readlines()
        entities = f2.readlines()
        for k, v in enumerate(entities):
            line = v.strip()
            if not line:
                f3.write("\n")
                continue
            f3.write(v.split("\t")[0] + "\t" + labels[k])
    print("write_labels success!")

def write_entities():
    with open("pred3.txt", "r", encoding="iso8859-1"
    ) as f, open("pred4.txt", "w", encoding="iso8859-1") as f4:
        for v in f:
            line = v.strip()
            if not line:
                f4.write("\n")
                continue
            entity = line.split("\t")[0]
            label = line.split("\t")[1]
            if label != "O":
                f4.write(entity + "\t" + label + "\n")
    print("write_entities success!")


if __name__ == '__main__':
    # write_labels()
    write_entities()
