import os
import sys
import csv

def csv_writer(data, path):

    with open(path+".csv", "w", newline='',encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)




def collect_data(path):
    name_out = 0
    items = os.listdir(path)
    anek_list = []
    for item in items:
        with open(path + "/"+ item, encoding='utf-8') as f:
            print("started with " + item)
            lines = f.readlines()
            i = 0
            while i != len(lines):
                if "====================================================================================================" in lines[i]:
                    i = i + 4
                    anek = []
                    while "====================================================================================================" not in lines[i]:
                        anek.append(lines[i].rstrip())
                        i += 1

                    if "[фотография]" not in " ".join(anek):
                        if "http" not in " ".join(anek):
                            if "----------------------------------------------------------------------------------------------------" not in " ".join(anek):
                                anek_list.append([len(anek_list)," ".join(anek)])

                    i+=2



        if len(anek_list) >= 10000:
            csv_writer(anek_list, str(name_out))
            name_out += 1
            anek_list.clear()

    if len(anek_list) > 0:
        csv_writer(anek_list, str(name_out))
        name_out += 1
        anek_list.clear()

if __name__ == "__main__":
    data_path = sys.argv[1]
    collect_data(data_path)
