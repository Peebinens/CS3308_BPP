import csv
from collections import defaultdict

# read data from csv file
def clean_csv(filename, container_set):
    sort_container_set = container_set.copy()
    for container in sort_container_set:
        container.sort()
    print(sort_container_set)
    grouped_data = defaultdict(list)
    error_sta_code = [] 
    with open(filename, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # print(row)
            sta_code = row['sta_code']
            now_box = [float(row['长(CM)']), float(row['宽(CM)']), float(row['高(CM)'])]
            temp_container = sort_container_set.copy()
            now_box.sort()
            for container in container_set:
                if now_box[0] > container[0] or now_box[1] > container[1] or now_box[2] > container[2]:
                    temp_container.remove(container)
            if len(temp_container) == 0:
                error_sta_code.append(sta_code)
    return error_sta_code

def rewrite_csv(old_filename, new_filename, container_set):
    error_sta_code = clean_csv(old_filename, container_set)
    with open(old_filename, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        with open(new_filename, mode='w', newline='', encoding='utf-8-sig') as new_file:
            writer = csv.DictWriter(new_file, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row['sta_code'] not in error_sta_code:
                    writer.writerow(row)


if __name__ == "__main__":
    container_set = [[35, 23, 13], [37, 36, 13], [38, 26, 13], [40, 28 ,16], [42, 30, 18], [42, 30, 40], [52, 40, 17], [54, 45, 36]]
    rewrite_csv('data.csv', 'data_clean.csv', container_set)

